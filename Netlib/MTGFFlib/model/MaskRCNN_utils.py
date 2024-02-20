import torch
import torch.nn as nn
from typing import List, OrderedDict
from torch.nn.functional import one_hot,softmax,binary_cross_entropy_with_logits,cross_entropy,grid_sample
from torchvision.ops import roi_align, box_iou,nms,box_convert,clip_boxes_to_image,remove_small_boxes


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, threshold: float = 0.5):
    "reutrn n,img_h,imgw"
    device = masks.device  # N, 28, 28 or N, 1, 28, 28
    masks = masks[:, None] if masks.dim() == 3 else masks
    masks = masks.float()

    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)  # (N, h, w, 2)

    img_masks = grid_sample(
        masks, grid.to(masks.dtype), align_corners=False
    )  # N,1, h, w
    img_masks = img_masks >= threshold
    return img_masks.squeeze(1).long()

@torch.no_grad()
def anchors_to_max_boxes_delta(anchors: torch.Tensor, boxes: torch.Tensor, indexes: torch.Tensor, weights: List[float] = [1.0,1.0,1.0,1.0]):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=anchors.device)
    target_boxes = boxes[indexes]
    target_boxes = box_convert(target_boxes, "xyxy", "cxcywh")
    anchors = box_convert(anchors, "xyxy", "cxcywh")
    reg_xy_delta = (target_boxes[:, :2] - anchors[:, :2]) / target_boxes[:, 2:] 
    reg_wh_delta = torch.log(target_boxes[:, 2:] / anchors[:, 2:])
    return torch.cat([reg_xy_delta*weights[0:2] , reg_wh_delta*weights[2:] ], dim=-1)


def apply_box_delta(anchors: torch.Tensor, reg: torch.Tensor, box_weight = [1.0,1.0,1.0,1.0]):
    # 将锚框根据预测变换为检测框形式
    if not isinstance(box_weight, torch.Tensor):
        box_weight = torch.tensor(box_weight, device=anchors.device)
    anchors = box_convert(anchors, "xyxy", "cxcywh")
    anchors[:, :2] = anchors[:, :2] + anchors[:, 2:] * (reg[:, :2]/box_weight[:2])
    anchors[:, 2:] = anchors[:, 2:] * torch.exp(reg[:, 2:]/box_weight[2:])
    return box_convert(anchors, "cxcywh", "xyxy")

def remove_big_boxes(boxes: torch.Tensor, size: float) -> torch.Tensor:
    # 去除大框
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    keep = (h < size) & (w < size)
    return keep

@torch.no_grad()
def generate_detection_targets(
    batched_rois: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    target_cls: List[torch.Tensor],
    target_masks: List[torch.Tensor],
    out_size: int,
    iou_thresh: float=0.3,
    box_weight = [10.0,10.0,5.0,5.0]
):
    # todo: 消除unbalanced cls
    batched_cls = []
    batched_reg = []
    batched_masks = []

    for rois, boxes, cls, masks in zip(
        batched_rois, target_boxes, target_cls, target_masks
    ):
        # (rois,4),(n,4),(n,),(n,28,28)

        if 0 in rois.shape:
            continue
        if 0 in boxes.shape:
            reg = torch.zeros_like(rois, device=rois.device)
            cls = torch.zeros(len(rois), dtype=torch.long, device=rois.device)
            masks = torch.zeros(
                len(rois), out_size, out_size, dtype=torch.long, device=rois.device
            )
            batched_cls.append(cls)
            batched_reg.append(reg)
            batched_masks.append(masks)
            continue
        iou = box_iou(rois, boxes)
        max_iou, indexes = torch.max(iou, dim=-1)
        cls_bool = max_iou < 0.3

        cls = cls[indexes]
        masks = masks[indexes]
        cls[cls_bool] = 0

        # print(f'stage2 pos cls label:{cls!=}')
        reg = anchors_to_max_boxes_delta(rois, boxes, indexes, weights=box_weight)
        batched_cls.append(cls)
        batched_reg.append(reg)
        batched_masks.append(masks)
    return batched_cls, batched_reg, batched_masks

@torch.no_grad()
def generate_mul_anchors(
    anchors_per_bin: torch.Tensor,
    img_size,
    scale=[4, 8],
    aspect_ratio=[0.5, 1, 2],
    size_range=[16, 32, 64],
):
    r"中心对齐：(x+0.5)*scale-0.5, 生成锚框坐标(k,h,w,4)"
    # to do: 生成不同尺度的锚框
    mul_anchors = []
    anchors_per_bin = anchors_per_bin.unsqueeze(1).unsqueeze(1)  # k,1,1,2
    for i, s in enumerate(scale):
        # to do: 生成不同长宽比的锚框
        # size = (-1,size_range[i]) if i == 0 else (size_range[i-1],size_range[i])
        # 筛选在size范围内的anchors
        # _anchors_per_bin = anchors_per_bin[(anchors_per_bin[...,0] >= size[0]) & (anchors_per_bin[...,0] <= size[1])] # k,1,1,2 dtype:float
        anchors = torch.zeros(len(anchors_per_bin), img_size // s, img_size // s, 4, device=anchors_per_bin.device)
        x_coordinate, y_coordinate = torch.meshgrid(
            [torch.arange(0.5 * s - 0.5, img_size, step=s,device=anchors_per_bin.device) for _ in range(2)],
            indexing="xy",
        )
        xy = torch.stack([x_coordinate, y_coordinate], dim=-1).unsqueeze(0)
        anchors[..., :2] = xy - anchors_per_bin / 2
        anchors[..., 2:] = xy + anchors_per_bin / 2
        mul_anchors.append(anchors)
    return mul_anchors  # k,h,w,4


def balanced_pos_neg_sample(pos: torch.Tensor, neg: torch.Tensor, sample_ratio=0.3):
    prob = pos.sum() / (neg.sum() * sample_ratio)
    neg = neg * (torch.rand(*neg.shape, device=neg.device) <= prob)
    return neg


class bn_act_conv(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size,
        stride=1,
        padding=0,
        active_f: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_c),
            active_f(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        return self.conv(x)

# ROIs to one image
def roi_outputs(
    masks: List[torch.Tensor],
    rois: List[torch.Tensor],
    classes: List[torch.Tensor],
    output_size: List[int],
    threshold: float = 0.5,
):
    N = len(masks)
    output = torch.zeros((N, 2, output_size[0], output_size[1]), device=masks[0].device)
    for i, (mask, roi, cls) in enumerate(zip(masks, rois, classes)):
        if len(mask) == 0:
            continue
        out = _do_paste_mask(
            mask, roi, output_size[0], output_size[1], threshold=threshold
        )  # N,h,w
        out = torch.cat([torch.zeros_like(out[0:1]), out], dim=0)
        output[i][0] = out.argmax(dim=0)
        # to paste N classes instance to one image:
        out[1:] = out[1:] * cls[:, None, None]
        output[i][1] = out.max(dim=0).values

    return output.long()  # N,2,h,w


class CategoryDetection(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()  # 256-> 128 -> 64
        self.conv = nn.Sequential(
            bn_act_conv(in_channels, 128, 2, 2),
            )
        self.block_cls = bn_act_conv(128,32,3,1,1)
        self.cls_score = nn.Sequential(
            nn.Linear(32 * 7 **2, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.block_cls(x)
        x = x.flatten(start_dim=1)
        return self.cls_score(x)

class FocalLoss(nn.Module):
    def __init__(
        self,
        gama: float = 2,
        weight=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        self.gama = gama
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return focal_loss(output, target, self.gama, self.weight, self.label_smoothing)


def focal_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    gama: float = 2,
    weight=None,
    label_smoothing: float = 0.0,
):
    if output.shape == target.shape:
        ce_loss = binary_cross_entropy_with_logits(output, target, weight=weight, reduction="none")
    else:
        ce_loss = cross_entropy(
            output,
            target=target,
            reduction="none",
            weight=weight,
            label_smoothing=label_smoothing,
        )
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gama * ce_loss
    return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.smooth = 1e-5

    def forward(self, input, target):
        return dice_loss(input, target, self.smooth)


def dice_loss(input, target, smooth=1e-5, reduction="mean"):
    N = target.size(0)

    target = one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2)[
        :, 1:
    ]  # N, C, H, W
    input = softmax(input, dim=1)[:, 1:]
    input_flat = input.view(N, -1)
    target_flat = target.reshape(N, -1)
    intersection = input_flat * target_flat

    dice = (
        2
        * (intersection.sum(1) + smooth)
        / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    )
    if reduction == "mean":
        loss = 1 - dice.mean()
    elif reduction == "sum":
        loss = N - dice.sum()
    return loss

"""
 giou loss.
"""
def giou_loss(pred, target, reduction="mean"):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                    (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                    torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                    torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return (1-gious).mean()