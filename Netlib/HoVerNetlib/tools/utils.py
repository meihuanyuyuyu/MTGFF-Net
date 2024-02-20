import torch.nn as nn
from torch.nn.functional import (
    interpolate,
    cross_entropy,
    binary_cross_entropy_with_logits,
    one_hot,
    softmax,
    grid_sample,
)
import torch
from typing import List
from torch import Tensor
from torchvision.ops import (
    box_convert,
    box_iou,
    clip_boxes_to_image,
    remove_small_boxes,
    nms,
)
from torchvision.utils import draw_segmentation_masks, save_image
import numpy as np


def remove_big_boxes(boxes: Tensor, size: float) -> Tensor:
    # 去除大框
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    keep = (h < size) & (w < size)
    return keep


@torch.no_grad()
def apply_box_delta(anchors: Tensor, reg: Tensor, box_weight = [1.0,1.0,1.0,1.0]):
    # 将锚框根据预测变换为检测框形式
    if not isinstance(box_weight, torch.Tensor):
        box_weight = torch.tensor(box_weight, device=anchors.device)
    anchors = box_convert(anchors, "xyxy", "cxcywh")
    anchors[:, :2] = anchors[:, :2] + anchors[:, 2:] * (reg[:, :2]/box_weight[:2])
    anchors[:, 2:] = anchors[:, 2:] * torch.exp(reg[:, 2:]/box_weight[2:])
    return box_convert(anchors, "cxcywh", "xyxy")


@torch.no_grad()
def generate_anchors(anchors_per_bin: Tensor, img_size, scale):
    r"中心对齐：(x+0.5)*scale-0.5, 生成锚框坐标(k,h,w,4)"
    anchors_per_bin = anchors_per_bin.unsqueeze(1).unsqueeze(1)
    anchors = torch.zeros(len(anchors_per_bin), img_size // scale, img_size // scale, 4)
    x_coordinate, y_coordinate = torch.meshgrid(
        [torch.arange(0.5 * scale - 0.5, img_size, step=scale) for _ in range(2)],
        indexing="xy",
    )
    xy = torch.stack([x_coordinate, y_coordinate], dim=-1).unsqueeze(0)
    anchors[..., :2] = xy - anchors_per_bin / 2
    anchors[..., 2:] = xy + anchors_per_bin / 2
    return anchors  # k,h,w,4


@torch.no_grad()
def generate_mul_anchors(
    anchors_per_bin: Tensor,
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


@torch.no_grad()
def anchors_to_max_boxes_delta(anchors: Tensor, boxes: Tensor, indexes: Tensor, weights: List[float] = [1.0,1.0,1.0,1.0]):
    if not isinstance(weights, Tensor):
        weights = torch.tensor(weights, device=anchors.device)
    target_boxes = boxes[indexes]
    target_boxes = box_convert(target_boxes, "xyxy", "cxcywh")
    anchors = box_convert(anchors, "xyxy", "cxcywh")
    reg_xy_delta = (target_boxes[:, :2] - anchors[:, :2]) / target_boxes[:, 2:] 
    reg_wh_delta = torch.log(target_boxes[:, 2:] / anchors[:, 2:])
    return torch.cat([reg_xy_delta*weights[0:2] , reg_wh_delta*weights[2:] ], dim=-1)


############################# Model utils ################################


@torch.no_grad()
def proposal_layer(
    anchors: Tensor,
    rpn_logists: Tensor,
    rpn_reges: Tensor,
    pre_nms_k: int = 6000,
    threshold: int = 0.2,
    img_size: int = 256,
    is_train: bool = False,
):
    r"根据模型输出的分数和回归,在非极大值抑制前选取置信度最高的正例框(eg. 2000个),然后对这些rois进行去除超出图像大小的框,还有过小的框，最后经过nms输出rois并且修正偏移量"
    anchors = anchors.view(-1, 4)
    boxes = []
    scores = []
    for _ in range(len(rpn_logists)):
        rpn_logist_per_img = rpn_logists[_].flatten()
        rpn_reg_per_img = rpn_reges[_].view(4, -1).T.contiguous()  # 4*15,64,64
        # 筛选正例阈值
        rpn_logist_per_img_bool = rpn_logist_per_img >= 0.5
        # 没有目标时返回空张量
        if rpn_logist_per_img_bool.sum() == 0:
            boxes.append(anchors[rpn_logist_per_img_bool]), scores.append(
                anchors[rpn_logist_per_img_bool]
            )
            continue
        rpn_logist_per_img = rpn_logist_per_img[rpn_logist_per_img_bool]
        rpn_reg_per_img = rpn_reg_per_img[rpn_logist_per_img_bool]
        temp_anchors = anchors[rpn_logist_per_img_bool]
        # 筛选正例前k个和正例预测
        k = min(len(rpn_logist_per_img), pre_nms_k)
        indexes = torch.topk(rpn_logist_per_img, k).indices
        rpn_logist_per_img = rpn_logist_per_img[indexes]
        rpn_reg_per_img = rpn_reg_per_img[indexes]
        temp_anchors = temp_anchors[indexes]
        # 筛选过大过小框
        boxes_per_img = apply_box_delta(temp_anchors, rpn_reg_per_img)
        boxes_per_img = clip_boxes_to_image(boxes_per_img, [img_size, img_size])
        indexes = remove_small_boxes(boxes_per_img, 1)
        boxes_per_img = boxes_per_img[indexes]
        rpn_logist_per_img = rpn_logist_per_img[indexes]
        indexes = remove_big_boxes(boxes_per_img, 60)
        boxes_per_img = boxes_per_img[indexes]
        rpn_logist_per_img = rpn_logist_per_img[indexes]
        # 非极大值抑制
        indexes = nms(boxes_per_img, rpn_logist_per_img, threshold)
        # 保留一定数量
        if is_train:
            post_nms_num = min(len(indexes), 2000)
            indexes = indexes[:post_nms_num]
            boxes.append(boxes_per_img[indexes])
            scores.append(rpn_logist_per_img[indexes])
        else:
            post_nms_num = min(len(indexes), 1500)
            indexes = indexes[:post_nms_num]
            boxes.append(boxes_per_img[indexes])
            scores.append(rpn_logist_per_img[indexes])
    return boxes, scores


@torch.no_grad()
def rois2img(proposals, out_cls, out_masks):
    r"return:(n,256,256)"
    result = torch.zeros(
        len(proposals), 256, 256, dtype=torch.float, device=proposals[0].device
    )
    for _, (proposals, out_cls, out_masks) in enumerate(
        zip(proposals, out_cls, out_masks)
    ):
        if len(proposals) == 0:
            continue
        proposals = proposals.long()
        # proposals[:,2:] += 1 #??
        proposals = clip_boxes_to_image(proposals, [256, 256])
        for num_box in range(len(proposals)):
            w = (proposals[num_box, 2] - proposals[num_box, 0]).item()
            h = (proposals[num_box, 3] - proposals[num_box, 1]).item()
            mask = (
                interpolate(
                    out_masks[num_box : num_box + 1, None].float(),
                    size=[h, w],
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            result[
                _,
                proposals[num_box, 1] : proposals[num_box, 1] + h,
                proposals[num_box, 0] : proposals[num_box, 0] + w,
            ] += (
                mask.round().long()
                & (
                    result[
                        _,
                        proposals[num_box, 1] : proposals[num_box, 1] + h,
                        proposals[num_box, 0] : proposals[num_box, 0] + w,
                    ]
                    == False
                )
            ) * out_cls[
                num_box
            ].item()
    return result.long()


@torch.no_grad()
def convert_prediction_to_numpy(proposals: Tensor, out_cls: Tensor, out_masks: Tensor):
    label = torch.zeros(2, 256, 256, dtype=torch.float, device=proposals.device)
    if 0 in proposals.shape:
        return label.permute(1, 2, 0).unsqueeze(0).cpu().numpy()
    proposals = proposals.long()
    # proposals[:,2:] +=1
    proposals = clip_boxes_to_image(proposals, [256, 256])
    for _, box in enumerate(proposals):
        x = box[0].item()
        y = box[1].item()
        w = (box[2] - box[0]).item()
        h = (box[3] - box[1]).item()
        mask = (
            interpolate(
                out_masks[_ : _ + 1, None].float(),
                size=[h, w],
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask_pos = mask.round().long() & (label[1, y : y + h, x : x + w] == False)
        label[1, y : y + h, x : x + w] = (
            mask_pos * out_cls[_] + label[1, y : y + h, x : x + w]
        )
        label[0, y : y + h, x : x + w] = (
            mask_pos * (_ + 1) + label[0, y : y + h, x : x + w]
        )
    return label.permute(1, 2, 0).unsqueeze(0).cpu().numpy()


######################### Balanced_pos_neg_sample ###########################
def balanced_pos_neg_sample(pos: Tensor, neg: Tensor, sample_ratio=0.3):
    prob = pos.sum() / (neg.sum() * sample_ratio)
    neg = neg * (torch.rand(*neg.shape, device=neg.device) <= prob)
    return neg


########################### Label Generator ###############################
@torch.no_grad()
def generate_rpn_targets(
    default_anchors: Tensor, target_boxes: List[Tensor], pos_threshold=0.6
):
    r"target_boxes:[(n,4),(n,4)],return:(batch,15,64,64),(batch,15,64,64,4)"
    # n,k,h,w
    rpn_logist_label = torch.zeros(
        len(target_boxes),
        default_anchors.shape[0],
        default_anchors.shape[1],
        default_anchors.shape[2],
        device=default_anchors.device,
    )
    rpn_reg_label = torch.zeros(
        len(target_boxes), *default_anchors.shape, device=default_anchors.device
    )
    default_anchors = default_anchors.flatten(end_dim=-2)
    for _, boxes_per_img in enumerate(target_boxes):
        if len(boxes_per_img) == 0:
            continue
        iou = box_iou(default_anchors, boxes_per_img)
        max_iou, indexes = torch.max(iou, dim=-1)
        # pos
        pos_indexes = max_iou > pos_threshold
        pos_indexes = pos_indexes.view(*rpn_logist_label[_].shape)
        rpn_logist_label[_][pos_indexes] = 1
        # neg
        neg_indexes = max_iou < 0.3
        neg_indexes = balanced_pos_neg_sample(pos_indexes, neg_indexes, 0.2).view(
            *rpn_logist_label[_].shape
        )
        rpn_logist_label[_][
            torch.logical_not(torch.logical_or(neg_indexes, pos_indexes))
        ] = -1
        rpn_reg_label[_] = anchors_to_max_boxes_delta(
            default_anchors, boxes_per_img, indexes
        ).view(*rpn_reg_label[0].shape)

    return rpn_logist_label, rpn_reg_label


@torch.no_grad()
def generate_detection_targets(
    batched_rois: List[Tensor],
    target_boxes: List[Tensor],
    target_cls: List[Tensor],
    target_masks: List[Tensor],
    out_size: int,
    iou_thresh: float,
    box_weight = [1.0,1.0,1.0,1.0]
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
def box2grid(rois_boxes: Tensor, output_size=[28, 28]):
    r"rois_boxes:(rois,4)xyxy, output_size:[h,w],return:rois,h,w,2"
    grids = torch.zeros(0, output_size[0], output_size[1], 2, device=rois_boxes.device)
    rois_boxes = rois_boxes / 127.5 - 1
    x1 = rois_boxes[:, 0]
    x2 = rois_boxes[:, 2]
    y1 = rois_boxes[:, 1]
    y2 = rois_boxes[:, 3]
    for _ in range(len(rois_boxes)):
        x = torch.linspace(x1[_].item(), x2[_].item(), output_size[1])
        y = torch.linspace(y1[_].item(), y2[_].item(), output_size[0])
        grid_x, grid_y = torch.meshgrid([x, y], indexing="xy")
        grid = torch.stack([grid_x, grid_y], dim=-1).to(device=rois_boxes.device)
        grids = torch.cat([grids, grid.unsqueeze(0)], dim=0)
    return grids


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


# ******************************* draw utils ***************************


def draw_instance_map(imgs, preds, fp):
    r"give predicted numpy results (n,h,w,2) and return saved imgs"
    preds = (
        torch.from_numpy(preds.astype(np.float64))
        .long()
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    color = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        dtype=torch.float32,
    )
    for _, (img, pred) in enumerate(zip(imgs, preds)):
        img = (img * 255).to(dtype=torch.uint8)
        mask_map = pred[1]
        ins_map = pred[0]
        mask = color[mask_map].permute(2, 0, 1)
        ins_map = (
            one_hot(ins_map, num_classes=ins_map.max() + 1).permute(2, 0, 1)[1:].bool()
        )
        if not len(ins_map) == 0:
            ins = draw_segmentation_masks(img, ins_map, alpha=0.7) / 255
        else:
            ins = img / 255
        save_image(ins, fp + f"/{_}ins.png")
        save_image(mask, fp + f"/{_}cls.png")


# ******************************* loss function ***************************


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

    def forward(self, output: Tensor, target: Tensor):
        return focal_loss(output, target, self.gama, self.weight, self.label_smoothing)


def focal_loss(
    output: Tensor,
    target: Tensor,
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
