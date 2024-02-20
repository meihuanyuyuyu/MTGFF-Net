from typing import List
from .utils import rois2img
from sklearn.metrics import f1_score,jaccard_score
from torchvision.utils import draw_bounding_boxes, make_grid, save_image
from torchvision.ops import box_iou
import torch
#from detectron2.structures.masks import polygons_to_bitmask


@torch.no_grad()
def proposal_targetboxes_miou(
    proposal: List[torch.Tensor], target_boxes: List[torch.Tensor]
):
    max_ious = torch.zeros(0, device=proposal[0].device)
    all_acc = torch.zeros(0, device=proposal[0].device)
    all_recall = torch.zeros(0, device=proposal[0].device)

    for proposal, target_boxes in zip(proposal, target_boxes):
        iou = box_iou(proposal, target_boxes)
        if len(target_boxes) == 0:
            continue
        max_iou, index = torch.max(iou, dim=-1)
        max_ious = torch.cat([max_ious, max_iou], dim=0)
        # 大于0.7阈值算正确的预测框。
        pos_index = max_iou > 0.5
        acc = max_iou[pos_index].sum() / len(max_iou)
        all_acc = torch.cat([all_acc, acc[None]])
        # 召回率：
        recall = max_iou[pos_index].sum() / len(target_boxes)
        all_recall = torch.cat([all_recall, recall[None]])
    return (
        max_ious.mean().nan_to_num(0).item(),
        all_acc.mean().nan_to_num(0).item(),
        all_recall.mean().nan_to_num(0).item(),
    )


@torch.no_grad()
def stage1_val(
    boxes: List[torch.Tensor], preds: List[torch.Tensor], imgs: torch.Tensor, fp
):
    r"根据一阶段roi和boxes,统计验证集上miou并可视化保存效果图片"
    imgs = (imgs * 255).to(torch.uint8)
    pic = torch.zeros(0, 3, 256, 256, device=imgs.device)
    for img, box, pred in zip(imgs, boxes, preds):
        true = (draw_bounding_boxes(img, box) / 255).to(device=imgs.device).unsqueeze(0)
        pred_boxes = (
            (draw_bounding_boxes(img, pred) / 255).to(device=imgs.device).unsqueeze(0)
        )
        pic = torch.cat([pic, true, pred_boxes], dim=0)
    pic = make_grid(pic, 2)
    save_image(pic, fp=fp)
    return proposal_targetboxes_miou(preds, boxes)


@torch.no_grad()
# 可视化一阶段二阶段检测框以及全局掩码，返回评价指标
def stage2_val(
    stage1_rois,
    proposal,
    pred_masks,
    pred_clses,
    target_boxes,
    target_masks,
    target_cls,
    label,
    fp,
    color,
):
    pic = torch.zeros(0, 3, 256, 256, device="cuda")
    pred_mask = rois2img(proposal, pred_clses, pred_masks)
    for _, (label, stage1_rois, rois, boxes, pred_mask) in enumerate(
        zip(label, stage1_rois, proposal, target_boxes, pred_mask)
    ):
        masks_label = ((color[label[1]].permute(2, 0, 1)) * 255).to(torch.uint8)
        pred_mask = ((color[pred_mask].permute(2, 0, 1)) * 255).to(torch.uint8)
        s0 = draw_bounding_boxes(masks_label, boxes).cuda()
        s1 = draw_bounding_boxes(masks_label, stage1_rois).cuda()
        s2 = draw_bounding_boxes(pred_mask, rois).cuda()
        pic = torch.cat([pic, s0[None], s1[None], s2[None]], dim=0)
    pic = make_grid((pic / 255).float(), _ + 1)
    save_image(pic, fp=fp)
    return proposal_stage2_metric(
        proposal, pred_masks, pred_clses, target_boxes, target_masks, target_cls, 0.5
    )


@torch.no_grad()
def proposal_stage2_metric(
    proposal: List[torch.Tensor],
    pred_masks,
    pred_clses,
    target_boxes: List[torch.Tensor],
    target_masks: List[torch.Tensor],
    target_cls: List[torch.Tensor],
    iou_thresh: float = 0.5,
):
    "每张图片boxes平均iou,正例类别标签中正确的占比,mask 平均iou"
    max_boxes_ious = torch.zeros(0, device="cuda")
    max_masks_ious = torch.zeros(0, device="cuda")
    all_cls = []
    for roi, pred_mask, pred_cls, boxes, masks, cls in zip(
        proposal, pred_masks, pred_clses, target_boxes, target_masks, target_cls
    ):
        if len(boxes) == 0:
            continue
        iou = box_iou(roi, boxes)
        max_iou, index = torch.max(iou, dim=-1)

        masks = masks[index]
        cls = cls[index]

        cls[max_iou < iou_thresh] = 0
        max_masks_iou = jaccard(pred_mask, masks)
        cls = f1_score(cls.cpu().numpy(), pred_cls.cpu().numpy(), average="micro")

        max_masks_ious = torch.cat(
            [
                max_masks_ious,
                torch.tensor([max_masks_iou], device=max_boxes_ious.device),
            ],
            dim=0,
        )
        max_boxes_ious = torch.cat([max_boxes_ious, max_iou], dim=0)
        all_cls.append(cls)
    return (
        max_boxes_ious.mean().nan_to_num(0).item(),
        max_masks_ious.mean().nan_to_num(0).item(),
        torch.tensor(all_cls).mean().nan_to_num(0).item(),
    )


@torch.no_grad()
def mask_category_acc(out: torch.Tensor, target_masks: torch.Tensor):
    return (
        (out[target_masks != 0] == target_masks[target_masks != 0])
        .float()
        .mean()
        .nan_to_num(0)
        .item()
    )


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    r"pred:(n,256,256),target:(n,256,256) 2xy\(x+y)"
    scores = torch.zeros((len(target))).to(device=pred.device)
    pred = pred.bool().long()
    target = target.bool().long()
    for idx in range(len(pred)):
        inter = ((pred[idx] + target[idx]) == 2).sum()
        union = pred[idx].sum() + target[idx].sum()
        scores[idx] = 2 * inter / (union+1e-7)
    return scores.mean().item()


@torch.no_grad()
def jaccard(masks: torch.Tensor, target_masks: torch.Tensor):
    TP = (masks * (masks != 0) == target_masks * (target_masks != 0)).sum()
    FP_FN = torch.ne(masks, target_masks).sum()
    return (TP / (TP + FP_FN)).nan_to_num(0).item()


def miou(pred: torch.Tensor, target: torch.Tensor) -> float:
    scores = []
    for idx in range(len(pred)):
        scores.append(jaccard(pred[idx], target[idx]))
    return torch.tensor(scores).mean().nan_to_num(0).item()


'''@torch.no_grad()
def instance2results(preds: List[dict], target: List[dict]):
    ious = []
    cls_acc = []
    mask_iou = []
    for pred_per_img, gt_per_img in zip(preds, target):
        pred_boxes = pred_per_img["instances"]._fields["pred_boxes"].tensor.cpu()
        if len(pred_boxes) == 0:
            continue
        if not gt_per_img["instances"]:
            continue
        gt_boxes = gt_per_img["instances"]._fields["gt_boxes"].tensor

        pred_masks = pred_per_img["instances"]._fields["pred_masks"].cpu()
        gt_mask = gt_per_img["instances"]._fields["gt_masks"].polygons
        gts = torch.zeros(len(gt_mask), 256, 256)
        for _, p in enumerate(gt_mask):
            p = torch.from_numpy(polygons_to_bitmask(p, 256, 256))
            gts[_] = p
        gt_cls = gt_per_img["instances"]._fields["gt_classes"]
        pred_cls = pred_per_img["instances"]._fields["pred_classes"].cpu() + 1

        iou = box_iou(pred_boxes, gt_boxes)
        max_iou, index = torch.max(iou, dim=-1)
        ious.append(max_iou.mean().nan_to_num(0).item())
        cls_acc.append(
            ((pred_cls == gt_cls[index]).sum() / len(pred_cls)).nan_to_num(0).item()
        )
        mask_iou.append(jaccard(pred_masks, gts[index]))
    return (
        torch.tensor(ious).mean().nan_to_num(0).item(),
        torch.tensor(cls_acc).mean().nan_to_num(0).item(),
        torch.tensor(mask_iou).mean().nan_to_num(0).item(),
    )'''

def compute_box_pre_rec_f1(pred_boxes,target_boxes,threshold=[0.5,0.6,0.7]):
    # 一张图片内多个IOU阈值的F1
    iou = box_iou(pred_boxes, target_boxes)
    max_iou, index = torch.max(iou, dim=1)
    precision = []
    recall = []
    f1s = []
    for  t in threshold:
        tp = torch.unique(index[max_iou > t],sorted=True).size(0)
        fp = max_iou.size(0) - tp
        fn = target_boxes.size(0) - tp
        p = tp / (tp + fp) if tp + fp != 0 else 1.0
        r = tp / (tp + fn) if tp + fn != 0 else 1.0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1s.append(f1)
    return precision,recall,f1s

@torch.no_grad()
def compute_AP(pred_box:torch.Tensor,boxes_scores:torch.Tensor,gt,threshold=0.5):
    if len(gt) ==0 and len(pred_box) == 0:
        return 1.0
    if len(gt) ==0 and len(pred_box) != 0:
        return 0.0
    if len(gt) !=0 and len(pred_box) == 0:
        return 0.0
    index = boxes_scores.sort(descending=True).indices
    tp = torch.zeros_like(index)
    fp = torch.zeros_like(index)
    pred_box = pred_box[index]
    # pred_boxes与gt不重复的精准率与召回率
    iou = box_iou(pred_box, gt) # n,m
    max_iou, index = torch.max(iou, dim=1) # n
    overlay = {}
    for num,idx in enumerate(index):
        if max_iou[num]>=threshold:
            if idx not in overlay:
                tp[num] = 1
                overlay[idx] = 1
            else:
                fp[num] = 1 
        else:
            fp[num] = 1
    tp = torch.cumsum(tp,dim=0) if sum(tp) != 0 else 0.0
    fp = torch.cumsum(fp,dim=0) if sum(fp) !=0 else 0.0
    recall = tp/ len(gt) 
    precision = tp/ (tp+fp+1e-10)
    if isinstance(recall,torch.Tensor)  and isinstance(precision,torch.Tensor):
        return voc_ap(recall,precision).nan_to_num(0.0).item()
    else:
        return 0.0



def voc_ap(rec:torch.Tensor, prec:torch.Tensor, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in torch.arange(0., 1.1, 0.1):
            if torch.sum(rec >= t) == 0:
                p = 0
            else:
                p = torch.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = torch.cat((torch.tensor([0.],device=rec.device), rec, torch.tensor([1.],device=rec.device)),dim=0)
        mpre = torch.cat((torch.tensor([0.],device=rec.device), prec, torch.tensor([0.],device=rec.device)),dim=0)

        # compute the precision envelope
        for i in range(mpre.size(0) - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = torch.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    






    


