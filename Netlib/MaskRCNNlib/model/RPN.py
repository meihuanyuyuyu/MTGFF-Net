import torch
import torch.nn as nn
from typing import List, OrderedDict
import torch.nn.functional as F
from .MaskRCNN_utils import (
    bn_act_conv,
    apply_box_delta,
    clip_boxes_to_image,
    remove_big_boxes,
    nms,
    focal_loss,
    box_iou,
    anchors_to_max_boxes_delta,
    balanced_pos_neg_sample,
)


class RPN_(nn.Module):
    def __init__(
        self,
        in_c,
        default_anchors=List[torch.Tensor],
        stride=[4, 8],
        rpn_pos_threshold=0.7,
        rpn_fraction_ratio: float = 0.3,
        pre_nms_k: int = 6000,
        post_nms_k: int = 3000,
        threshold: int = 0.2,
        img_size=[256, 256],
    ) -> None:
        super().__init__()
        self.stride = stride
        self.rpn_pos_threshold = rpn_pos_threshold
        self.rpn_fraction_ratio = rpn_fraction_ratio
        self.pre_nms_k = pre_nms_k
        self.nms_threshold = threshold
        self.post_nms_k = post_nms_k
        self.img_size = img_size
        self.default_anchors = default_anchors
        self.k = len(default_anchors[0])
        self.base_conv = bn_act_conv(in_c, 256, 3, 1, 1)

        self.conv_cls = nn.Sequential(
            *[bn_act_conv(256, 256, 3, 1, 1) for _ in range(2)]
        )
        self.conv_reg = nn.Sequential(
            *[bn_act_conv(256, 256, 3, 1, 1) for _ in range(2)]
        )
        self.logist = nn.Conv2d(256, self.k, 1, 1)
        self.reg = nn.Conv2d(256, 4 * self.k, 1, 1)

    @torch.no_grad()
    def output_proposal(self, logist: List[torch.Tensor], reg: List[torch.Tensor]):
        "output a list that contain Instance data structure,which has pred_bbox,scores attributes"
        out = []
        # per img
        for _ in range(len(logist[0])):
            bbox = []
            scores = []
            # per level
            for a, l, r in zip(self.default_anchors, logist, reg): # n,4k,h,w
                a = a.view(-1, 4)
                rpn_logist_per_img: torch.Tensor = l[_].flatten()
                rpn_logist_per_img = torch.sigmoid(rpn_logist_per_img)
                # each feature map predicts reg need to multiply stride
                rpn_reg_per_img = r[_].view(4, -1).T.contiguous()  # n,4k,h,w

                rpn_logist_per_img_bool = rpn_logist_per_img >= 0.5
                if rpn_logist_per_img_bool.sum() == 0:
                    bbox.append(a[rpn_logist_per_img_bool]), scores.append(
                        rpn_logist_per_img[rpn_logist_per_img_bool]
                    )
                    continue

                rpn_logist_per_img = rpn_logist_per_img[rpn_logist_per_img_bool]
                rpn_reg_per_img = rpn_reg_per_img[rpn_logist_per_img_bool]
                pos_anchors = a[rpn_logist_per_img_bool]
                k = min(len(rpn_logist_per_img), self.pre_nms_k)

                indexes = torch.topk(rpn_logist_per_img, k).indices
                rpn_logist_per_img = rpn_logist_per_img[indexes]
                rpn_reg_per_img = rpn_reg_per_img[indexes]
                pos_anchors = pos_anchors[indexes]

                proposals = apply_box_delta(pos_anchors, rpn_reg_per_img)
                proposals = clip_boxes_to_image(proposals, self.img_size)
                indexes = remove_big_boxes(proposals, 120)
                proposals = proposals[indexes]
                rpn_logist_per_img = rpn_logist_per_img[indexes]

                bbox.append(proposals)
                scores.append(rpn_logist_per_img)
            # merge different level proposals and to deal with per img
            bbox = torch.cat(bbox, dim=0)
            scores = torch.cat(scores)
            indexes = nms(bbox, scores, self.nms_threshold)
            # 保留一定数量
            if self.training:
                post_nms_num = min(len(indexes), self.post_nms_k)
                indexes = indexes[:post_nms_num]
                ins = {}
                ins["pred_bbox"] = bbox[indexes]
                ins["scores"] = scores[indexes]
                out.append(ins)
            else:
                post_nms_num = min(len(indexes), self.post_nms_k - 500)
                indexes = indexes[:post_nms_num]
                ins = {}
                ins["pred_bbox"] = bbox[indexes]
                ins["scores"] = scores[indexes]
                out.append(ins)
        return out

    @torch.no_grad()
    def generate_rpn_targets(self, target_boxes):
        # per level and img gt
        # merge diffrent level anchors [k,h,w,4]->-1,4
        anchors = [anchors.reshape(-1, 4) for anchors in self.default_anchors]

        anchors = torch.cat(anchors, dim=0)
        # split different level:
        gt_logist = torch.zeros(
            len(target_boxes), anchors.shape[0], device=anchors.device
        )
        gt_reg = torch.zeros(len(target_boxes), *anchors.shape, device=anchors.device)

        for idx, boxes_per_img in enumerate(target_boxes):
            if len(boxes_per_img) == 0:
                continue
            iou = box_iou(anchors, boxes_per_img)
            max_iou, max_idx = iou.max(dim=-1)
            # pos_anchors,neg_anchors,caculate diffrent level reg
            pos_idx = max_iou >= self.rpn_pos_threshold
            gt_logist[idx][pos_idx] = 1

            neg_idx = max_iou < 0.3
            neg_idx = balanced_pos_neg_sample(pos_idx, neg_idx, self.rpn_fraction_ratio)
            gt_logist[idx][torch.logical_not(torch.logical_or(pos_idx, neg_idx))] = -1
            gt_reg[idx] = anchors_to_max_boxes_delta(anchors, boxes_per_img, max_idx)
        # transpose first batch to first level
        return {"target_logist": gt_logist, "target_reg": gt_reg}

    def compute_rpn_loss(
        self,
        rpn_logist: List[torch.Tensor],
        rpn_reg: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
    ):
        # generate groud truth
        # n,k*h*w*s,4
        training_targets = self.generate_rpn_targets(target_boxes)
        rpn_logist_target, rpn_reg_target = (
            training_targets["target_logist"],
            training_targets["target_reg"],
        )
        # print(target_boxes[0].shape,(rpn_logist_target[0]==1).sum(),(rpn_logist_target[0]==0).sum())

        # per img to cat different level n,k*h*w
        # [(n,4*k,h,w), ...]->[n,k*h*w+k*h2*w2]
        rpn_logist = torch.cat(
            [x.reshape(len(target_boxes), -1) for x in rpn_logist], dim=-1
        )
        rpn_reg = torch.cat(
            [
                x.view(len(target_boxes), 4, -1).permute(0, 2, 1).contiguous()
                for x in rpn_reg
            ],
            dim=1,
        )

        loss_logist = focal_loss(
            rpn_logist[rpn_logist_target >= 0],
            rpn_logist_target[rpn_logist_target >= 0],
        )
        loss_reg = F.smooth_l1_loss(
            rpn_reg[rpn_logist_target == 1], rpn_reg_target[rpn_logist_target == 1]
        )
        # print(loss_logist,loss_reg)
        return {"rpn_loss": 5 * loss_reg + loss_logist}

    def forward(self, x: OrderedDict, target_boxes: List[torch.Tensor] = None):
        logist_out = []
        reg_out = []
        for l in self.stride:
            feature = x["f{}".format(l)]
            feature = self.base_conv(feature)
            cls_tower = self.conv_cls(feature)
            reg_tower = self.conv_reg(feature)
            logist_out.append(self.logist(cls_tower))
            reg_out.append(self.reg(reg_tower))
        if self.training:
            losses = self.compute_rpn_loss(
                logist_out, reg_out, target_boxes=target_boxes
            )
            proposals = self.output_proposal(logist_out, reg_out)
        else:
            losses = None
            proposals = self.output_proposal(logist_out, reg_out)
        return proposals, losses
