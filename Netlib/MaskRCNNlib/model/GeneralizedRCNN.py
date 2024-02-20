from .MaskRCNN_utils import *
from .ROIHead import AdaptiveFeaturePooling
from .RPN import RPN_
from .Backbone import Resnet50FPN, Resnet101FPN
from .MaskHead import MaskHead
from .BoxHead import BoxHead, BoxRefinement
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import List, OrderedDict
from ..tools.metric import proposal_stage2_metric, jaccard_score, compute_box_pre_rec_f1,compute_AP
from torch.cuda.amp import autocast

class MRCNN(nn.Module):
    def __init__(
        self,
        anchors,
        backbone=Resnet50FPN,
        bottom_up= False,
        proposal_generator=RPN_,
        stride=[4, 8],
        rpn_pos_threshold=0.7,
        rpn_fraction_ratio=0.3,
        pre_nms_k=6000,
        nms_threshold=0.2,
        post_nms_k=3000,
        roi_head=AdaptiveFeaturePooling,
        roi_pos_threshold=[0.5, 0.6, 0.7],
        stage2_max_proposal=[1000,800,600],
        roi_resolution=28,
        img_size=[256, 256],
        stage2_sample_ratio=5,
        box_detection=BoxHead,
        box_weight= [1.0, 1.0, 1.0, 1.0],
        expand=False,
        expand_ratio=0.2,
        use_gt_box=False,
        post_decttion_score_threshold=0.5,
        detection_per_img=1000,
        num_classes=7,
        use_semantic = False,
        seg_stride=[2,4,8],
        fuse_feature=None,
        mode = 3,
    ) -> None:
        super().__init__()
        self.device = anchors[0].device
        self.stride = stride
        self.img_size = img_size
        anchors = generate_mul_anchors(anchors,img_size[0],stride)
        self.output_size = roi_resolution
        self.post_detection_score_thresh = post_decttion_score_threshold
        self.box_weight = torch.tensor(box_weight).to(device=self.device)
        self.proposal_generator = proposal_generator(
            256,
            anchors,
            stride=stride,
            rpn_pos_threshold=rpn_pos_threshold,
            rpn_fraction_ratio=rpn_fraction_ratio,
            pre_nms_k=pre_nms_k,
            post_nms_k=post_nms_k,
            threshold=nms_threshold,
            img_size=img_size,
        )
        self.roi_head = roi_head(feature_map=['f4','f8'],img_size=256, output_size=14,mode='sum')
        self.stage2_sample_ratio = stage2_sample_ratio
        self.stage2_max_proposal = (
            stage2_max_proposal  # max boxes per img to train stage2
        )
        self.nms_threshold = nms_threshold
        self.roi_pos_threshold = roi_pos_threshold
        self.detections_per_img = detection_per_img
        self.num_classes = num_classes
        self.use_gt_box = use_gt_box
        # roi detection
        self.refine_boxes = BoxRefinement(
            self.roi_head,
            box_detection,
            use_expand=expand,
            expand_ratio=expand_ratio,
        )
        self.category_detection = CategoryDetection(
            256, num_classes=num_classes
        )
        self.mask_detection = MaskHead(256)
        self.mode = mode

        # Global Semantic Segmentation
        if use_semantic:
            self.global_semantic = GlobalSemanticSeg(256,3)
            self.seg_stride = seg_stride
            if not fuse_feature:
                self.fuse_feature =  FeatureFusion(output_size=14)
            else:
                self.fuse_feature = fuse_feature
        else:
            self.global_semantic = None

        def weight_init():
            for m in self.modules():
                classname = m.__class__.__name__
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if "norm" in classname.lower():
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                if "linear" in classname.lower():
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        weight_init()
        self.backbone = backbone(bottom_up=bottom_up)

    @torch.no_grad()
    def stage2_proposal_sample(
        self, proposals, target_boxes, post_rpn_thresh: float = 0.7 , max_proposal=500
    ):
        r"将第一阶段输出进行正负例平衡采样,并选取不超过256的数量来作为二阶段网络训练。"
        batched_rois = [x['pred_bbox'] for x in proposals]
        for _ in range(len(proposals)):
            N = len(batched_rois[_])
            if N == 0 or target_boxes[_].shape[0] == 0:
                continue
            rois_per_img = batched_rois[_]
            target_boxes_per_img = target_boxes[_]
            iou = box_iou(rois_per_img, target_boxes_per_img)
            max_iou = torch.max(iou, dim=-1).values
            pos = max_iou >= post_rpn_thresh
            neg = balanced_pos_neg_sample(
                pos, max_iou < 0.3, sample_ratio=self.stage2_sample_ratio
            )
            keep = torch.logical_or(pos, neg)
            batched_rois[_] = rois_per_img[keep]
            if N > max_proposal:
                batched_rois[_] = batched_rois[_][: max_proposal]
                # print(f"clip stage2 proposal from {N} to {self.stage2_max_proposal}")

        return batched_rois

    def postprocess_detection(
        self,
        detection_cls: Tensor,
        detection_masks: Tensor,
        proposal: List[Tensor],
        semantci_mask: Tensor = None,
    ):
        r"return {pred_box:[torch.tensor],scores:[torch.tensor],pred_cls:[],pred_mask:[]}"
        scores, detection_cls = torch.max(F.softmax(detection_cls, dim=1), dim=1)
        detection_masks = torch.argmax(detection_masks, dim=1)
        num_boxes_per_img = [boxes.shape[0] for boxes in proposal]
        detection_masks = detection_masks.split(num_boxes_per_img, 0)
        detection_cls = detection_cls.split(num_boxes_per_img, 0)
        scores = scores.split(num_boxes_per_img, 0)
        proposal_list = []
        score_list = []
        cls_list = []
        masks_list = []

        for boxes, cls, score, masks in zip(
            proposal, detection_cls, scores, detection_masks
        ):
            # remove background proposals
            cls_bool = cls.bool()
            boxes, score, cls, masks = (
                boxes[cls_bool],
                score[cls_bool],
                cls[cls_bool],
                masks[cls_bool],
            )

            # remove low scoring boxes
            score_bool = score >= self.post_detection_score_thresh
            # print(score.min(),score.max(),self.post_detection_score_thresh)
            boxes, score, cls, masks = (
                boxes[score_bool],
                score[score_bool],
                cls[score_bool],
                masks[score_bool],
            )

            boxes = clip_boxes_to_image(boxes, self.img_size)
            keep = remove_big_boxes(boxes, 90)
            boxes, score, cls, masks = boxes[keep], score[keep], cls[keep], masks[keep]
            keep = remove_small_boxes(boxes, 1.5)
            boxes, score, cls, masks = boxes[keep], score[keep], cls[keep], masks[keep]

            keep = nms(boxes, score, self.nms_threshold)
            keep = keep[: self.detections_per_img]
            boxes, score, cls, masks = boxes[keep], score[keep], cls[keep], masks[keep]

            proposal_list.append(boxes)
            score_list.append(score)
            cls_list.append(cls)
            masks_list.append(masks)
        return {
            "pred_boxes": proposal_list,
            "scores": score_list,
            "pred_classes": cls_list,
            "pred_masks": masks_list,
            "pred_semantic_masks": torch.argmax(semantci_mask, dim=1) if semantci_mask is not None else None,
        }

    def compute_gt_detection_loss(
        self, detection_cls, detection_masks, target_cls, target_masks
    ):
        r"计算gt的二阶段loss,返回gt损失字典"
        if len(target_cls) == 0:
            return {
                "gt_cls_loss": torch.tensor(0.0, device=detection_cls.device),
                "gt_mask_loss": torch.tensor(0.0, device=detection_cls.device),
            }
        else:
            target_cls = torch.cat(target_cls, dim=0).long()
            target_masks = torch.cat(target_masks, dim=0).long()
            gt_cls_loss = focal_loss(detection_cls, target_cls)
            mask_loss = F.cross_entropy(detection_masks, target_masks)
            return {"gt_cls_loss": gt_cls_loss, "gt_mask_loss": mask_loss}

    def compute_detection_loss(
        self,
        batched_rois,
        detection_box_cls,
        detection_box_reg,
        detection_masks,
        target_boxes,
        target_cls,
        target_masks,
        out_size=14,
    ):
        cls, reg, masks = generate_detection_targets(
            batched_rois,
            target_boxes,
            target_cls,
            target_masks,
            out_size=out_size,
            iou_thresh=self.roi_pos_threshold,
            box_weight = self.box_weight
        )
        if len(cls) != 0:
            cls = torch.cat(cls, dim=0)
            # print('二阶段标签:',torch.bincount(cls))
            # print('输出cls：',torch.bincount(detection_box_cls.argmax(dim=-1)))
            # gt_cls = torch.cat(target_cls,dim=0).long()
            masks = torch.cat(masks, dim=0)
            reg = torch.cat(reg, dim=0)
            cls_loss = focal_loss(detection_box_cls, cls, gama=2) # n,7
            mask_loss = F.cross_entropy(detection_masks[cls != 0], masks[cls != 0])
            reg_loss = F.smooth_l1_loss(detection_box_reg[cls != 0], reg[cls != 0])
            # print(f'cls_loss:{cls_loss},mask_loss:{mask_loss},reg_loss:{reg_loss}')
            loss = 3 * cls_loss + mask_loss + 5 * reg_loss
            # print('detection loss:',loss)
        else:
            loss = None
        return {"detection_loss": loss}

    @torch.no_grad()
    def log_metric(
        self, proposals, preds, gt_boxes, gt_masks, gt_cls, gt_semantic
    ) -> OrderedDict:
        s2_proposals = preds["pred_boxes"]
        scores = preds['scores']
        pred_masks = preds["pred_masks"]
        pred_cls = preds["pred_classes"]
        pred_semantic_masks = (
            preds["pred_semantic_masks"] if "pred_semantic_masks" in preds else None
        )

        stage1_box_F1 = []
        stage2_box_F1 = []
        stage2_box_AP = []
        # stage1 box iou 查看一阶段检测框预测质量,计算f1,pre,rec以及iou
        for pred_boxes,s2_boxes,score, gt in zip(proposals, s2_proposals, scores, gt_boxes):
            (p,r,f1)= compute_box_pre_rec_f1(pred_boxes, gt, [0.5])
            (p,r,s2_f1) = compute_box_pre_rec_f1(s2_boxes, gt, [0.5,0.6,0.7])
            aps = []
            for t in [0.5,0.6,0.7]:
                ap =compute_AP(pred_box=s2_boxes,boxes_scores=score,gt=gt,threshold=t)
                aps.append(ap)
            stage2_box_AP.append(aps)
            stage1_box_F1.append(f1) # (n,2)
            stage2_box_F1.append(s2_f1) # (n,3)
        stage1_box_F1 = torch.tensor(stage1_box_F1).mean().item()
        stage2_box_F1 = torch.tensor(stage2_box_F1).mean(dim=0) # (3,)
        stage2_box_AP = torch.tensor(stage2_box_AP).mean(dim=0)

        box_ious, mask_iou, categ_acc = proposal_stage2_metric(
            s2_proposals, pred_masks, pred_cls, gt_boxes, gt_masks, gt_cls
        )

        if pred_semantic_masks is not None:
            semantic_iou = jaccard_score(
                gt_semantic.flatten().cpu().numpy(),
                pred_semantic_masks.flatten().cpu().numpy(),
                average='micro',
            )
        else:
            semantic_iou = None
        return stage1_box_F1, stage2_box_F1,stage2_box_AP, mask_iou, categ_acc, semantic_iou

    @torch.no_grad()
    def inference(self, pred_dict: OrderedDict[str, Tensor], img_size: List[int]=[256,256]):
        pred_boxes = pred_dict["pred_boxes"]
        pred_masks = pred_dict["pred_masks"]
        pred_cls = pred_dict["pred_classes"]
        return (
            roi_outputs(
                masks=pred_masks, rois=pred_boxes, classes=pred_cls, output_size=img_size
            )
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

    @autocast()
    def forward(
        self,
        img: Tensor,
        target_boxes: List[Tensor] = None,
        masks: List[Tensor] = None,
        cls: List[Tensor] = None,
        semantic: Tensor = None,
    ):
        features: OrderedDict = self.backbone(img)
        if self.global_semantic is not None:
            semantic_features = {f"f{k}": features.get(f"f{k}") for k in self.seg_stride}
        
        if self.training:
            losses = {}
            proposals, loss = self.proposal_generator(features, target_boxes)
            losses.update(loss)
            if self.mode ==1:
                return losses
            batched_rois = self.stage2_proposal_sample(
                proposals, target_boxes, self.roi_pos_threshold[-1], max_proposal=self.stage2_max_proposal[-1]
            )
            batched_rois = [torch.detach(rois) for rois in batched_rois]
            detection_reg = self.refine_boxes(batched_rois, features)
            
            # ROIAlign:
            roi_feature = self.roi_head(batched_rois, features)
            detection_cls = self.category_detection(roi_feature)
            if self.global_semantic is not None:
                semantic_features, pred_semantic_mask, loss = self.global_semantic(semantic_features, semantic)
                losses.update(loss)
                fused_roi_feature = self.fuse_feature(roi_feature, semantic_features, batched_rois)
                detection_masks = self.mask_detection(fused_roi_feature)
            else:
                detection_masks = self.mask_detection(roi_feature)

            # training with gt:
            if self.use_gt_box:
                gt_roi_feature = self.roi_head(target_boxes, features)
                gt_detection_cls = self.category_detection(gt_roi_feature)
                gt_detection_masks = self.mask_detection(gt_roi_feature)
                losses.update(
                    self.compute_gt_detection_loss(
                        gt_detection_cls, gt_detection_masks, cls, masks
                    )
                )

            detection_loss = self.compute_detection_loss(
                batched_rois,
                detection_cls,
                detection_reg,
                detection_masks,
                target_boxes,
                cls,
                masks,
                out_size=self.output_size,
            )
            losses.update(detection_loss)
            return losses
        # inference:
        else:
            proposals, _ = self.proposal_generator(features)
            s1_batched_rois = [p['pred_bbox'] for p in proposals]

            detection_reg = self.refine_boxes(s1_batched_rois, features)
            batched_rois = self.refine_boxes.refine_boxes(
                s1_batched_rois, detection_reg, self.box_weight
            )
            roi_feature = self.roi_head(batched_rois, features)
            detection_cls = self.category_detection(roi_feature)
            if self.global_semantic is not None:
                semantic_feature, pred_semantic_mask, _ = self.global_semantic(semantic_features)
                fused_roi_feature = self.fuse_feature(roi_feature, semantic_feature, batched_rois)
                detection_masks = self.mask_detection(fused_roi_feature)
            else:
                detection_masks = self.mask_detection(roi_feature)
                pred_semantic_mask = None

            pred = self.postprocess_detection(
                detection_cls, detection_masks, batched_rois, pred_semantic_mask
            )
            metric = (
                self.log_metric(
                    s1_batched_rois, pred, target_boxes, masks, cls, semantic
                )
                if target_boxes is not None
                else None
            )
            return pred, metric


