from typing import List, OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
from .MaskRCNN_utils import box_convert,apply_box_delta,clip_boxes_to_image,bn_act_conv

class BoxHead(nn.Module):
    def __init__(self, in_channels, out_channels=4, out_size=14):
        super().__init__()
        # self.conv = bn_act_conv(in_channels,64,3,1,1) 256->256 256->64 64->16
        assert out_size % 14 == 0
        self.block_reg = nn.Sequential(
            bn_act_conv(in_channels, 128, 3, 1, 1),
            bn_act_conv(128, 64, 1, 1 ),
            bn_act_conv(64, 16, 1, 1),
        )
        self.fc = nn.Linear(
            14 * 14 * 16,
            out_features=out_channels,
        )

    def forward(self, x):
        reg_x = self.block_reg(x)
        # 256-> 64*7**2
        reg_x = reg_x.flatten(start_dim=1)  # n,1024
        return self.fc(reg_x)


class BoxRefinement(nn.Module):
    def __init__(
        self,
        roi_head,
        box_head=BoxHead,
        use_expand=True,
        expand_ratio=0.2,
    ) -> None:
        super().__init__()
        self.box_head = box_head(256, out_channels=4)
        self.expand = use_expand
        self.roi_head = roi_head
        self.expand_ratio = expand_ratio
        if self.expand:
            self.refine_head = None

    @torch.no_grad()
    def refine_boxes(self, boxes: List[Tensor], reg: Tensor, box_weights=[10.0, 10.0, 5.0, 5.0], img_size=(256, 256)):
        reg = reg.split([box.shape[0] for box in boxes])
        for _ in range(len(boxes)):
            boxes[_] = apply_box_delta(boxes[_], reg[_], box_weight=box_weights)
            boxes[_] = clip_boxes_to_image(boxes[_], img_size)
        return boxes

    @torch.no_grad()
    def expand_boxes(self, boxes: List[Tensor], delta: float = 0.2):
        out = []
        for roi in boxes:
            expand_box = roi.clone()
            expand_box = box_convert(expand_box, "xyxy", "cxcuwh")
            expand_box[:, :2] *= (1 + delta)
            out.append(box_convert(expand_box, "cxcywh", "xyxy"))
        return out

    def forward(self, boxes: List[Tensor], features: OrderedDict):
        boxes_feature = self.roi_head(boxes, features)
        if self.expand:
            expand_boxes = self.expand_boxes(boxes, delta=self.expand_ratio)
            expand_boxes_feature = self.roi_head(expand_boxes, features)
            boxes_feature = self.refine_head(boxes_feature, expand_boxes_feature)
        reg = self.box_head(boxes_feature)
        return reg


