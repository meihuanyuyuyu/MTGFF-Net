import torch
import torch.nn as nn
from typing import List, OrderedDict
from .MaskRCNN_utils import bn_act_conv,dice_loss,focal_loss


class MSCAN(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()
        self.depthWiseConv1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, 64, 1, 1, 0, groups=in_c),
        )
        self.depthWiseConv2 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, 64, 3, 1, 1, groups=in_c),
        )
        self.depthWiseConv3 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, 64, 5, 1, 2, groups=in_c),
        )
        self.conv = bn_act_conv(64, in_c, 1, 1, 0)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
    
    def forward(self, x:torch.Tensor):
        att = self.depthWiseConv1(x) + self.depthWiseConv2(x) + self.depthWiseConv3(x) +x 
        return self.bn(x * self.conv(att))


class GlobalSemanticSeg(nn.Module):
    def __init__(self, in_channels, num_classes, weight=torch.tensor([1,1.5,1]) ) -> None:
        '2.interior 1.boundary 0.background'
        super().__init__()  # 256-> 128 -> 64
        self.weight = weight.to('cuda').float()
        self.up_2 = nn.Sequential(
            bn_act_conv(64, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.up_4 = nn.Sequential(
            bn_act_conv(in_channels, 64, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        )
        self.up_8 = nn.Sequential(
            bn_act_conv(in_channels, 64, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        )

        self.s1 = MSCAN(64)
        self.s2 = MSCAN(64)
        self.semantic_out = bn_act_conv(64, num_classes,1,1,0)
        self.feature_out = bn_act_conv(64, in_channels, 1, 1, 0)
    
    def compute_loss(self, semantic_pred:torch.Tensor, semantic_target:torch.Tensor):
        #print(semantic_target.shape, semantic_target.unique())
        loss_d = dice_loss(semantic_pred, semantic_target)
        loss_f = focal_loss(semantic_pred, semantic_target, weight=self.weight)
        #print("dice:", loss_d)
        semantic_loss = 0.* loss_d +0.25* loss_f
        #print("focal:", loss_f)
        return {"semantic_loss": semantic_loss}

    def forward(self, feature:OrderedDict, seg_stride= [2,4], target=None):
        'target: (N,h,w) 0,1,2'
        for idx,stride in enumerate(seg_stride):
            if idx==0:
                x = getattr(self,f"up_{stride}")(feature[f"f{stride}"])
            else:
                x += getattr(self,f"up_{stride}")(feature[f"f{stride}"])
        x = self.s1(x)
        x = self.s2(x)
        if self.training:
            semantic = self.semantic_out(x)
            feature = self.feature_out(x)
            loss = self.compute_loss(semantic, target)
            return  feature, semantic, loss
        else:
            feature = self.feature_out(x)
            semantic = self.semantic_out(x)
            return feature, semantic, None