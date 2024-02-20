import torch
import torch.nn as nn
from typing import List, OrderedDict
from torchvision.ops import roi_align

class AdaptiveFeaturePooling(nn.Module):
    def __init__(
        self, feature_map= ['f4','f8'], output_size: int = 14, mode: str = "sum", img_size: int = 256
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.mode = mode
        self.img_size = img_size
        self.feature_map = feature_map

    def forward(self, boxes: List[torch.tensor], feature: OrderedDict):
        assert self.mode == "max" or self.mode == "sum"
        aggregated_feature = []
        for key in self.feature_map:
            out = roi_align(
                feature[key], 
                boxes,
                self.output_size,
                spatial_scale=feature[key].shape[-1] / self.img_size,
                aligned=False,
            )
            aggregated_feature.append(out)
        if self.mode == "max":
            return torch.max(torch.stack(aggregated_feature, dim=1), dim=1).values
        if self.mode == "sum":
            return torch.sum(torch.stack(aggregated_feature, dim=1), dim=1) # [N,C,H,W]