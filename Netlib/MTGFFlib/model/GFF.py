from typing import OrderedDict
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from .MaskRCNN_utils import roi_align,bn_act_conv

class FeatureFusion(nn.Module):
    def __init__(self,output_size=14, use_gff= False, in_c = 256, out_c = 256) -> None:
        super().__init__()
        self.output_size = output_size
        self.use_gff = use_gff
        if use_gff:
            self.conv_semantic = bn_act_conv(in_c, 64, 3, 1, 1)
            self.conv_fpn = bn_act_conv(in_c, 64, 3, 1, 1)
            self.conv = bn_act_conv(64, 196 , 1, 1 ,0)
            self.gcn = nn.Sequential(
                GCNConv(in_c, 16),
                nn.ReLU(),
            )
            self.gcn2 = GCNConv(16, in_c)
        
    def forward(self, roi_feature:OrderedDict, semantic:torch.Tensor, boxes:torch.Tensor):
        roi_semantic = roi_align(semantic, boxes= boxes, output_size=self.output_size, spatial_scale=1.0, sampling_ratio=-1)
        if not self.use_gff:
            return roi_feature + roi_semantic
        else:
            affinity = self.conv(self.conv_semantic(roi_semantic) + self.conv_fpn(roi_feature)) # n,196,14,14
            if affinity.shape[0] == 0:
                return roi_feature
            affinity = affinity.reshape(affinity.shape[0], 196, 196) # n,196,196
            affinity = torch.sigmoid(affinity) # 0-1
            # each ROI do GFF:
            embeddings = torch.zeros_like(roi_feature) # n,256,14,14
            for n in range(roi_feature.shape[0]):
                values = affinity[n].flatten() #196*196 边权重
                edge_index = torch.ones_like(affinity[n]).nonzero().T.contiguous() # 边集 [2, 196*196]
                feature = roi_feature[n].reshape(-1, 196).T # 256,196 -> 196,256
                x1 = self.gcn(feature, edge_index, values)
                x2 = self.gcn2(x1, edge_index, values)
                x2 = x2.T.reshape(roi_feature.shape[1], self.output_size, self.output_size) # [196,256] -> [256,196] -> [256,14,14]
                embeddings[n] = x2
            return roi_semantic + embeddings
