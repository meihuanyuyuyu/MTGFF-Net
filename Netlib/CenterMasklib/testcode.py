from centermask.config.centermask_config import centermaskres50
import torch
from detectron2.config.defaults import _C 
from detectron2.modeling import build_model


cfg = _C.clone()
cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
cfg.merge_from_file("CenterMasklib/centermask/configs_yaml/conic_centermask50.yaml")
cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 1500
cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.35
model= build_model(cfg)
model.load_state_dict(torch.load('model_para/conic/1/centermaskres50/best.pth'))
for n,p in model.named_parameters():
    print(n,p.shape)
model.eval()

from CentermaskInfer import CenteMaskInfer
from centermask.train_net.tools.dataset import CentermaskCONICData

data_set = CentermaskCONICData('data/CoNIC_Challenge',28)

data,label = data_set[0]
print(data['image'].shape,data["image"].max(),data["image"].min())
#print(len(data['instances'].gt_classes))
import matplotlib.pyplot as plt

img = data['image'].to('cuda')
preds = model([{'image':img}])
print(preds)
