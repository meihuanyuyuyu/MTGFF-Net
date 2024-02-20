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
    clip_boxes_to_image,
)
from torchvision.utils import draw_segmentation_masks, save_image
import numpy as np


def remove_big_boxes(boxes: torch.Tensor, size: float) -> torch.Tensor:
    # 去除大框
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    keep = (h < size) & (w < size)
    return keep



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

