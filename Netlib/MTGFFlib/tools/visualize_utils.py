import os
from typing import OrderedDict
import torch
from torchvision.utils import draw_bounding_boxes, make_grid, save_image
from ..model.MaskRCNN_utils import roi_outputs


class Visualizer:
    COLOR = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        dtype=torch.float32,
        device="cpu",
    )
    CONIC_CLASSES = ["bag", "neu", "epi", "lym", "pla", "eos", "con"]
    PANNUKE_CLASSES = ["bag", "inf", "neo", "con", "dead", "epi"]

    def __init__(
        self, img_rgb: torch.Tensor, gt=None, color=COLOR, label=CONIC_CLASSES
    ) -> None:
        "only support 1 img,3 dims"
        self.img = img_rgb
        self.label = label
        self.color = color
        self.pic = None
        self.gt = ((color[gt[1]].permute(2, 0, 1)) * 255).to(dtype=torch.uint8)

    def draw_predict(self, preds: OrderedDict, s1_boxes: torch.Tensor = None):
        r"keys:pred_boxes,scores,pred_classes,pred_masks"


        boxes = preds["pred_boxes"] if "pred_boxes" in preds else None
        scores = preds["scores"] if "scores" in preds else None
        classes = preds["pred_classes"] if "pred_classes" in preds else None
        masks = preds["pred_masks"] if "pred_masks" in preds else None
        semantic_masks: torch.Tensor = (
            preds["pred_semantic_masks"] if "semantic_masks" in preds else None
        )

        if s1_boxes is not None:
            box1_pic = draw_bounding_boxes(self.gt, s1_boxes[0]) / 255
        else:
            box1_pic = self.gt / 255

        box2_pic = (
            draw_bounding_boxes(
                self.gt,
                boxes[0],
                labels=[f"{_:.2f}" for _ in scores[0].tolist()],
                font_size=1,
            )
            / 255
        )

        if masks:
            pred_mask = roi_outputs(
                masks, boxes, classes, output_size=self.img.shape[-2:]
            )[0]
            pic = torch.cat(
                [
                    box1_pic[None],
                    box2_pic[None],
                    self.gt[None] / 255,
                    self.color[pred_mask[1]].permute(2, 0, 1)[None],
                ]
            )
        else:
            pic = torch.cat([box1_pic[None], box2_pic[None], self.gt[None] / 255])

        if semantic_masks:
            pic = torch.cat(
                [
                    pic,
                    semantic_masks[0:1, None].expand(
                        1, 3, self.img.shape[-2], self.img.shape[-1]
                    ),
                ],
                dim=0,
            )

        self.pic = torch.cat([self.img[None], pic], dim=0) # 4,3,256,256

    def save(self, dir, num):
        assert self.pic is not None, "please call draw_predict first"
        grid = make_grid(self.pic, 2)
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_image(grid, os.path.join(dir, "{i}.png".format(i=num)))

