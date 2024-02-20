import os
from torchvision.ops import masks_to_boxes, remove_small_boxes
from torch.nn.functional import one_hot
from typing import List
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset, Subset, DataLoader
from .utils import remove_big_boxes
from detectron2.structures import BoxMode
from detectron2.data.detection_utils import annotations_to_instances
import cv2


class CentermaskCONICData(Dataset):
    def __init__(self,dataset_dir:str , masks_size=14, img_size=[256,256], transfs: list = []) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        self.img_size = img_size
        print("Loading data into memory")
        self.imgs = (
            torch.from_numpy(
                np.load(os.path.join(dataset_dir, "images.npy")).astype(np.float64) / 255
            )
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.labels = (
            torch.from_numpy(
                np.load(os.path.join(dataset_dir, "labels.npy")).astype(np.float64)
            )
            .long()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        print("Finishing loading data")

    @staticmethod
    def mask2polgy(mask):
        contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        segmentation = []
        for contour in contours:
            contour_list = contour.flatten().tolist()
            if len(contour_list) > 4:
                segmentation.append(contour_list)
        if len(segmentation) == 0:
            return None
        return segmentation
    
    def transforms(self, imgs, labels):
        for _ in self.transfs:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def generate_targets_boxes(
        self, labels: torch.Tensor, mask_size: int = 14
    ) -> List[dict]:
        # need bbox bbox_mode,categoryID,segmentation
        ann = []
        max_instance = labels[0].max()
        if max_instance == 0:
            return []
        instances: torch.Tensor = one_hot(labels[0], max_instance + 1).permute(2, 0, 1)[
            1:
        ]
        boxes = masks_to_boxes(instances)
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
        index = remove_small_boxes(boxes, 3)
        instances = instances[index]
        boxes = boxes[index]
        index = remove_big_boxes(boxes, 80)
        boxes = boxes[index]
        instances = instances[index]  # num,256,256

        # cls = one_hot(labels[1:]*instances,num_classes=7).permute(0,3,1,2).sum(dim=[2,3])[:,1:].argmax(-1)
        # masks = grid_sample(labels[None,1:].float()*instances.unsqueeze(1),grids,align_corners=True).round().long().squeeze(1) #rois,28,28
        # masks =crop(labels[None,1:].float()*instances.unsqueeze(0),boxes[:])
        cls = (
            one_hot(labels[1:] * instances, num_classes=7)
            .permute(0, 3, 1, 2)
            .sum(dim=[2, 3])[:, 1:]  # void background
            .argmax(-1)  # postive class start from index 0
        )

        for box, one_cls, mask in zip(
            boxes, cls, instances.squeeze(1).to(dtype=torch.uint8)
        ):
            ins_dict = {}
            ins_dict.update(
                {
                    "bbox": box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": one_cls.item(),
                }
            )
            ins_dict.update(
                {
                    "segmentation": CentermaskCONICData.mask2polgy(
                        np.asfortranarray(mask)
                    )
                }
            )
            if ins_dict["segmentation"] is None:
                continue
            ann.append(ins_dict)
        return ann

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if len(self.transfs) != 0:
            img, label = self.transforms(img, label)
        anns = self.generate_targets_boxes(label, mask_size=self.masks_size)
        if len(anns) == 0:
            ins = None
        else:
            ins = annotations_to_instances(anns, (256, 256), "polygon")
        return {"image": img, "instances": ins}, label


# PN Dataset===================================================================================

class CentermaskPNCData(Dataset):
    def __init__(self, dataset_dir:str ,masks_size=28, transfs: list = None) -> None:
        super().__init__(masks_size, transfs)
        print("Loading data into memory")
        self.imgs_list = os.listdir("data/PANNUKE/images")
        self.imgs_list.sort(key=lambda x: int(x[:-4]))
        self.transfs = transfs
        self.labels = (
            torch.from_numpy(np.load("data/PANNUKE/masks.npy").astype(np.float32))
            .long()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        print(f"Finishing loading {len(self.imgs_list)} data")
    
    def generate_targets_boxes(
        self, labels: torch.Tensor, mask_size: int = 14
    ) -> List[dict]:
        # need bbox bbox_mode,categoryID,segmentation
        ann = []
        max_instance = labels[0].max()
        if max_instance == 0:
            return []
        instances: torch.Tensor = one_hot(labels[0], max_instance + 1).permute(2, 0, 1)[
            1:
        ]
        boxes = masks_to_boxes(instances)
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
        index = remove_small_boxes(boxes, 3)
        instances = instances[index]
        boxes = boxes[index]
        index = remove_big_boxes(boxes, 80)
        boxes = boxes[index]
        instances = instances[index]  # num,256,256

        # cls = one_hot(labels[1:]*instances,num_classes=7).permute(0,3,1,2).sum(dim=[2,3])[:,1:].argmax(-1)
        # masks = grid_sample(labels[None,1:].float()*instances.unsqueeze(1),grids,align_corners=True).round().long().squeeze(1) #rois,28,28
        # masks =crop(labels[None,1:].float()*instances.unsqueeze(0),boxes[:])
        cls = (
            one_hot(labels[1:] * instances, num_classes=7)
            .permute(0, 3, 1, 2)
            .sum(dim=[2, 3])[:, 1:]  # void background
            .argmax(-1)  # postive class start from index 0
        )

        for box, one_cls, mask in zip(
            boxes, cls, instances.squeeze(1).to(dtype=torch.uint8)
        ):
            ins_dict = {}
            ins_dict.update(
                {
                    "bbox": box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": one_cls.item(),
                }
            )
            ins_dict.update(
                {
                    "segmentation": CentermaskCONICData.mask2polgy(
                        np.asfortranarray(mask)
                    )
                }
            )
            if ins_dict["segmentation"] is None:
                continue
            ann.append(ins_dict)
        return ann
    
    def transforms(self,img,label):
        for _ in self.transfs:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def __getitem__(self, index):
        img = Image.open(os.path.join("data/PANNUKE/images", self.imgs_list[index]))
        img = pil_to_tensor(img) / 255
        label = self.labels[index]
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        anns = self.generate_targets_boxes(label, mask_size=self.masks_size)
        if len(anns) == 0:
            ins = None
        else:
            ins = annotations_to_instances(anns, (256, 256), "polygon")
        return {"image": img, "instances": ins}, label


def collect_fn_detectron(data):
    return [
        torch.stack(sub_data, dim=0) if _ != 0 else list(sub_data)
        for _, sub_data in enumerate(zip(*data))
    ]


class CenterMaksMoNuSegDS(Dataset):
    def __init__(self,dataset_dir,transf=[],train=True,mask_size = 14) -> None:
        super().__init__()
        self.transf = transf
        self.masks_size = mask_size
        self.train_dir = os.path.join(dataset_dir,'MoNuSegData') if train else os.path.join(dataset_dir,'MoNuSegTestData') 
        print("Loading dataset from", dataset_dir)
        if train:
            self.imgs = torch.load(os.path.join(self.train_dir,'TissueImages/imgs.pt'))
            self.labels = torch.from_numpy(np.load(os.path.join(self.train_dir,'Annotations/masks.npy')).astype(np.int16)).permute(0,3,1,2).flip(dims=[1]).long().contiguous()
        else:
            self.imgs = torch.load(os.path.join(self.train_dir,'imgs.pt'))
            self.labels = torch.from_numpy(np.load(os.path.join(self.train_dir,'masks.npy')).astype(np.int32)).permute(0,3,1,2).flip(dims=[1]).long().contiguous()
        print("Dataset loaded")
    
    def transfer(self,img,label):
        for t in self.transf:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.imgs)
    
    def generate_targets_boxes(
        self, labels: torch.Tensor, mask_size: int = 14
    ) -> List[dict]:
        # need bbox bbox_mode,categoryID,segmentation
        ann = []
        max_instance = labels[0].max()
        if max_instance == 0:
            return []
        instances: torch.Tensor = one_hot(labels[0], max_instance + 1).permute(2, 0, 1)[
            1:
        ]
        instances = instances[instances.sum(dim=[1,2])>0]
        boxes = masks_to_boxes(instances)
        boxes[:, 2:] = boxes[:, 2:]
        boxes[:, :2] = boxes[:, :2] 
        index = remove_small_boxes(boxes, 3)
        instances = instances[index]
        boxes = boxes[index]  # num,256,256

        # cls = one_hot(labels[1:]*instances,num_classes=7).permute(0,3,1,2).sum(dim=[2,3])[:,1:].argmax(-1)
        # masks = grid_sample(labels[None,1:].float()*instances.unsqueeze(1),grids,align_corners=True).round().long().squeeze(1) #rois,28,28
        # masks =crop(labels[None,1:].float()*instances.unsqueeze(0),boxes[:])
        cls = torch.ones((len(boxes)),dtype=torch.int32)

        for box, one_cls, mask in zip(
            boxes, cls, instances.squeeze(1).to(dtype=torch.uint8)
        ):
            ins_dict = {}
            ins_dict.update(
                {
                    "bbox": box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": one_cls.item(),
                }
            )
            ins_dict.update(
                {
                    "segmentation": CentermaskCONICData.mask2polgy(
                        np.asfortranarray(mask)
                    )
                }
            )
            if ins_dict["segmentation"] is None:
                continue
            ann.append(ins_dict)
        return ann

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if len(self.transf) != 0:
            img, label = self.transfer(img, label)
        anns = self.generate_targets_boxes(label, mask_size=self.masks_size)
        if len(anns) == 0:
            ins = None
        else:
            ins = annotations_to_instances(anns, (256, 256), "polygon")
        return {"image": img, "instances": ins}, label
    


