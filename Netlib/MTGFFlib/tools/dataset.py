import os
from torchvision.ops import masks_to_boxes, remove_small_boxes, roi_align
from torch.nn.functional import one_hot
import csv
import scipy.io as sio
import torch
import numpy as np
from PIL import Image
from skimage.morphology import erosion,dilation
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset, Subset, DataLoader
from .utils import remove_big_boxes

class MtgffMonusegDataset(Dataset):
    # semantic: boundry 1, interior 2, background 0.
    def __init__(self, masks_size=28, img_size = [256,256], transfs: list = None, train = True) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        self.img_size = img_size
        print("Loading data into memory")
        if train:
            self.imgs = torch.load("data/monuseg/imgs.pt")
            self.labels = torch.from_numpy(np.load("data/monuseg/labels.npy").astype(np.int32)).permute(0,3,1,2).contiguous()
        else:
            self.imgs = torch.load("data/monuseg/MoNuSegTestData/imgs.pt")
            self.labels = torch.from_numpy(np.load("data/monuseg/MoNuSegTestData/labels.npy").astype(np.int32)).permute(0,3,1,2).contiguous()
        print("Finishing loading data")

    def __len__(self):
        return len(self.imgs)
    
    def transforms(self, imgs, labels):
        for _ in self.transfs:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def generate_targets_boxes(self, labels: torch.Tensor, mask_size: int = 28):
        max_instance = labels[0].max()
        if max_instance == 0:
            return (
                torch.zeros(0, 4),
                torch.zeros(0, mask_size, mask_size, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(*self.img_size,dtype=torch.long)
            )
        semantic_map = torch.zeros_like(labels[0])
        semantic_map[(labels[0]!=0)] = 2
        # -----------------------boxes------------------------------
        instances = torch.zeros((0,labels.shape[-2],labels.shape[-1]),device=labels.device, dtype=labels.dtype)
        for idx in range(1,max_instance+1):
            if(labels[0]==idx).sum() != 0:
                instances = torch.cat([instances, (labels[0]==idx).float().unsqueeze(0)],dim=0)
                boundry = torch.from_numpy(dilation(labels[0]==idx).astype(np.int32)- erosion(labels[0]==idx).astype(np.int32)).bool()
                semantic_map[boundry] = 1
        boxes = masks_to_boxes(instances)
        # enlarge box boundary
        boxes[:, 2:] = boxes[:, 2:] + 1.0
        boxes[:, :2] = boxes[:, :2] - 1.0
        index = remove_big_boxes(boxes, 70)
        boxes = boxes[index]
        instances = instances[index]  # num,256,256

        index = remove_small_boxes(boxes, 2)
        boxes = boxes[index]
        instances = instances[index]
        # ------------------------masks--------------------------------
        # grids = box2grid(boxes,[mask_size,mask_size])
        index = torch.arange(0, len(boxes), device=boxes.device).unsqueeze(1)
        rois = torch.cat([index, boxes], dim=1)
        masks = (
            roi_align(
                labels[None, 1:] * instances.unsqueeze(1).float(),
                rois,
                output_size=mask_size,
                spatial_scale=1,
                sampling_ratio=1,
                aligned=True,
            )
            .round()
            .long()
            .squeeze(1)
        )
        # masks = grid_sample(labels[None,1:].float()*instances.unsqueeze(1),grids,align_corners=True).round().long().squeeze(1) #rois,28,28
        # masks =crop(labels[None,1:].float()*instances.unsqueeze(0),boxes[:])
        cls = (
            one_hot(masks, num_classes=7)
            .permute(0, 3, 1, 2)
            .sum(dim=[2, 3])[:, 1:]
            .argmax(-1)
            + 1
        )
        return boxes, masks.bool().long(), cls, semantic_map.long()

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls, semantic_map = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls, semantic_map

import scipy.io as scio

class MtgffCPMDataset(MtgffMonusegDataset):
    def __init__(self, dataset_dir, masks_size=28, img_size=[256, 256], transfs: list = None, train=True) -> None:
        self.transfs = transfs
        self.masks_size = masks_size
        self.img_size = img_size
        self.train_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'test')
        print("Loading dataset from", dataset_dir)
        self.imgs_dir = os.listdir(os.path.join(self.train_dir, 'Images'))
        self.imgs_dir.sort()
        self.labels = os.listdir(os.path.join(self.train_dir, 'Labels'))
        self.labels.sort()
    
    def __len__(self):
        return len(self.imgs_dir)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.train_dir, 'Images', self.imgs_dir[index]))
        img = pil_to_tensor(img).float() / 255
        label = scio.loadmat(os.path.join(self.train_dir, 'Labels', self.labels[index]))['inst_map']
        label = torch.from_numpy(label).long() # h,w
        label = torch.stack([label,(label!=0)*1],dim=0) # 2,h,w
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls, semantic_map = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls, semantic_map

class MtgffDSBdataset(MtgffMonusegDataset):
    def __init__(self, dataset_dir, masks_size=28, img_size=[256, 256], transfs: list = None, train=True) -> None:
        self.transfs = transfs
        self.masks_size = masks_size
        self.img_size = img_size
        self.train_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'test')
        print("Loading dataset from", dataset_dir)
        self.data_dir = os.listdir(self.train_dir)
        self.data_dir.sort()
    
    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.train_dir, self.data_dir[index], "images", self.data_dir[index]+".png")).convert('RGB')
        img = pil_to_tensor(img).float() / 255
        instances = os.listdir(os.path.join(self.train_dir, self.data_dir[index], "masks"))
        instances.sort()
        label = torch.zeros(2, img.shape[1], img.shape[2], dtype=torch.long)
        for idx, ins in enumerate(instances):
            mask = Image.open(os.path.join(self.train_dir, self.data_dir[index], "masks", ins))
            mask = pil_to_tensor(mask).bool() * 1
            label[0][label[0]==0] += mask[0][label[0]==0] * (idx+1)
            label[1][label[1]==0] += mask[0][label[1]==0].long()
        #print(label.shape,' ', img.shape)
        if len(self.transfs) != 0:
            img, label = self.transforms(img, label)
        boxes, masks, cls, semantic_map = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls, semantic_map

class MtgffConicDataset(MtgffMonusegDataset):
    def __init__(self, dataset_dir, masks_size=28, img_size=[256, 256], transfs: list = None, train=True) -> None:
        self.transfs = transfs
        self.masks_size = masks_size
        self.img_size = img_size
        print("Loading data into memory")
        self.imgs = torch.load("data/conic/imgs.pt")
        self.labels = torch.load("data/conic/labels.pt").long()
        print("Finishing loading data")

    def transforms(self, imgs, labels):
        for _ in self.transfs:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls, semantic = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls, semantic

class MtgffPannukeDataset(MtgffMonusegDataset):
    def __init__(self, dataset_dir, masks_size=28, img_size=[256, 256], transfs: list = None, train=True) -> None:
        self.transfs = transfs
        self.masks_size = masks_size
        self.img_size = img_size
        print("Loading data into memory")
        self.labels = torch.from_numpy(np.load("data/PanNuke/masks.npy").astype(np.int32)).permute(0,3,1,2).contiguous().long()
        self.imgs_list = os.listdir("data/PanNuke/images")
        self.imgs_list.sort()

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join("data/PanNuke/images", self.imgs_list[index]))
        img = pil_to_tensor(img) / 255
        label = self.labels[index]
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls, semantic = self.generate_targets_boxes(label, mask_size=self.masks_size)
        return img, label, boxes, masks, cls, semantic

def collect_fn_semantic(data):
    return [
        torch.stack(sub_data, dim=0) if _ == 0 or _ == 1 or _==5 else list(sub_data)
        for _, sub_data in enumerate(zip(*data))
    ]

def collect_fn(data):
    return [
        torch.stack(sub_data, dim=0) if _ == 0 or _ == 1 else list(sub_data)
        for _, sub_data in enumerate(zip(*data))
    ]
