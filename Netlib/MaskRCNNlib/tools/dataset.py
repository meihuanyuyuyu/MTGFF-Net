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
from .utils import remove_big_boxes , remove_small_boxes


class MRCNNLizardDataset(Dataset):
    def __init__(self,data_path, transf = [], mask_size=28, cropped_img_size=[256,256], set_indices =1) -> None:
        super().__init__()
        self.data_path = data_path
        self.transf = transf
        self.mask_size = mask_size
        self.img_size = cropped_img_size
        reader = csv.reader(open(os.path.join(data_path,"lizard_labels/Lizard_Labels/info.csv")))
        print('Reading csv file and loading data into memory...')
        self.train_list = []
        self.label_list = []
        for i,row in enumerate(reader):
            if i == 0:
                continue
            if row[0].startswith('c'):
                img_path = os.path.join(data_path,'lizard_images1/Lizard_Images1')
            else:
                img_path = os.path.join(data_path,'lizard_images2/Lizard_Images2')

            if int(row[-1]) == set_indices:
                img = pil_to_tensor(Image.open(os.path.join(img_path,row[0]+'.png')))/255
                self.train_list.append(img)
                label = sio.loadmat(os.path.join(data_path,f'lizard_labels/Lizard_Labels/Labels/{row[0]}.mat'))
                ins_map =torch.from_numpy(label['inst_map'])
                class_map = torch.from_numpy(label['class'])
                self.label_list.append((ins_map,class_map))
        print('Finish spliting and loading data into memory')
        
    def __len__(self):
        return len(self.train_list)
    
    def generate_targets_boxes(self, labels: torch.Tensor, class_labels, mask_size: int = 14):
        max_instance = labels.max()   
        classes = [] 
        classes_map = labels.clone()
        if max_instance == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0, mask_size, mask_size), dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(*self.img_size, dtype=torch.long),
                torch.zeros(*self.img_size, dtype=torch.long)
            )
        nums = 1
        for idx in range(1,max_instance+1):
            if len(labels[labels==idx])!=0:
                labels[labels==idx] = nums
                classes_map[classes_map==idx] = class_labels[idx-1]
                nums+=1
                classes.append(class_labels[idx-1])
            else:
                continue
        classes = torch.cat(classes)
        labels = labels.long()
        # -----------------------boxes------------------------------
        instances = one_hot(labels, labels.max()+1).permute(2, 0, 1)[1:] # n,h,w
        semantic_map = torch.zeros_like(labels, dtype=torch.long)
        semantic_map[(labels!=0)] = 2
        for i in range(len(instances)):
            boundary = torch.from_numpy(dilation(instances[i]) - erosion(instances[i])).bool()
            semantic_map[boundary] = 1
        # n,h,w
        boxes = masks_to_boxes(instances)
        boxes[:, 2:] = boxes[:, 2:] +1 
        boxes[:, :2] = boxes[:, :2] -1 
        index = remove_small_boxes(boxes, 1.5)
        instances = instances[index]
        boxes = boxes[index]
        index = remove_big_boxes(boxes, 70)
        boxes = boxes[index]
        instances = instances[index]  # num,256,256
        # ------------------------masks--------------------------------
        # grids = box2grid(boxes,[mask_size,mask_size])
        index = torch.arange(0, len(boxes), device=boxes.device).unsqueeze(1)
        rois = torch.cat([index, boxes], dim=1)
        masks = (
            roi_align(
                labels[None, None] * instances.unsqueeze(1).float(),
                rois,
                output_size=mask_size,
                spatial_scale=1,
                sampling_ratio=1,
                aligned=False,
            )
            .round()
            .long()
            .squeeze(1)
        )
        # masks = grid_sample(labels[None,1:].float()*instances.unsqueeze(1),grids,align_corners=True).round().long().squeeze(1) #rois,28,28
        # masks =crop(labels[None,1:].float()*instances.unsqueeze(0),boxes[:])
        return boxes, masks.bool().long(), classes, semantic_map, classes_map

    def transforms(self, imgs, labels):
        for t in self.transf:
            imgs, labels = t(imgs, labels)
        return imgs, labels

    def __getitem__(self, index):
        img = self.train_list[index]
        ins_map,classes = self.label_list[index]
        if len(self.transf)!= 0:
            img,ins_map = self.transforms(img,ins_map) #512,512
        boxes, masks, cls, semantic_map, classes_map =self.generate_targets_boxes(ins_map,classes ,mask_size=self.mask_size)
        label = torch.cat([ins_map[None],classes_map[None]],dim=0).long()
        return img, label ,boxes,masks,cls ,semantic_map
        
class ConicDataset(Dataset):
    def __init__(self, masks_size=28, transfs: list = None) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        print("Loading data into memory")
        self.imgs = torch.load("data/conic/imgs.pt")
        self.labels =  torch.load("data/conic/labels.pt")
        print("Finishing loading data")

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
            )
        # -----------------------boxes------------------------------
        instances = one_hot(labels[0], max_instance + 1).permute(2, 0, 1)[1:]  # n,h,w
        boxes = masks_to_boxes(instances)
        # enlarge box boundary
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
        index = remove_big_boxes(boxes, 70)
        boxes = boxes[index]
        instances = instances[index]  # num,256,256
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
        return boxes, masks.bool().long(), cls

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls

class MonusegDataset(Dataset):
    def __init__(self, masks_size=28, transfs: list = None, train = True) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        print("Loading data into memory")
        if train:
            self.imgs = torch.load("data/monuseg/MoNuSeg2018TrainingData/imgs.pt")
            self.labels = (
                torch.from_numpy(
                    np.load("data/monuseg/MoNuSeg2018TrainingData/labels.npy").astype(np.int32)
                )
                .long()
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            self.imgs = torch.load('data/monuseg/imgs.pt')
            self.labels = torch.from_numpy(np.load('data/monuseg/labels.npy').astype(np.int32)).long().permute(0,3,1,2).contiguous()
        print("Finishing loading data")

    def __len__(self):
        return len(self.imgs)
    
    def transforms(self, imgs, labels):
        for _ in self.transfs:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def generate_targets_boxes(self, labels: torch.Tensor, mask_size: int = 28):
        max_instance = labels[0].max()
        #print(labels[0].unique())
        if max_instance == 0:
            return (
                torch.zeros(0, 4),
                torch.zeros(0, mask_size, mask_size, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        # -----------------------boxes------------------------------
        instances = torch.zeros((0,labels.shape[-2],labels.shape[-1]),device=labels.device, dtype=labels.dtype)
        for idx in range(1,max_instance+1):
            if(labels[0]==idx).sum() != 0:
                instances = torch.cat([instances, (labels[0]==idx).float().unsqueeze(0)],dim=0)
        boxes = masks_to_boxes(instances)
        # enlarge box boundary
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
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
        return boxes, masks.bool().long(), cls

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls

import scipy.io as scio

class CPMDataset(Dataset):
    def __init__(self,dataset_dir, masks_size=28, transfs: list = None, train = True) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        self.train_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'test')
        print("Loading data into memory")
        self.imgs_dir = os.listdir(os.path.join(self.train_dir, 'Images'))
        self.imgs_dir.sort()
        self.labels = os.listdir(os.path.join(self.train_dir, 'Labels'))
        self.labels.sort()
    
    def __len__(self):
        return len(self.imgs_dir)
    
    def transforms(self, imgs, labels):
        for _ in self.transfs:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def generate_targets_boxes(self, labels: torch.Tensor, mask_size: int = 28):
        max_instance = labels[0].max()
        #print(labels[0].unique())
        if max_instance == 0:
            return (
                torch.zeros(0, 4),
                torch.zeros(0, mask_size, mask_size, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        # -----------------------boxes------------------------------
        instances = torch.zeros((0,labels.shape[-2],labels.shape[-1]),device=labels.device, dtype=labels.dtype)
        for idx in range(1,max_instance+1):
            if(labels[0]==idx).sum() != 0:
                instances = torch.cat([instances, (labels[0]==idx).float().unsqueeze(0)],dim=0)
        boxes = masks_to_boxes(instances)
        # enlarge box boundary
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
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
        cls = (
            one_hot(masks, num_classes=7)
            .permute(0, 3, 1, 2)
            .sum(dim=[2, 3])[:, 1:]
            .argmax(-1)
            + 1
        )
        return boxes, masks.bool().long(), cls

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.train_dir, 'Images', self.imgs_dir[index]))
        img = pil_to_tensor(img).float() / 255
        label = scio.loadmat(os.path.join(self.train_dir, 'Labels', self.labels[index]))['inst_map']
        label = torch.from_numpy(label).long() # h,w
        label = torch.stack([label,(label!=0)*1],dim=0)
        if self.transfs is not None:
            img, label = self.transforms(img, label)
        boxes, masks, cls = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls

class DSBDataset(Dataset):
    def __init__(self,dataset_dir, masks_size=28, transfs: list = None, train = True) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        self.train_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'test')
        print("Loading data into memory")
        self.data_dir = os.listdir(self.train_dir)
        self.data_dir.sort()
    
    def transfer(self,img,label):
        for t in self.transfs:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.data_dir)
    
    def generate_targets_boxes(self, labels: torch.Tensor, mask_size: int = 28):
        max_instance = labels[0].max()
        #print(labels[0].unique())
        if max_instance == 0:
            return (
                torch.zeros(0, 4),
                torch.zeros(0, mask_size, mask_size, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        # -----------------------boxes------------------------------
        instances = torch.zeros((0,labels.shape[-2],labels.shape[-1]),device=labels.device, dtype=labels.dtype)
        for idx in range(1,max_instance+1):
            if(labels[0]==idx).sum() != 0:
                instances = torch.cat([instances, (labels[0]==idx).float().unsqueeze(0)],dim=0)
        boxes = masks_to_boxes(instances)
        # enlarge box boundary
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
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
        return boxes, masks.bool().long(), cls

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
        if self.transfs is not None:
            img, label = self.transfer(img, label)
        boxes, masks, cls = self.generate_targets_boxes(
            label, mask_size=self.masks_size
        )
        return img, label, boxes, masks, cls


def collect_fn(data):
    return [
        torch.stack(sub_data, dim=0) if _ == 0 or _ == 1 else list(sub_data)
        for _, sub_data in enumerate(zip(*data))
    ]
