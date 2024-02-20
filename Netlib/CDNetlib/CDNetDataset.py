from .data_prepare.SegFix_offset_helper import Sobel, DTOffsetHelper
from skimage.morphology import dilation, disk, erosion, remove_small_objects
from scipy.ndimage import distance_transform_edt, gaussian_filter
from torch.utils.data import Dataset,DataLoader,Subset
import copy
import os
import torch

import numpy as np


class CDNetDataset_CONIC(Dataset):
    def __init__(self, dataset_dir, tranfs=[]) -> None:
        super().__init__()
        self.tranfs = tranfs
        print("Loading dataset from", dataset_dir)
        self.imgs = torch.load(os.path.join(dataset_dir, "imgs.pt"))
        self.labels =  torch.load(os.path.join(dataset_dir, "labels.pt"))
        print("Dataset loaded")

    def transfer(self, img, label):
        for t in self.tranfs:
            img, label = t(img, label)
        return img, label

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.tranfs is not None:
            img, label = self.transfer(img, label)
        instance_label = label[0]
        classes_label = label[1]
        boundry, point, direction = generate_CDNET_target(instance_label.numpy())
        classes_label = classes_label.long()
        boundry = torch.from_numpy(boundry).long()
        point = torch.from_numpy(point).float()
        direction = torch.from_numpy(direction).long()
        return img, classes_label, boundry, point, direction

import os
from PIL.Image import open
from torchvision.transforms.functional import pil_to_tensor

class CDNetDataset_PANNUKE(Dataset):
    def __init__(self, dataset_dir, tranfs=[]) -> None:
        super().__init__()
        self.tranfs = tranfs
        print("Loading dataset from", dataset_dir)
        self.img_dir = os.path.join(dataset_dir, "images")
        self.img_list = os.listdir(self.img_dir)
        self.img_list.sort(key=lambda x: int(x.split(".")[0]))
        self.labels = (
            torch.from_numpy(
                np.load(os.path.join(dataset_dir, "masks.npy")).astype(np.float32)
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        print("Dataset loaded")

    def transfer(self, img, label):
        for t in self.tranfs:
            img, label = t(img, label)
        return img, label

    def __getitem__(self, index):
        # need to be modified
        img = open(os.path.join(self.img_dir, self.img_list[index]))  # PIL image
        img = pil_to_tensor(img).float() / 255
        label = self.labels[index]
        if self.tranfs is not None:
            img, label = self.transfer(img, label)
        instance_label = label[0]
        classes_label = label[1]
        boundry, point, direction = generate_CDNET_target(instance_label.numpy())
        classes_label = classes_label.long()
        boundry = torch.from_numpy(boundry).long()
        point = torch.from_numpy(point).float()
        direction = torch.from_numpy(direction).long()
        return img, classes_label, boundry, point, direction

def generate_CDNET_target(label):
    r"Input:instance_label\
    1. 3-classes: background, cell, boundary,2.point 3. Direction Map: 8 directions"
    # blank label
    if np.max(label) == 0:
        return (
            np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8),
            np.zeros((label.shape[0], label.shape[1]), dtype=np.float32),
            np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8),
        )

    out = []
    label_inside = label
    # boundary map
    new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    new_label[label_inside > 0] = 1  # inside
    new_label = remove_small_objects(new_label.astype(np.bool8), 5).astype(np.uint8)
    new_label_inside = copy.deepcopy(new_label)

    boun_instance = np.logical_and(
        dilation(label_inside), np.logical_not(erosion(label_inside, disk(1)))
    )
    new_label[boun_instance > 0] = 2  # h,w
    out.append(new_label)

    # direction map

    height, width = label.shape[0], label.shape[1]

    distance_center_map = np.zeros((height, width), dtype=np.float32)

    dir_map = np.zeros((height, width, 2), dtype=np.float32)
    ksize = 11
    point_number = 0
    label_point = np.zeros((height, width), dtype=np.float32)

    mask = label
    markers_unique = np.unique(label)
    markers_len = len(np.unique(label)) - 1

    for k in markers_unique[1:]:
        nucleus = (mask == k).astype(np.int32)

        # 将距离最大的点作为中心点
        d_m = distance_transform_edt(nucleus)
        center = np.unravel_index(np.argmax(d_m), d_m.shape)

        local_maxi = [center]
        assert nucleus[center[0], center[1]] > 0
        label_point[center[0], center[1]] = 255.0

        nucleus = dilation(nucleus, disk(1))
        point_map_k = np.zeros((height, width), dtype=np.int32)
        point_map_k[local_maxi[0][0], local_maxi[0][1]] = 1
        int_pos = distance_transform_edt(1 - point_map_k)
        int_pos = int_pos * nucleus
        distance_center_i = (1 - int_pos / (int_pos.max() + 0.0000001)) * nucleus
        distance_center_map = distance_center_map + distance_center_i

        dir_i = np.zeros_like(dir_map)
        sobel_kernel = Sobel.kernel(ksize=ksize)
        dir_i = (
            torch.nn.functional.conv2d(
                torch.from_numpy(distance_center_i).float().view(1, 1, height, width),
                sobel_kernel,
                padding=ksize // 2,
            )
            .squeeze()
            .permute(1, 2, 0)
            .numpy()
        )
        dir_i[(nucleus == 0), :] = 0
        dir_map[(nucleus != 0), :] = 0
        dir_map += dir_i
        point_number = point_number + 1
    assert int(label_point.sum() / 255) == markers_len

    label_point_gaussian = gaussian_filter(label_point, sigma=2, order=0).astype(
        np.float16
    )
    out.append(label_point_gaussian / label_point_gaussian.max())

    # 角度
    """    angle = np.degrees(np.arctan2(dir_map[:, :, 0], dir_map[:, :, 1]))
    angle[new_label_inside == 0] = 0"""

    angle = np.degrees(np.arctan2(dir_map[:, :, 0], dir_map[:, :, 1]))

    angle[new_label_inside == 0] = 0
    vector = DTOffsetHelper.angle_to_vector(angle, return_tensor=False)
    # direction class
    label_direction = DTOffsetHelper.vector_to_label(vector, return_tensor=False)

    # input = instance level

    label_direction[new_label_inside == 0] = -1
    label_direction_new2 = label_direction + 1
    out.append(label_direction_new2)
    return out

class CDNetDataset_MoNuSeg(Dataset):
    def __init__(self,dataset_dir,tranfs=[],train=True):
        super().__init__()
        self.tranfs = tranfs
        self.train_dir = os.path.join(dataset_dir,'MoNuSeg2018TrainingData') if train else os.path.join(dataset_dir,'MoNuSegTestData') 
        #self.test_dir = os.path.join(dataset_dir,'MoNuSegTestData')
        
        print("Loading dataset from", dataset_dir)
        if train:
            self.imgs = torch.load(os.path.join(self.train_dir,'imgs.pt'))
            self.labels = torch.from_numpy(np.load(os.path.join(self.train_dir,'labels.npy')).astype(np.int16)).permute(0,3,1,2).contiguous()
        else:
            self.imgs = torch.load(os.path.join(self.train_dir,'imgs.pt'))
            self.labels = torch.from_numpy(np.load(os.path.join(self.train_dir,'labels.npy')).astype(np.int16)).permute(0,3,1,2).contiguous()
        print("Dataset loaded")
    
    def transfer(self,img,label):
        for t in self.tranfs:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.tranfs is not None:
            img,label = self.transfer(img,label)
        instance_label = label[0]
        classes_label = label[1]
        boundry, point, direction = generate_CDNET_target(instance_label.numpy())
        classes_label = classes_label.long()
        boundry = torch.from_numpy(boundry).long()
        point = torch.from_numpy(point).float()
        direction = torch.from_numpy(direction).long()
        return img, classes_label, boundry, point, direction

import PIL.Image as Image
import scipy.io as scio

class CDNetDataset_CPM(Dataset):
    def __init__(self,dataset_dir,tranfs=[],train=True):
        super().__init__()
        self.tranfs = tranfs
        self.train_dir = os.path.join(dataset_dir,"train") if train else os.path.join(dataset_dir,"test")
        print("Loading dataset from", dataset_dir)
        self.imgs_dir = os.listdir(os.path.join(self.train_dir,"Images"))
        self.imgs_dir.sort()
        self.labels_dir = os.listdir(os.path.join(self.train_dir,"Labels"))
        self.labels_dir.sort()

    def transfer(self,img,label):
        for t in self.tranfs:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.imgs_dir)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.train_dir, 'Images', self.imgs_dir[index]))
        img = pil_to_tensor(img).float() / 255
        label = scio.loadmat(os.path.join(self.train_dir, 'Labels', self.labels_dir[index]))['inst_map']
        label = torch.from_numpy(label).long() # h,w
        label = torch.stack([label,(label!=0)*1],dim=0) # 2,h,w
        if len(self.tranfs) != 0:
            img, label = self.transfer(img, label)
        instance_label = label[0]
        classes_label = label[1]
        boundry, point, direction = generate_CDNET_target(instance_label.numpy())
        classes_label = classes_label.long()
        boundry = torch.from_numpy(boundry).long()
        point = torch.from_numpy(point).float()
        direction = torch.from_numpy(direction).long()
        return img, classes_label, boundry, point, direction

class CDNetDataset_DSB(Dataset):
    def __init__(self,dataset_dir,tranfs=[],train=True):
        super().__init__()
        self.tranfs = tranfs
        self.train_dir = os.path.join(dataset_dir,"train") if train else os.path.join(dataset_dir,"test")
        print("Loading dataset from", dataset_dir)
        self.data_dir = os.listdir(self.train_dir)
        self.data_dir.sort()

    def transfer(self,img,label):
        for t in self.tranfs:
            img,label = t(img,label)
        return img,label
    
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
        if len(self.tranfs) != 0:
            img, label = self.transfer(img, label)
            
        instance_label = label[0]
        classes_label = label[1]
        boundry, point, direction = generate_CDNET_target(instance_label.numpy())
        classes_label = classes_label.long()
        boundry = torch.from_numpy(boundry).long()
        point = torch.from_numpy(point).float()
        direction = torch.from_numpy(direction).long()
        return img, classes_label, boundry, point, direction