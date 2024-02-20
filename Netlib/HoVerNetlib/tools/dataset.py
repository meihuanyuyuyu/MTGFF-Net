import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.ndimage import center_of_mass
import scipy.io as scio


class HoverNetCoNIC(Dataset):
    def __init__(self, img_size: list, transf: list = []) -> None:
        super().__init__()
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(img_size[0]), torch.arange(img_size[1]), indexing="xy"
        )
        self.transf = transf
        print("Loading data into memory")
        self.imgs = torch.load("data/conic/imgs.pt")
        self.labels =  torch.load("data/conic/labels.pt")
        print("Finishing loading data")

    def transforms(self, imgs, labels):
        for _ in self.transf:
            imgs, labels = _(imgs, labels)
        return imgs, labels

    def hv_label_generator(self, label: torch.Tensor):
        "label:2,h,w"
        hv = torch.zeros_like(label, dtype=torch.float32)
        instance_map = label[0]
        for idx in range(1, instance_map.max() + 1):
            mask = instance_map == idx
            mask_np = mask.numpy()
            center_y, center_x = center_of_mass(mask_np)
            # center_x,center_y = torch.from_numpy(center_x),torch.from_numpy(center_y)
            x = self.grid_x[mask] - center_x
            x[x < 0] = x[x < 0] / x.min().abs()  # -7 / 7
            x[x > 0] = x[x > 0] / x.max()  # 7,
            hv[0, mask] = x

            y = self.grid_y[mask] - center_y
            y[y < 0] = y[y < 0] / y.min().abs()
            y[y > 0] = y[y > 0] / y.max()
            hv[1, mask] = y
        return hv

    def __getitem__(self, index):
        # 2,256,256,3,256,256
        label = self.labels[index]
        img = self.imgs[index]
        if len(self.transf) != 0:
            img, label = self.transforms(img, label)
        hv = self.hv_label_generator(label)
        np = label[0].bool().long()
        nc = label[1].long()
        return img, hv, np, nc

# PN Dataset===================================================================================

class PannukeHover(HoverNetCoNIC):
    r"dataset to train Pannuke data"

    def __init__(self, img_size: list, transf: list = []) -> None:
        self.img_size = img_size
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(self.img_size[0]),
            torch.arange(self.img_size[1]),
            indexing="xy",
        )
        self.transf = transf
        print("Loading data into memory")
        self.imgs_list = os.listdir("data/PanNuke/images")
        self.imgs_list.sort(key=lambda x: int(x[:-4]))
        # to pack imgs

        self.labels = (
            torch.from_numpy(
                np.load("data/PanNuke/masks.npy").astype(
                    np.float64
                )
            )
            .long()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        print("Finishing loading data")

    def __getitem__(self, index):
        # 2,256,256,3,256,256
        img = Image.open(os.path.join("data/PanNuke/images", self.imgs_list[index]))
        img = pil_to_tensor(img) / 255
        label = self.labels[index]
        if len(self.transf) != 0:
            img, label = self.transforms(img, label)
        hv = self.hv_label_generator(label)
        np = label[0].bool().long()
        nc = label[1].long()
        return img, hv, np, nc
    

class HoverNetDataset_MoNuSeg(Dataset):
    def __init__(self,dataset_dir,img_size=[256,256],tranfs=[],train=True) -> None:
        super().__init__()
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(img_size[0]), torch.arange(img_size[1]), indexing="xy"
        )
        self.tranfs = tranfs
        self.train_dir = os.path.join(dataset_dir,'MoNuSeg2018TrainingData') if train else os.path.join(dataset_dir,'MoNuSegTestData') 
        #self.test_dir = os.path.join(dataset_dir,'MoNuSegTestData')
        
        print("Loading dataset from", dataset_dir)
        if train:
            self.imgs = torch.load(os.path.join(self.train_dir,'imgs.pt'))
            self.labels = torch.from_numpy(np.load(os.path.join(self.train_dir,'labels.npy')).astype(np.int32)).permute(0,3,1,2).contiguous()
        else:
            self.imgs = torch.load(os.path.join(self.train_dir,'imgs.pt'))
            self.labels = torch.from_numpy(np.load(os.path.join(self.train_dir,'labels.npy')).astype(np.int32)).permute(0,3,1,2).contiguous()
        print("Dataset loaded")

    def transfer(self,img,label):
        for t in self.tranfs:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.imgs)
    
    def hv_label_generator(self, label: torch.Tensor):
        "label:2,h,w"
        hv = torch.zeros_like(label, dtype=torch.float32)
        instance_map = label[0]
        for idx in range(1, instance_map.max() + 1):
            mask = instance_map == idx
            mask_np = mask.numpy()
            center_y, center_x = center_of_mass(mask_np)
            # center_x,center_y = torch.from_numpy(center_x),torch.from_numpy(center_y)
            
            x = self.grid_x[mask] - center_x
            if x.shape == torch.Size([0]):
                continue
            x[x < 0] = x[x < 0] / x.min().abs()  # -7 / 7
            x[x > 0] = x[x > 0] / x.max()  # 7,
            hv[0, mask] = x

            y = self.grid_y[mask] - center_y
            y[y < 0] = y[y < 0] / y.min().abs()
            y[y > 0] = y[y > 0] / y.max()
            hv[1, mask] = y
        return hv

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if len(self.tranfs) != 0:
            img, label = self.transfer(img, label)
        #print(label.shape,label[0].max())
        hv = self.hv_label_generator(label)
        np = label[1].bool().long() # 01掩码
        nc = label[1].long() # 类别掩码
        return img, hv, np, nc
        
class HoverNetDataset_CPM(Dataset):
    def __init__(self,dataset_dir,img_size=[256,256],tranfs=[],train=True):
        super().__init__()
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(img_size[0]), torch.arange(img_size[1]), indexing="xy"
        )
        self.tranfs = tranfs
        self.train_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'test')
        print("Loading dataset from", dataset_dir)
        self.imgs_dir = os.listdir(os.path.join(self.train_dir, 'Images'))
        self.imgs_dir.sort()
        self.labels = os.listdir(os.path.join(self.train_dir, 'Labels'))
        self.labels.sort()

    def transfer(self,img,label):
        for t in self.tranfs:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.imgs_dir)
    
    def hv_label_generator(self, label: torch.Tensor):
        "label:2,h,w"
        hv = torch.zeros_like(label, dtype=torch.float32)
        instance_map = label[0]
        for idx in range(1, instance_map.max() + 1):
            mask = instance_map == idx
            mask_np = mask.numpy()
            center_y, center_x = center_of_mass(mask_np)
            # center_x,center_y = torch.from_numpy(center_x),torch.from_numpy(center_y)
            
            x = self.grid_x[mask] - center_x
            if x.shape == torch.Size([0]):
                continue
            x[x < 0] = x[x < 0] / x.min().abs()  # -7 / 7
            x[x > 0] = x[x > 0] / x.max()  # 7,
            hv[0, mask] = x

            y = self.grid_y[mask] - center_y
            y[y < 0] = y[y < 0] / y.min().abs()
            y[y > 0] = y[y > 0] / y.max()
            hv[1, mask] = y
        return hv

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.train_dir, 'Images', self.imgs_dir[index]))
        img = pil_to_tensor(img).float() / 255
        label = scio.loadmat(os.path.join(self.train_dir, 'Labels', self.labels[index]))['inst_map']
        label = torch.from_numpy(label).long() # h,w
        label = torch.stack([label,(label!=0)*1],dim=0) # 2,h,w
        if len(self.tranfs) != 0:
            img, label = self.transfer(img, label)
        #print("img:", img.shape)
        #print(label.shape)
        hv = self.hv_label_generator(label)
        np = label[1].bool().long()
        nc = label[1].long()
        return img, hv, np, nc

class HoverNetDataset_DSB(Dataset):
    def __init__(self,dataset_dir,img_size=[256,256],tranfs=[],train=True) -> None:
        super().__init__()
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(img_size[0]), torch.arange(img_size[1]), indexing="xy"
        )
        self.tranfs = tranfs
        self.train_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'test')
        print("Loading dataset from", dataset_dir)
        self.data_dir = os.listdir(self.train_dir)
        self.data_dir.sort()

    def transfer(self,img,label):
        for t in self.tranfs:
            img,label = t(img,label)
        return img,label
    
    def __len__(self):
        return len(self.data_dir)
    
    def hv_label_generator(self, label: torch.Tensor):
        "label:2,h,w"
        hv = torch.zeros_like(label, dtype=torch.float32)
        instance_map = label[0]
        for idx in range(1, instance_map.max() + 1):
            mask = instance_map == idx
            mask_np = mask.numpy()
            center_y, center_x = center_of_mass(mask_np)
            # center_x,center_y = torch.from_numpy(center_x),torch.from_numpy(center_y)
            
            x = self.grid_x[mask] - center_x
            if x.shape == torch.Size([0]):
                continue
            x[x < 0] = x[x < 0] / x.min().abs()  # -7 / 7
            x[x > 0] = x[x > 0] / x.max()  # 7,
            hv[0, mask] = x

            y = self.grid_y[mask] - center_y
            y[y < 0] = y[y < 0] / y.min().abs()
            y[y > 0] = y[y > 0] / y.max()
            hv[1, mask] = y
        return hv

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
        if len(self.tranfs) != 0:
            img, label = self.transfer(img, label)
        hv = self.hv_label_generator(label)
        np = label[1].bool().long()
        nc = label[1].long()
        return img, hv, np, nc