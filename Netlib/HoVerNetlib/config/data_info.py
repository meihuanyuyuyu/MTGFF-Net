import torch

class CONIC(object):
    classes = 7
    dataset_dir = "data/CoNIC_Challenge"
    def __init__(self,split_num):
        self.ds_index = torch.load(f"splitted_index/conic_splitted_indices_{split_num}.pt")
        self.split_sets_path = f"splitted_index/conic_splitted_indices_{split_num}.pt"
    
    def __getitem__(self, index):
        return self.ds_index[index]
        

class PANNUKE(object):
    classes = 6
    dataset_dir = "data/PanNuke"
    def __init__(self,split_num):
        self.ds_index = torch.load(f"splitted_index/pannuke_splitted_indices_{split_num}.pt")
        self.split_sets_path = f"splitted_index/pannuke_splitted_indices_{split_num}.pt"
    
    def __getitem__(self, index):
        return self.ds_index[index]


class MONUSEG(object):
    classes = 2
    dataset_dir = "data/monuseg"
    def __init__(self,split_num):
        self.ds_index = torch.load('splitted_index/monuseg_splitted_indices.pt')
    
    def __getitem__(self, index):
        return self.ds_index[index]
    
class DSB(object):
    classes = 2
    dataset_dir = "data/DSB2018"
    def __init__(self,split_num):
        self.ds_index = torch.load('splitted_index/dsb_splitted_indices.pt')
    
    def __getitem__(self, index):
        return self.ds_index[index]

class CPM(object):
    classes = 2
    dataset_dir = "data/cpm17"
    def __init__(self,split_num):
        self.ds_index = torch.load('splitted_index/cpm_splitted_indices.pt')
    
    def __getitem__(self, index):
        return self.ds_index[index]
    