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
    dataset_dir = "data/PANNUKE"
    def __init__(self,split_num):
        self.ds_index = torch.load(f"splitted_index/pannuke_splitted_indices_{split_num}.pt")
        self.split_sets_path = f"splitted_index/pannuke_splitted_indices_{split_num}.pt"
    
    def __getitem__(self, index):
        return self.ds_index[index]

class MoNuSeg(object):
    classes = 2
    dataset_dir = "data/MONUSEG/data"
    def __init__(self,split_num):
        self.ds_index = torch.load('splitted_index/monuseg_splitted_indices.pt')
    
    def __getitem__(self, index):
        return self.ds_index[index]

