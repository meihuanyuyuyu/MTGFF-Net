import torch
import os
from .data_info import CONIC,PANNUKE,MONUSEG, DSB, CPM
from Netlib.CDNetlib.model_unet_MandDandP import Unet


class CDNet_C:
    BATCH_SIZE = 16
    EPOCHS = 100
    LR = 2e-4
    MILESTONES = [40, 70, 90]
    WEIGHT_DECAY = 5e-4
    GAMMA = 0.1
    NUM_WORKERS = 16
    VAL_EPOCH = 10
    BACKBONE_NAME = "resnet50"
    model = Unet
    def __init__(self, ds, device, ds_num=1) -> None:
        self.DEVICE = device
        self.ds = ds
        self.ds_num = ds_num

        self.exp_data_dir = f"exp_data/{ds}/{self.__class__.__name__}"
        if not os.path.exists(self.exp_data_dir):
            os.makedirs(self.exp_data_dir)
        self.model_para = (
            f"model_para/{ds}/{ds_num}/{self.__class__.__name__}_{self.BACKBONE_NAME}"
        )
        if not os.path.exists(self.model_para):
            os.makedirs(self.model_para)
        self.figure_dir = os.path.join(f"figures/{ds}/{ds_num}", f"{self.__class__.__name__}")
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)
        self.LOG_DIR = f"log/{ds}/{ds_num}/{self.__class__.__name__}+_{self.BACKBONE_NAME}"
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        data = CONIC(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"

class CDNet_MoNuSeg(CDNet_C):
    BATCH_SIZE = 16
    EPOCHS = 1000
    LR = 2e-4
    MILESTONES = [200, 500, 700,900]
    WEIGHT_DECAY = 5e-4
    GAMMA = 0.5
    NUM_WORKERS = 8
    VAL_EPOCH = 30
    BACKBONE_NAME = "resnet50"
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = MONUSEG(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"

class CDNet_P(CDNet_C):
    BATCH_SIZE = 8
    EPOCHS = 120
    LR = 2e-4
    MILESTONES = [40, 70, 100]
    WEIGHT_DECAY = 5e-4
    GAMMA = 0.5
    NUM_WORKERS = 8
    VAL_EPOCH = 10
    BACKBONE_NAME = "resnet50"
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = PANNUKE(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"

class CDNet_CPM(CDNet_MoNuSeg):
    BATCH_SIZE = 8
    EPOCHS = 1000
    LR = 2e-4
    MILESTONES = [200, 500, 800,900]
    WEIGHT_DECAY = 5e-4
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = CPM(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir

class CDNet_DSB(CDNet_CPM):
    EPOCHS = 120
    MILESTONES = [40, 80, 90, 110]
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = DSB(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir
