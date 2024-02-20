from ..model.HoverNet import HoverNet, val_np_tp_hv
from .data_info import CONIC, PANNUKE,MONUSEG, DSB, CPM
import os


class HoverConfig1:
    """
    pq       SQ       DQ     AJI  multi_pq+
    0.489302 0.772661 0.66092 0.51402    0.45295
    """

    img_size = [1000,1000]
    lr = 2e-4
    num_classes = {"conic": 7, "pannuke": 6, "monuseg": 2, "dsb": 2, "cpm": 2}
    lr_s = [40, 60]
    weight_decay = 5e-4
    batch = 4
    model = HoverNet
    epoch = 70
    numpy_result_dir = "numpy_result"

    def __init__(self, ds, device, ds_num=1) -> None:
        self.device = device
        self.ds = ds
        self.ds_num = ds_num
        self.exp_data_dir = f"exp_data/{ds}/{self.__class__.__name__}"
        if not os.path.exists(self.exp_data_dir):
            os.makedirs(self.exp_data_dir)
        self.model_para = f"model_para/{ds}/{ds_num}/{self.__class__.__name__}"
        self.num_classes = self.num_classes[ds]
        if not os.path.exists(self.model_para):
            os.makedirs(self.model_para)
        self.figure_dir = os.path.join(
            f"figures/{ds}/{ds_num}", f"{self.__class__.__name__}"
        )
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)
        self.LOG_DIR = f"log/{ds}/{ds_num}/{self.__class__.__name__}"
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        data = CONIC(ds_num) if ds == "conic" else PANNUKE(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data["train"]
        self.VAL_SET_INDEX = data["val"]
        self.TEST_SET_INDEX = data["test"]
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = (
            f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"
        )

class HoverConfig2(HoverConfig1):
    batch = 2
    weight_decay = 5e-4
    lr = 2e-5
    epoch = 30
    lr_s = [20]
    # load_model_para = f'model_para/{HoverConfig1.__name__}_mode{1}.pt'

class HoverPannuke(HoverConfig1):
    lr = 2e-4
    lr_s = [10, 20, 30]
    epoch = 35
    img_size = [256,256]
    gamma = 0.5
    batch = 4
    weight_decay = 5e-4
    def __init__(self, ds, device,ds_num ) -> None:
        super().__init__(ds, device, ds_num)
        data = PANNUKE(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data["train"]
        self.VAL_SET_INDEX = data["val"]
        self.TEST_SET_INDEX = data["test"]
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = (
            f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"
        )

class HoverNetMoNuSeg(HoverConfig1):
    lr =2e-4
    epoch = 600
    img_size = [256,256]
    lr_s = [300, 500]
    gamma = 0.5
    batch = 4
    weight_decay = 5e-4
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = MONUSEG(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data["train"]
        self.VAL_SET_INDEX = data["val"]
        self.TEST_SET_INDEX = data["test"]
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = (
            f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"
        )

class HoverNetCPM(HoverNetMoNuSeg):
    lr =2e-4
    epoch = 500
    lr_s = [100 ,300, 400]
    batch = 8
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = CPM(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data["train"]
        self.VAL_SET_INDEX = data["val"]
        self.TEST_SET_INDEX = data["test"]
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = (
            f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"
        )

class HoverNetDSB(HoverNetCPM):
    epoch = 80
    batch =4 
    lr_s = [20, 40, 60]
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data= DSB(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data["train"]
        self.VAL_SET_INDEX = data["val"]
        self.TEST_SET_INDEX = data["test"]
        self.DATASET_FP = data.dataset_dir