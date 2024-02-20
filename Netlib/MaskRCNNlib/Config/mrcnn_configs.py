import os
import torch
import numpy as np
import random
from .data_info import CONIC, PANNUKE, MONUSEG , DSB, CPM
from ..model.GeneralizedRCNN import (
    MRCNN,
    Resnet101FPN,
    Resnet50FPN,
    RPN_,
    AdaptiveFeaturePooling,
    BoxHead,
)


class AnchorWH:
    _conic_wh = torch.tensor(
        [
            [5.3540, 5.4052],
            [7.9408, 7.9664],
            [10.3460, 10.2176],
            [12.6745, 12.4069],
            [15.3027, 14.8266],
            [17.9222, 17.2318],
            [20.4959, 19.7666],
            [23.0637, 22.3939],
            [25.6120, 24.9868],
            [28.3959, 27.4598],
            [30.5813, 30.7583],
            [33.7484, 32.6694],
            [36.7553, 35.0461],
            [36.5035, 40.8511],
            [50.0039, 47.4951],
        ]
    )
    _pannuke_wh = torch.tensor(
        [
            [4.0000, 4.0000],
            [7.9223, 7.8770],
            [12.6948, 12.6294],
            [15.1480, 15.3712],
            [17.3568, 17.1002],
            [19.9136, 19.4749],
            [22.7131, 22.0952],
            [25.2366, 24.6645],
            [27.7057, 27.2810],
            [30.7165, 30.3411],
            [33.5621, 33.3514],
            [35.7139, 35.5922],
            [38.4946, 38.2610],
            [41.8065, 41.3216],
            [48.4015, 48.4544],
        ]
    )

    def __init__(self, ds, num_anchors) -> None:
        if ds == "pannuke":
            self.anchorwh = self._pannuke_wh
            if num_anchors != 15:
                self.anchorwh = torch.load("Netlib/MaskRCNNlib/tools/anchors_wh_pannuke.pt")[num_anchors]
        else:
            self.anchorwh = self._conic_wh
            if num_anchors != 15:
                self.anchorwh = torch.load("Netlib/MaskRCNNlib/tools/anchors_wh.pt")[num_anchors]
                # self.anchorwh= torch.load('tools/anchors_wh.pt')[num_anchors]

def setup_seed(seed: int, benchmark=True):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = benchmark
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.enabled = benchmark

color = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    dtype=torch.float32,
    device="cpu",
)

class MRCNNConfig1:
    'whitout any improvement'
    batch = 4
    lr = 2e-4
    num_anchors = 14
    lr_s: list = [400, 700, 950]
    epoch = 1000
    gamma = 0.5
    # MRCNN:
    # backbone--------------------
    BACKBONE = Resnet50FPN
    BOTTOM_UP = False
    # PROPAOSAL_GENERATOR----------
    PROPAOSAL_GENERATOR = RPN_
    STRIDE = [4, 8]
    RPN_POS_THRESHOLD = 0.66
    RPN_FRACTION_RATIO = 0.3
    BOX_WEIGHT = [10.0, 10.0, 5.0, 5.0]
    NMS_THRESHOLD = 0.2
    PRE_NMS_K = 2000
    POST_NMS_K = 1000

    ROI_HEAD = AdaptiveFeaturePooling
    ROI_RESOLUTION = 28
    STAGE2_MAX_PROPOSAL = [800]
    # detection--------------------
    BOX_DETECTION = BoxHead
    EXPAND = False
    EXPAND_RATIO = 0.2
    USE_GT_BOX = False
    STAGE2_SAMPLE_RATIO = 7.0
    ROI_POS_THRESHOLD = [0.6]
    POST_DETECTION_SCORE_THRESHOLD = 0.5
    NUM_CLASSES = 7
    DETECTION_PER_IMG = 400
    # semantic segmentation--------
    USE_SEMANTIC = False
    SEG_STRIDE = [2, 4, 8]
    FUSE_FEATURE = None
    weight_decay = 6e-4
    img_size = [256, 256]
    model = MRCNN

    def __init__(self, ds, device, ds_num=1) -> None:
        self.device = device
        self.anchor_wh = AnchorWH(ds, self.num_anchors).anchorwh
        self.exp_data_dir = f"exp_data/{ds}/{self.__class__.__name__}"
        if not os.path.exists(self.exp_data_dir):
            os.makedirs(self.exp_data_dir)
        self.model_para = (
            f"model_para/{ds}/{ds_num}/{self.__class__.__name__}"
        )
        if not os.path.exists(self.model_para):
            os.makedirs(self.model_para)
        self.figure_dir = os.path.join(f"figures/{ds}/{ds_num}", f"{self.__class__.__name__}")
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)
        self.LOG_DIR = f"log/{ds}/{ds_num}/{self.__class__.__name__}"
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        data = CONIC(ds_num) if ds == "conic" else PANNUKE(ds_num) 
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = f"{ds}/{ds_num}/{self.__class__.__name__}_test_result"

class MrcnnMonuSeg(MRCNNConfig1):
    batch = 8
    lr = 2e-4
    epoch = 800
    lr_s = [200, 400, 600]

    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = MONUSEG(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir

class MrcnnPannuke(MrcnnMonuSeg):
    epoch = 40
    lr_s = [20, 30, 35]

    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = PANNUKE(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir

class MrcnnCPM(MrcnnMonuSeg):
    batch = 8
    lr = 2e-4
    epoch = 800

    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = CPM(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir


class MrcnnDSB(MrcnnMonuSeg):
    batch = 8
    epoch = 120
    lr_s = [40, 80, 100]

    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = DSB(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir



class Infer_MRCNN:
    img_size = [256, 256]
    model = MRCNN
    # backbone--------------------
    BACKBONE = Resnet50FPN
    BOTTOM_UP = False
    # PROPAOSAL_GENERATOR----------
    PROPAOSAL_GENERATOR = RPN_
    STRIDE = [4, 8]
    RPN_POS_THRESHOLD = 0.7
    RPN_FRACTION_RATIO = 0.3
    BOX_WEIGHT = [10.0, 10.0, 5.0, 5.0]
    NMS_THRESHOLD = 0.2
    PRE_NMS_K = 2000
    POST_NMS_K = 1000

    ROI_HEAD = AdaptiveFeaturePooling
    ROI_RESOLUTION = 28
    STAGE2_MAX_PROPOSAL = [800]
    # detection--------------------
    BOX_DETECTION = BoxHead
    EXPAND = False
    EXPAND_RATIO = 0.2
    USE_GT_BOX = False
    STAGE2_SAMPLE_RATIO = 5.0
    ROI_POS_THRESHOLD = [0.6]
    POST_DETECTION_SCORE_THRESHOLD = 0.5
    NUM_CLASSES = 7
    DETECTION_PER_IMG = 400
    # semantic segmentation--------
    USE_SEMANTIC = False
    SEG_STRIDE = [2, 4, 8]
    FUSE_FEATURE = None
    num_anchors = 14
    model_para = "model_para/MRCNNConfig1_mode3.pt"


class Infer_MRCNN_2(Infer_MRCNN):
    STRIDE = [4, 8 ]
    model_para = "model_para/MRCNNConfig2_mode3.pt"
    USE_SEMANTIC = True



