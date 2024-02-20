import os
import torch
import numpy as np
import random
from .data_info import CONIC, PANNUKE, MONUSEG , DSB, CPM
from ..model.GeneralizedRCNN import (
    MTGFFNet,
    Resnet101FPN,
    FPNwithBottomup,
    Resnet50FPN,
    RPN_,
    FeatureFusion,
    AdaptiveFeaturePooling,
    BoxHead,
)

class AnchorWH:
    '''
    clusted anchor sizes for CONIC and PanNuke datasets.
    '''
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
                self.anchorwh = torch.load("Netlib/MTGFFlib/tools/anchors_wh_pannuke.pt")[num_anchors]
        else:
            self.anchorwh = self._conic_wh
            if num_anchors != 15:
                self.anchorwh = torch.load("Netlib/MTGFFlib/tools/anchors_wh.pt")[num_anchors]
 
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

class MTGFFConfig:
    batch = 4
    lr = 2e-4
    num_anchors = 14 # anchors per bin
    epoch = 1200
    lr_s: list = [300, 600, 900, 1100]
    gamma = 0.5
    # MTGFFNet
    USE_H_STAIN = True
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
    USE_DEFORM = True
    USE_GIOU = False
    # roi head---------------------
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
    USE_SEMANTIC = True
    SEG_STRIDE = [2, 4 ]
    FUSE_FEATURE = FeatureFusion
    weight_decay = 6e-4
    img_size = [256, 256]
    model = MTGFFNet
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
    
class MTGFFMonuSeg(MTGFFConfig):
    # AJI 0.595
    lr = 1e-4
    USE_DEFORM = False
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = MONUSEG(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir



class MTGFFMonuSeg3(MTGFFMonuSeg):
    # rfa 0.35: PQ 0.620846 0.751187 DQ 0.825906 AJI 0.641895 
    # 0.642033 0.7578 0.846431 0.661502 dice score:0.813770 [4,8]
    epoch = 1000
    USE_GIOU = False
    batch = 4
    NUM_CLASSES =2
    lr_s = [200,500, 800]
    RPN_POS_THRESHOLD = 0.68
    RPN_FRACTION_RATIO = 0.3
    PRE_NMS_K = 2500
    POST_NMS_K = 1500

class MTGFFMonuSeg2(MTGFFMonuSeg3):
    # 0.525384 0.735843 0.7138 0.556465 dice score:0.726058
    STRIDE = [4]

class MTGFFMonuSeg4(MTGFFMonuSeg3):
    # 0.649918 0.765517 0.848754 0.672544 dice score:0.819
    STRIDE = [4, 8 ,16]

       
class MTGFFCPM(MTGFFConfig):
    """
    PQ       SQ        DQ        AJI
    0.595741 0.762604 0.779211 0.627692
    
    """
    lr = 2e-4
    USE_DEFORM = False
    USE_H_STAIN = True
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = CPM(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = f"{ds}/{ds_num}/{self.__class__.__name__}_test_result"

class MTGFFCPM2(MTGFFCPM):
    """
    0.587741 0.757598 0.773094 0.622144 rpn_pos_threshold = 0.7
    0.528459 0.704377 0.699264 0.598335 rpn_pos_threshold = 0.6
    0.592139 0.752442 0.78527 0.619363 rpn_pos_threshold = 0.65
    0.51 resnet101 0.457933 0.770871 0.592468 0.424136
    0.577073 0.75646 0.759034 0.607358  resnet101 rfa 
    """
    weight_decay = 6e-4
    RPN_FRACTION_RATIO = 0.4
    RPN_POS_THRESHOLD = 0.67
    BACKBONE = Resnet50FPN

class MTGFFCPM3(MTGFFCPM):
    """ 
    expand box +2 DQ down
    expand box +0 DQ down
    0.389219 0.741232 0.522799 0.41678
    wd 1e-3 0.518849 0.750431 0.688808 0.542789   0.521774
    """
    RPN_FRACTION_RATIO = 0.3
    RPN_POS_THRESHOLD = 0.68
    BACKBONE = Resnet101FPN 
    STRIDE = [4, 8, 16]
    
class MTGFFCPM4(MTGFFCPM):
    """ 
    resnet101 + dropout 0.5
    0.591991 0.747234 0.78906 0.625328  [4]
    0.610049 0.756137 0.804114 0.636042   0.608732 [4,8]
    0.612431 0.761643 0.801921 0.648775   0.613154 [4,8,16]

    """
    RPN_FRACTION_RATIO = 0.3
    RPN_POS_THRESHOLD = 0.68
    BACKBONE = Resnet101FPN 
    STRIDE = [4]

class MTGFFDSB(MTGFFConfig):
    """
    0.711493 0.826533 0.856662 0.743072
    """
    lr = 2e-4
    batch = 4
    epoch = 100
    lr_s = [30, 60, 70, 90]
    STRIDE = [4, 8, 16]
    BACKBONE = Resnet101FPN 
    RPN_POS_THRESHOLD = 0.68
    RPN_FRACTION_RATIO = 0.4
    USE_DEFORM = False
    USE_H_STAIN = False
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = DSB(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir

class MTGFFDSB2(MTGFFDSB):
    """
    +dropout 0.5 
    """

class MTGFFDSB3(MTGFFDSB):
    """
    0.723987 0.841502 0.856557 0.731735 
    """
    STRIDE = [4, 8]

class MTGFFDSB4(MTGFFDSB):
    """
    0.709815 0.848081 0.844492 0.73066
    """
    STRIDE = [4]

class MTGFFConic(MTGFFConfig):
    lr = 1e-4
    epoch = 60
    lr_s = [20, 40, 50, 55]
    NUM_CLASSES = 7
    BACKBONE = Resnet101FPN 
    RPN_POS_THRESHOLD = 0.67
    RPN_FRACTION_RATIO = 0.3
    PRE_NMS_K = 2500
    batch = 2
    POST_NMS_K = 1500
    USE_DEFORM = False
    USE_H_STAIN = True
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = CONIC(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir

class MTGFFPanNuke(MTGFFConfig):
    """
    0.528459 0.704377 0.699264 0.598335
    """
    epoch = 80
    BACKBONE = Resnet101FPN 
    RPN_POS_THRESHOLD = 0.67
    RPN_FRACTION_RATIO = 0.3
    lr_s: list = [30, 60, 70]
    lr = 1e-4
    batch = 16
    USE_DEFORM = False
    USE_H_STAIN = True
    num_anchors = 15
    NUM_CLASSES = 6
    def __init__(self, ds, device, ds_num=1) -> None:
        super().__init__(ds, device, ds_num)
        data = PANNUKE(ds_num)
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.DATASET_FP = data.dataset_dir