from . import defaults
import os
from detectron2.modeling import build_model, GeneralizedRCNN
from .data_info import CONIC, PANNUKE,MoNuSeg


class centermaskres50:
    num_class = 7
    batch = 8
    lr = 2e-4
    resume = False
    weight_decay = 5e-4
    lr_s: list = [70, 90]
    epoch = 100
    NUM_WORKERS = 1
    numpy_result_dir = "numpy_result"

    def __init__(self, ds, device,ds_num) -> None:
        self.cfg = self.setup(device)
        self.model = build_model(self.cfg)
        self.DEVICE = device
        self.ds = ds
        self.ds_num = ds_num
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
        self.DATASET_DIR = data.dataset_dir
        self.load_model_para = f"{self.model_para}/best.pth"
        self.TEST_RESULT_DIR = f"{self.ds}/{ds_num}/{self.__class__.__name__}_test_result"

    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("Netlib/CenterMasklib/centermask/configs_yaml/centermask_R_50_FPN_ms_3x.yaml")
        # 模型settings：
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]

        # cfg.MODEL.RESNETS.NORM = "GN"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        # cfg.MODEL.RESNETS.DEPTH = 50
        # cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        # cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]

        # cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 1500
        # cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 800
        # cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 6000
        # cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 6000
        # cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4"]
        # cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16]
        # cfg.MODEL.FCOS.CENTER_SAMPLE = True
        # cfg.MODEL.FCOS.POS_RADIUS = (
        #     0.8  # it is used to enlarge or shrink the positive samples reg
        # )
        # cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.35
        # cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.35
        # cfg.MODEL.FCOS.NMS_TH = 0.18
        # cfg.MODEL.FCOS.NUM_CLASSES = 2
        # cfg.MODEL.FCOS.USE_SCALE = True
        # cfg.MODEL.FCOS.SIZES_OF_INTEREST = [
        #     32,
        #     64,
        #     128,
        # ]  # size of interest for each level ,it is used to filter out the small objects,
        # #cfg.MODEL.FCOS.NORM = 'GN'

        # cfg.MODEL.MASKIOU_ON = True
        # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        # cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3"]
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        # cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
        # cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
        # cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
        # cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
        # cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
    
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        # 保存配置文件
        with open("Netlib/CenterMasklib/centermask/configs_yaml/conic_centermask50.yaml", "w") as f:
            f.write(cfg.dump())
        cfg.freeze()
        return cfg

class Cenetermaskres101(centermaskres50):
    batch = 8
    lr = 2e-3

    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("CenterMasklib/centermask/configs_yaml/centermask_R_101_FPN_ms_3x.yaml")
        # 模型settings：
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0

        cfg.MODEL.RESNETS.DEPTH = 101
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4"]

        cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4"]
        cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16]
        cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 2000
        cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 1500
        cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 2500
        cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 2000
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
        cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
        cfg.MODEL.FCOS.NMS_TH = 0.3
        cfg.MODEL.FCOS.NUM_CLASSES = 7
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [16, 32, 128]
        cfg.MODEL.FCOS.TOP_LEVELS = 0
        cfg.MODEL.FCOS.POS_RADIUS = 1.0

        cfg.MODEL.MASKIOU_ON = False
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        with open("CenterMasklib/centermask/configs_yaml/conic_centermask101.yaml", "w") as f:
            f.write(cfg.dump())
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
        cfg.freeze()
        return cfg

class CenterMask101Pannuke(centermaskres50):
    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("CenterMasklib/centermask/configs_yaml/centermask_R_50_FPN_ms_3x.yaml")
        # 模型settings：
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0

        cfg.MODEL.RESNETS.DEPTH = 101
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4"]

        cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4"]
        cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16]
        cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 2000
        cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 1500
        cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 2500
        cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 2500

        cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.2
        cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.2
        cfg.MODEL.FCOS.NMS_TH = 0.4
        cfg.MODEL.FCOS.NUM_CLASSES = 5
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [16, 32, 128]
        cfg.MODEL.FCOS.TOP_LEVELS = 0
        cfg.MODEL.FCOS.POS_RADIUS = (
            1.0  # it is used to enlarge or shrink the positive samples reg
        )
        cfg.MODEL.MASKIOU_ON = False
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        with open("CenterMasklib/centermask/configs_yaml/pannuke_centermask101.yaml", "w") as f:
            f.write(cfg.dump())
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
        cfg.freeze()
        return cfg

class Infer50_C:
    model_para = "model_para/centermaskres50.pt"

    def __init__(self) -> None:
        cfg = defaults._C.clone()
        cfg.merge_from_file("CenterMasklib/centermask/configs_yaml/conic_centermask50.yaml")
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
        cfg.freeze()
        self.cfg = cfg
        self.model = GeneralizedRCNN

class Infer101_C:
    model_para = "model_para/Cenetermaskres101.pt"

    def __init__(self) -> None:
        cfg = defaults._C.clone()
        cfg.merge_from_file("CenterMasklib/centermask/configs_yaml/conic_centermask101.yaml")
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 2
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 16
        cfg.freeze()
        self.cfg = cfg
        self.model = GeneralizedRCNN


class CenterMask50MoNuSeg_as_the_same_settings(centermaskres50):
    epoch = 1000
    lr = 2e-4
    lr_s = [300, 500, 700, 900]
    gamma = 0.5
    NUM_WORKERS = 8
    batch = 4
    def __init__(self, ds, device, ds_num) -> None:
        super().__init__(ds, device, ds_num)
        data = MoNuSeg(ds_num)
        self.CLASSES = data.classes
        self.DATASET_DIR = data.dataset_dir
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.load_model_para = f"{self.model_para}/best.pth"
        self.cfg = self.setup(device)
    
    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("Netlib/CenterMasklib/centermask/configs_yaml/centermask_R_50_FPN_ms_3x.yaml")
        # 模型settings：
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 4
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 224
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        # 保存配置文件
        with open("Netlib/CenterMasklib/centermask/configs_yaml/MoNuSeg_centermask50.yaml", "w") as f:
            f.write(cfg.dump())
        cfg.freeze()
        return cfg

class CenterMask50MoNuSeg_3FPN(centermaskres50):
    epoch = 1000
    lr = 2e-4
    lr_s = [300, 500, 700, 900]
    gamma = 0.5
    NUM_WORKERS = 8
    batch = 4
    def __init__(self, ds, device, ds_num) -> None:
        super().__init__(ds, device, ds_num)
        data = MoNuSeg(ds_num)
        self.CLASSES = data.classes
        self.DATASET_DIR = data.dataset_dir
        self.CLASSES = data.classes
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.load_model_para = f"{self.model_para}/best.pth"
        self.cfg = self.setup(device)
    
    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("Netlib/CenterMasklib/centermask/configs_yaml/centermask_R_50_FPN_ms_3x.yaml")
        # 模型settings：
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        #cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 1500
        #cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 800
        #cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 6000
        #cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4"]
        cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16]
        cfg.MODEL.FCOS.CENTER_SAMPLE = True
        cfg.MODEL.FCOS.NUM_CLASSES = 2
        cfg.MODEL.FCOS.NMS_TH = 0.18
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [
            32,
            64,
            128,
        ]

        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3","p4"]
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 4
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 224

        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        # 保存配置文件
        with open("Netlib/CenterMasklib/centermask/configs_yaml/MoNuSeg_centermask50.yaml", "w") as f:
            f.write(cfg.dump())
        cfg.freeze()
        return cfg

class CenterMask50MoNuSeg_3FPN2(centermaskres50):
    epoch = 1000
    lr = 2e-4
    lr_s = [300, 500, 700, 900]
    gamma = 0.5
    NUM_WORKERS = 8
    batch = 4
    def __init__(self, ds, device, ds_num) -> None:
        super().__init__(ds, device, ds_num)
        self.cfg = self.setup(device)
        data = MoNuSeg(ds_num)
        self.CLASSES = data.classes
        self.DATASET_DIR = data.dataset_dir
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.load_model_para = f"{self.model_para}/best.pth"


    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("Netlib/CenterMasklib/centermask/configs_yaml/centermask_R_50_FPN_ms_3x.yaml")
        # 模型settings：
        #cfg.MODEL.RESNETS.NORM = "GN"
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        #cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 1500
        #cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 800
        #cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 6000
        #cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4"]
        cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16]
        cfg.MODEL.FCOS.CENTER_SAMPLE = True
        cfg.MODEL.FCOS.NUM_CLASSES = 2
        cfg.MODEL.FCOS.NMS_TH = 0.18
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [
            32,
            64,
            128,
        ]

        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3","p4"]
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 4
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 224
        #cfg.MODEL.ROI_MASK_HEAD.NORM = 'GN'
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        # 保存配置文件
        with open("Netlib/CenterMasklib/centermask/configs_yaml/MoNuSeg_centermask50.yaml", "w") as f:
            f.write(cfg.dump())
        cfg.freeze()
        return cfg

class CenterMask50MoNuSeg_3FPN3_GN(centermaskres50):
    epoch = 1000
    lr = 2e-4
    lr_s = [300, 500, 700, 900]
    gamma = 0.5
    NUM_WORKERS = 8
    batch = 4
    def __init__(self, ds, device, ds_num) -> None:
        super().__init__(ds, device, ds_num)
        self.cfg = self.setup(device)
        data = MoNuSeg(ds_num)
        self.CLASSES = data.classes
        self.DATASET_DIR = data.dataset_dir
        self.TRAIN_SET_INDEX = data['train']
        self.VAL_SET_INDEX = data['val']
        self.TEST_SET_INDEX = data['test']
        self.load_model_para = f"{self.model_para}/best.pth"


    def setup(self, device):
        cfg = defaults._C.clone()
        cfg.merge_from_file("Netlib/CenterMasklib/centermask/configs_yaml/centermask_R_50_FPN_ms_3x.yaml")
        # 模型settings：
        cfg.MODEL.RESNETS.NORM = "GN"
        cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
        cfg.MODEL.PIXEL_STD = [1, 1, 1]
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        #cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 1500
        #cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 800
        #cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 6000
        #cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3", "p4"]
        cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16]
        cfg.MODEL.FCOS.CENTER_SAMPLE = True
        cfg.MODEL.FCOS.NUM_CLASSES = 2
        cfg.MODEL.FCOS.NMS_TH = 0.18
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [
            32,
            64,
            128,
        ]

        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3","p4"]
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 4
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 224
        cfg.MODEL.ROI_MASK_HEAD.NORM = 'GN'
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.OUTPUT_DIR = ""
        cfg.freeze()
        return cfg
    




class Infer101_P:
    model_para = "model_para/CenterMask101Pannuke.pt"

    def __init__(self) -> None:
        cfg = defaults._C.clone()
        cfg.merge_from_file("centermask/configs_yaml/pannuke_centermask101.yaml")
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_LEVEL = 4
        cfg.MODEL.ROI_MASK_HEAD.CANONICAL_SIZE = 224
        cfg.freeze()
        self.cfg = cfg
        self.model = GeneralizedRCNN
