from .CDNet_config import *
from .data_info import *
import numpy as np
import torch
import random


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
        assert ds == "conic" or ds == "pannuke"
        if ds == "conic":
            self.anchorwh = self._conic_wh
            if num_anchors != 15:
                self.anchorwh = torch.load("tools/anchors_wh.pt")[num_anchors]
                # self.anchorwh= torch.load('tools/anchors_wh.pt')[num_anchors]
        if ds == "pannuke":
            self.anchorwh = self._pannuke_wh
            if num_anchors != 15:
                self.anchorwh = torch.load("tools/anchors_wh_pannuke.pt")[num_anchors]


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


# set your infering configuration here:
class Infer_config:
    numpy_result_dir = "Infering_results"

    def __init__(self, config, device, ds) -> None:
        # 将模型配置代理到预测配置。
        for att in dir(config):
            if not att.startswith("_"):
                setattr(self, att, getattr(config, att))
        self.ds = ds
        self.device = device
        if hasattr(config, "num_anchors"):
            self.anchor_wh = AnchorWH(ds, config.num_anchors).anchorwh
