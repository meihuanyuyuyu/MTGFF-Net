from .config import get_cfg
from .centermask_config import *
from .data_info import *
import numpy as np
import torch
import random


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

__all__ = [
    "get_cfg",
]
