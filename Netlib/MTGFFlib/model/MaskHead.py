import torch.nn as nn
from .MaskRCNN_utils import bn_act_conv
from torchvision.models.resnet import Bottleneck

class MaskHead(nn.Module):
    def __init__(self, in_c=256) -> None:
        super().__init__()
        self.conv = bn_act_conv(in_c, in_c // 2, 1, 1)  # 128
        self.blocks = nn.Sequential(
            Bottleneck(
                in_c // 2,
                in_c // 16,
                downsample=nn.Conv2d(in_c // 2, in_c // 4, 1, 1),
            ),  # 64
            Bottleneck(
                in_c // 4,
                in_c // 32,
                downsample=nn.Conv2d(in_c // 4, in_c // 8, 1, 1),
            ),  # 32
        )
        self.out = nn.ConvTranspose2d(in_channels=in_c // 8, out_channels= 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        return self.out(x)