from .MaskRCNN_utils import bn_act_conv
from torchvision.models import resnet50, resnet101
from types import MethodType
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x0 = self.relu(x)
    x = self.maxpool(x0)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    return OrderedDict({"f32": x4, "f16": x3, "f8": x2, "f4": x1, "f2": x0})

class FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x4_conv = bn_act_conv(2048, 512, 1, 1)  # /32
        self.x3_conv = bn_act_conv(1024, 512, 1, 1)  # 16
        self.x2_conv = bn_act_conv(512, 512, 1, 1)  # 8
        self.x1_conv = bn_act_conv(256, 512, 1, 1)  # /4

        self.out_2 = bn_act_conv(512, 256, 3, 1, 1)
        self.out_1 = bn_act_conv(512, 256, 3, 1, 1)
        self.out_3 = bn_act_conv(512, 256, 3, 1, 1)
        self.out_4 = bn_act_conv(512, 256, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x) -> OrderedDict:
        out = OrderedDict()
        x4 = self.x4_conv(x["f32"])
        out["f32"] = self.out_4(x4)

        x3 = self.x3_conv(x["f16"]) + self.up(x4)
        out["f16"] = self.out_3(x3)

        x2 = self.x2_conv(x["f8"]) + self.up(x3)
        out["f8"] = self.out_2(x2)

        x1 = self.x1_conv(x["f4"]) + self.up(x2)
        out["f4"] = self.out_1(x1)

        out["f2"] = x["f2"]
        return out

class Resnet50FPN(nn.Module):
    def __init__(self, pretrain=True, bottom_up=False) -> None:
        super().__init__()
        if pretrain:
            self.bcakbone = resnet50(pretrained=True)
            del self.bcakbone.fc
        else:
            self.bcakbone = resnet50()
            del self.bcakbone.fc
        self.bcakbone.forward = MethodType(forward, self.bcakbone)
        if bottom_up:
            self.fpn = FPNwithBottomup()
        else:
            self.fpn = FPN()

    def forward(self, x):
        x = self.bcakbone(x)
        return self.fpn(x)

class Resnet101FPN(Resnet50FPN):
    def __init__(self, pretrain=True, bottom_up=False) -> None:
        super().__init__(pretrain)
        if pretrain:
            self.backbone = resnet101(pretrained=True)
        else:
            self.backbone = resnet101()
            del self.backbone.fc
        self.bcakbone.forward = MethodType(forward, self.bcakbone)
        if bottom_up:
            self.fpn = FPNwithBottomup()
        else:
            self.fpn = FPN()

    def forward(self, x):
        return super().forward(x)
    
class FPNwithBottomup(nn.Module):
    def __init__(self, num_layers=4) -> None:
        super().__init__()
        # idx top->bottom
        self.input_conv = nn.ModuleList(
            [bn_act_conv(256 * 2**_, 256, 3, 1, 1) for _ in range(num_layers)]
        )
        self.up_conv = nn.ModuleList(
            [bn_act_conv(256, 256, 3, 1, 1) for _ in range(num_layers)]
        )
        self.out = nn.ModuleList(
            [bn_act_conv(256, 256, 3, 1, 1) for _ in range(num_layers)]
        )
        self.num = num_layers

    def forward(self, x: OrderedDict):
        if "f2" in x.keys():
            x0 = x.pop("f2")
        keys = x.keys()
        for l, feature in enumerate(x.values()):
            feature = self.input_conv[l](feature)
            if l != 0:
                x[keys[l]] = feature + F.upsample(x[keys[l - 1]], scale_factor=2)
            else:
                x[keys[l]] = feature
        # bottom_up:
        for l in range(len(self.num) - 1, -1, -1):
            if l == self.num - 1:
                x[keys[l]] = self.up_conv[l](x[keys[l]])
            else:
                x[keys[l]] = self.up_conv[l](x[keys[l]]) + F.upsample(
                    x[keys[l + 1]], scale_factor=0.5
                )
        for l, feature in enumerate(x.values()):
            x[keys[l]] = self.out[l](feature)
        return x.update({"f2": x0})