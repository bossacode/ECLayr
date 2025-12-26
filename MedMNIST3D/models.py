import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from eclayr.cubical.cubeclayr import CubECLayr, CubDECC


# 28 x 28
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(nn.ReLU())
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        # out = F.avg_pool2d(out, 4)
        # out = F.adaptive_avg_pool3d(out, output_size=4)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet18(ResNet):
    def __init__(self, in_channels, num_classes, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super().__init__(block, num_blocks, in_channels, num_classes)


# function for making postprocess layer
def make_postprocess(in_dim, dim_cfg:list):
    num_layers = len(dim_cfg)
    if num_layers == 1:
        postprocess = nn.Linear(in_dim, dim_cfg[0])
    elif num_layers > 1:
        postprocess = nn.Sequential(nn.Linear(in_dim, dim_cfg[0]))
        for i in range(1, num_layers):
            postprocess.append(nn.ReLU())
            postprocess.append(nn.Linear(dim_cfg[i-1], dim_cfg[i]))
    else:
        raise NotImplementedError("'dim_cfg' should contain at least one element.")
    return postprocess


class ECResNet18_i(ResNet18):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.eclayr = CubECLayr(postprocess=make_postprocess(kwargs["steps"], kwargs["postprocess_cfg"]), **kwargs)
        self.linear = nn.Linear(512 + kwargs["postprocess_cfg"][-1], num_classes)

    def forward(self, x):
        x_1 = F.relu(self.eclayr(x[:,[0]])) # ECLayr
        x = F.relu(self.bn1(self.conv1(x))) # ResNet
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.concat((x, x_1), dim=-1)
        x = self.linear(x)
        return x


class ECResNet18(ResNet18):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.eclayr_1 = CubECLayr(interval=kwargs["interval_1"], sublevel=kwargs["sublevel_1"],
                               postprocess=make_postprocess(kwargs["steps"], kwargs["postprocess_cfg"]), **kwargs)
        self.eclayr_2 = CubECLayr(interval=kwargs["interval_2"], sublevel=kwargs["sublevel_2"],
                               postprocess=make_postprocess(kwargs["steps"], kwargs["postprocess_cfg"]), **kwargs)
        self.linear = nn.Linear(512 + 2*kwargs["postprocess_cfg"][-1], num_classes)

    def forward(self, x):
        # first ECLayr
        x_1 = F.relu(self.eclayr_1(x[:,[0]]))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)

        # second ECLayr after res layer
        x_2 = x.mean(dim=1, keepdim=True)
        max_vals = x_2.amax(dim=(2, 3, 4), keepdim=True)    # shape: [B, C, 1, 1, 1]
        min_vals = x_2.amin(dim=(2, 3, 4), keepdim=True)    # shape: [B, C, 1, 1, 1]
        x_2 = (x_2 - min_vals) / (max_vals - min_vals)      # normalize x_2 between 0 and 1
        x_2 = F.relu(self.eclayr_2(x_2))

        x = F.relu(self.layer2(F.relu(x)))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.linear(x)
        return x
    

class DECResNet18(ResNet18):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.eclayr = CubECLayr(interval=kwargs["interval_1"], sublevel=kwargs["sublevel_1"],
                                  postprocess=make_postprocess(kwargs["steps"], kwargs["postprocess_cfg"]), **kwargs)
        self.decc = CubDECC(interval=kwargs["interval_2"], sublevel=kwargs["sublevel_2"],
                                 postprocess=make_postprocess(kwargs["steps"], kwargs["postprocess_cfg"]), **kwargs)
        self.linear = nn.Linear(512 + 2*kwargs["postprocess_cfg"][-1], num_classes)

    def forward(self, x):
        # first ECLayr
        x_1 = F.relu(self.eclayr(x[:,[0]]))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)

        # second DECC after res layer
        x_2 = x.mean(dim=1, keepdim=True)
        max_vals = x_2.amax(dim=(2, 3, 4), keepdim=True)    # shape: [B, C, 1, 1, 1]
        min_vals = x_2.amin(dim=(2, 3, 4), keepdim=True)    # shape: [B, C, 1, 1, 1]
        x_2 = (x_2 - min_vals) / (max_vals - min_vals)      # normalize x_2 between 0 and 1
        x_2 = F.relu(self.decc(x_2))

        x = F.relu(self.layer2(F.relu(x)))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.linear(x)
        return x