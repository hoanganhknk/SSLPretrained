# networks/resnet_cifar32.py

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet32_cifar']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetCIFAR(nn.Module):
    """
    CIFAR ResNet: conv3x3, không maxpool, 3 stage với [n1, n2, n3] block.
    Depth = 6n + 2 -> ResNet32: n = 5 -> [5,5,5].
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        # CIFAR: 32x32, conv3x3 stride=1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = self.avgpool(out)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        if return_feat:
            return logits, feat
        else:
            return logits


def resnet32_cifar(num_classes=10):
    # 6n+2 với n=5 -> 32
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes)
