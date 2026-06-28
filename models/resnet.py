"""
ResNet-20 for CIFAR-10
======================
Three variants:
  - FP32ResNet   : standard float32 (baseline)
  - FP16ResNet   : float16 (PyTorch autocast / half-precision)
  - LNSResNet    : LNS16 quantised convolutions + BN

ResNet-20 architecture following He et al. 2016 (CIFAR-10 variant).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lns_lib import LNSConv2d, LNSBatchNorm2d, LNSReLU


# ─── Residual Block ───────────────────────────────────────────────────────────

def _make_block(ConvCls, BNCls, ActCls, in_ch, out_ch, stride=1):
    """Build one BasicBlock with the given layer factories."""
    layers = nn.Sequential(
        ConvCls(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        BNCls(out_ch),
        ActCls(),
        ConvCls(out_ch, out_ch, 3, padding=1, bias=False),
        BNCls(out_ch),
    )
    shortcut = nn.Sequential()
    if stride != 1 or in_ch != out_ch:
        shortcut = nn.Sequential(
            ConvCls(in_ch, out_ch, 1, stride=stride, bias=False),
            BNCls(out_ch),
        )
    return layers, shortcut


class BasicBlock(nn.Module):
    def __init__(self, ConvCls, BNCls, ActCls, in_ch, out_ch, stride=1):
        super().__init__()
        self.layers, self.shortcut = _make_block(
            ConvCls, BNCls, ActCls, in_ch, out_ch, stride)
        self.act = ActCls()

    def forward(self, x):
        return self.act(self.layers(x) + self.shortcut(x))


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10 (32×32 input, 10 classes)."""

    def __init__(self, ConvCls=nn.Conv2d, BNCls=nn.BatchNorm2d, ActCls=nn.ReLU):
        super().__init__()
        cfg = [(16, 1), (32, 2), (64, 2)]   # (channels, stride_at_first_block)

        self.stem = nn.Sequential(
            ConvCls(3, 16, 3, padding=1, bias=False),
            BNCls(16),
            ActCls(),
        )
        layers = []
        in_ch = 16
        for out_ch, stride in cfg:
            # 3 blocks per stage (ResNet-20)
            layers.append(BasicBlock(ConvCls, BNCls, ActCls, in_ch, out_ch, stride))
            for _ in range(2):
                layers.append(BasicBlock(ConvCls, BNCls, ActCls, out_ch, out_ch, 1))
            in_ch = out_ch
        self.body   = nn.Sequential(*layers)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─── Variant factories ────────────────────────────────────────────────────────

def fp32_resnet20() -> ResNet20:
    """Standard FP32 ResNet-20."""
    return ResNet20(nn.Conv2d, nn.BatchNorm2d, nn.ReLU)


def fp16_resnet20() -> ResNet20:
    """FP16 ResNet-20 – model parameters stored in float16."""
    return ResNet20(nn.Conv2d, nn.BatchNorm2d, nn.ReLU).half()


def lns_resnet20() -> ResNet20:
    """LNS16 ResNet-20 – quantised conv + BN during forward."""
    return ResNet20(LNSConv2d, LNSBatchNorm2d, LNSReLU)
