"""
LNS16 DNN Operations
====================
PyTorch nn.Module subclasses that emulate LNS16 arithmetic:
  - LNSLinear   – fully-connected layer
  - LNSConv2d   – 2-D convolution layer
  - LNSBatchNorm2d – batch normalisation
  - LNSReLU     – ReLU activation
  - lns_softmax – softmax function

All layers quantise weights & activations through LNS16 before computation,
then use float32 PyTorch ops for the actual arithmetic (simulation approach).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lns16 import LNS16


def _q(t: torch.Tensor) -> torch.Tensor:
    """Quantise tensor through LNS16 round-trip."""
    return LNS16.quantise_tensor(t)


# ─── Linear Layer ─────────────────────────────────────────────────────────────

class LNSLinear(nn.Linear):
    """
    Fully-connected layer with LNS16 quantised weights and activations.
    Inherits trainable parameters from nn.Linear; quantisation is applied
    only during the forward pass (quantisation-aware inference simulation).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = _q(x)
        qw = _q(self.weight)
        qb = _q(self.bias) if self.bias is not None else None
        return F.linear(qx, qw, qb)


# ─── Conv2d Layer ─────────────────────────────────────────────────────────────

class LNSConv2d(nn.Conv2d):
    """
    2-D convolution with LNS16 quantised weights and activations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = _q(x)
        qw = _q(self.weight)
        qb = _q(self.bias) if self.bias is not None else None
        return F.conv2d(qx, qw, qb,
                        self.stride, self.padding,
                        self.dilation, self.groups)


# ─── BatchNorm2d ──────────────────────────────────────────────────────────────

class LNSBatchNorm2d(nn.BatchNorm2d):
    """
    Batch normalisation with LNS16 quantised parameters.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BN is applied in float; quantise output
        out = super().forward(x)
        return _q(out)


# ─── ReLU ─────────────────────────────────────────────────────────────────────

class LNSReLU(nn.Module):
    """
    ReLU in LNS domain: simply clips to 0 (in real representation negative
    numbers map to specific bit-patterns; clamping is exact).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _q(F.relu(x))


# ─── Softmax ──────────────────────────────────────────────────────────────────

def lns_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with LNS16 quantised input."""
    qx = _q(x)
    return F.softmax(qx, dim=dim)
