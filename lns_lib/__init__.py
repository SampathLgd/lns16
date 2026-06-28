"""
LNS (Logarithmic Number System) Library
Implements LNS16 - 16-bit Logarithmic Number System for DNN operations.

Format: 16-bit LNS
  - 1 bit sign
  - 6 bits integer part of log
  - 9 bits fractional part of log
  Base: 2
"""

from .lns16 import LNS16, lns_logadd, lns_logmul, lns_dot_product
from .dnn_ops import LNSLinear, LNSConv2d, LNSBatchNorm2d, LNSReLU, lns_softmax

__all__ = [
    "LNS16",
    "lns_logadd",
    "lns_logmul",
    "lns_dot_product",
    "LNSLinear",
    "LNSConv2d",
    "LNSBatchNorm2d",
    "LNSReLU",
    "lns_softmax",
]
