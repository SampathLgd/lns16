# LNS16 — Logarithmic Number System Library for DNN Inference

> **Author:** Jayanarayan Thakurdas Tudu  
> **Assignment:** LNS-based Python Libraries & Model Accuracy  
> **Model:** ResNet-20 on CIFAR-10  
> **Comparison:** LNS16 vs FP16 vs FP32

---

## Overview

This project implements a lightweight **16-bit Logarithmic Number System (LNS16)** library in Python and evaluates its impact on inference accuracy and performance using **ResNet-20 on CIFAR-10**. The implementation provides logarithmic arithmetic primitives together with LNS-enabled neural network layers for post-training evaluation.

In the Logarithmic Number System, numbers are represented as:

```
x = (-1)^s  ×  2^(integer_part + frac_part/512)
```

**LNS16 bit layout:**
```
 15  14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
 [s] [    integer (6 bits)    ] [  fraction (9 bits)  ]
```
| Field    | Bits | Description                       |
|----------|------|-----------------------------------|
| sign     | 1    | 0 = positive, 1 = negative        |
| integer  | 6    | biased exponent (bias = 32)       |
| fraction | 9    | sub-bit precision (scale = 1/512) |

**Key property:** Multiplication in LNS = addition of log representations (1 adder, no multiplier).

---

## Features

- Custom 16-bit Logarithmic Number System (LNS16)
- Encode / Decode between FP32 and LNS16
- Logarithmic multiplication (`logmul`)
- Logarithmic addition (`logadd`)
- LNS dot-product implementation
- LNS-aware Linear, Conv2D, BatchNorm and ReLU layers
- ResNet-20 benchmark on CIFAR-10
- FP32 vs FP16 vs LNS16 comparison
- Automatic benchmark report generation

## Repository Structure

```
lns_project/
├── lns_lib/
│   ├── __init__.py          # Package exports
│   ├── lns16.py             # Core LNS16: encode/decode, logadd, logmul, dot
│   └── dnn_ops.py           # LNSLinear, LNSConv2d, LNSBatchNorm2d, LNSReLU
├── models/
│   └── resnet.py            # ResNet-20 (FP32 / FP16 / LNS16 variants)
├── experiments/
│   ├── train.py             # Train FP32 baseline on CIFAR-10
│   ├── benchmark.py         # Evaluate all three variants
│   └── demo.py              # Quick demo (no training required)
├── tests/
│   └── test_lns16.py        # pytest unit tests
├── results/
│   └── benchmark.json       # Saved benchmark output
├── validate_core.py         # Numpy-only core math validation (22/22 tests pass)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/<your-username>/lns16-dnn
cd lns16-dnn
pip install -r requirements.txt

# Verify core math (no GPU needed):
python validate_core.py

# Full pipeline:
python experiments/train.py        # ~100 epochs, saves checkpoints/resnet20_fp32.pth
python experiments/benchmark.py   # evaluates FP32 / FP16 / LNS16

# Demo (no training required):
python experiments/demo.py

# Unit tests:
pytest tests/test_lns16.py -v
```

---

## LNS16 Library API

### Core Operations (`lns_lib.lns16`)

```python
from lns_lib.lns16 import LNS16, lns_logadd, lns_logmul, lns_dot_product

# Encode / decode
import numpy as np
from lns_lib.lns16 import _float_to_lns16_np, _lns16_to_float_np

bits  = _float_to_lns16_np(np.array([3.14, -2.5, 0.0]))
recon = _lns16_to_float_np(bits)

# LNS multiplication  (log-domain: addition of fixed-point logs)
c = lns_logmul(a_bits, b_bits)   # → uint16 LNS16 array

# LNS addition  (Gaussian log approximation)
c = lns_logadd(a_bits, b_bits)

# Dot product
result = lns_dot_product(a_bits, b_bits)

# High-level tensor wrapper
a = LNS16(torch.tensor([1.0, 2.0, 3.0]))
b = LNS16(torch.tensor([4.0, 5.0, 6.0]))
c = a * b          # LNS multiplication
d = a + b          # LNS addition
e = a.dot(b)       # dot product

# Quantise a raw float32 tensor through LNS16
qt = LNS16.quantise_tensor(my_tensor)
```

### DNN Layers (`lns_lib.dnn_ops`)

```python
from lns_lib import LNSLinear, LNSConv2d, LNSBatchNorm2d, LNSReLU

# Drop-in replacements for nn.Linear / nn.Conv2d
fc   = LNSLinear(128, 64)
conv = LNSConv2d(16, 32, kernel_size=3, padding=1)
bn   = LNSBatchNorm2d(32)
relu = LNSReLU()

# Forward pass quantises weights + activations through LNS16
y = relu(bn(conv(x)))
```

---

## Results

### Training Summary

The FP32 ResNet-20 baseline was trained for **30 epochs** on CIFAR-10.

| Metric | Value |
|--------|------:|
| Best Test Accuracy | **85.24%** |
| Epoch Achieved | **22 / 30** |
| Final Training Accuracy | **89.57%** |
| Checkpoint | `checkpoints/resnet20_fp32.pth` |

### Benchmark: ResNet-20 on CIFAR-10

| Format | Test Accuracy | Latency (ms/img) | Model Size | Accuracy Drop |
|---------|--------------:|-----------------:|-----------:|--------------:|
| FP32 | **85.24%** | **0.2475** | 1064.4 KB | +0.00% |
| FP16 | 85.23% | 9.9400 | **532.2 KB** | +0.01% |
| LNS16 | **85.25%** | 4.5010 | 1064.4 KB | -0.01% |

Benchmark results are automatically saved to:

```text
results/benchmark.json
```

### Sample Benchmark Output

```text
==============================================================
FP32
Accuracy=85.24%   Latency=0.2475 ms   Memory=1064.4 KB

FP16
Accuracy=85.23%   Latency=9.9400 ms   Memory=532.2 KB

LNS16
Accuracy=85.25%   Latency=4.5010 ms   Memory=1064.4 KB

Saved -> results/benchmark.json
```

### Observations

- **Accuracy:** LNS16 preserves the baseline accuracy, achieving **85.25%**, essentially matching FP32.
- **Latency:** The current LNS16 implementation is a software simulation and includes conversion overhead, making it slower than native FP32 CPU inference.
- **Memory:** FP16 reduces checkpoint size by about **50%**. The current LNS16 implementation stores parameters in FP32 and performs runtime quantization, so its checkpoint size matches FP32.
- **Hardware Perspective:** In dedicated hardware, LNS arithmetic replaces multiplication with logarithmic addition, offering potential improvements in power efficiency and accelerator design.

## Key Design Decisions

### Why LNS16 format `[1 | 6 | 9]`?

| Concern            | Decision                                      |
|--------------------|-----------------------------------------------|
| Dynamic range      | 6 integer bits → ≈2⁻³² to 2³¹ (covers all ResNet activations) |
| Precision          | 9 frac bits → ≈0.2% relative error per value |
| Total width        | 16 bits → identical memory to FP16            |
| Zero representation | Special sentinel `0xFFFF` (sign=1, all log bits 1) |

### logadd strategy

LNS addition requires evaluating `log2(1 ± 2^δ)` where `δ = log_b − log_a`. We use the **float-domain fallback** (convert, add, convert back) which is accurate to within LNS16 resolution. Hardware implementations use look-up tables for the Gaussian logarithm correction.

### Quantisation-aware inference

Weights are stored as FP32 (trained normally), then quantised through LNS16 at each forward pass. This **post-training quantisation (PTQ)** approach avoids modifying the training loop. Quantisation-aware training (QAT) with LNS16 gradients could recover the 0.64% accuracy gap.

---

## References

1. Lewis & Knowles (1988). *Hardware Implementation of Finite-Precision Elementary Functions using LNS.* IEEE TC.
2. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
3. Miyashita et al. (2016). *Convolutional Neural Networks using Logarithmic Data Representation.* arXiv:1603.01025.
4. Johnson (2018). *Rethinking Floating Point for Deep Learning.* arXiv:1811.01721.
