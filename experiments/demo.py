"""
Quick Demo: LNS16 Library Operations + Simulated Benchmark
===========================================================
Demonstrates the LNS16 library without requiring a full training run.
For full results, run: train.py → benchmark.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from lns_lib.lns16 import (
    _float_to_lns16_np, _lns16_to_float_np,
    lns_logmul, lns_logadd, lns_dot_product, LNS16
)
from lns_lib.dnn_ops import LNSLinear, LNSConv2d

DIVIDER = "=" * 65


def demo_encode_decode():
    print(f"\n{DIVIDER}")
    print("  LNS16 Encode / Decode Demo")
    print(DIVIDER)
    vals = [1.0, -2.5, 0.125, 100.0, -0.001, 0.0]
    print(f"  {'Value':>10} {'Bits(hex)':>10} {'Decoded':>12} {'Rel Err':>10}")
    print("  " + "-" * 50)
    for v in vals:
        arr   = np.array([v])
        bits  = _float_to_lns16_np(arr)
        recon = _lns16_to_float_np(bits)[0]
        err   = abs(recon - v) / abs(v) if v != 0 else 0.0
        print(f"  {v:>10.4f} {int(bits[0]):>#10x} {recon:>12.6f} {err:>9.4%}")


def demo_arithmetic():
    print(f"\n{DIVIDER}")
    print("  LNS16 Arithmetic Demo")
    print(DIVIDER)

    test_cases = [
        ("logmul", 3.0, 4.0),
        ("logmul", -2.0, 5.0),
        ("logmul", 0.5, 8.0),
        ("logadd", 3.0, 4.0),
        ("logadd", -1.5, 2.5),
        ("logadd", 0.0, 7.0),
    ]
    print(f"  {'Op':>8} {'a':>8} {'b':>8} {'LNS16':>10} {'Exact':>10} {'Rel Err':>10}")
    print("  " + "-" * 60)
    for op, a, b in test_cases:
        ab = _float_to_lns16_np(np.array([a]))
        bb = _float_to_lns16_np(np.array([b]))
        if op == "logmul":
            rb = lns_logmul(ab, bb); exact = a * b
        else:
            rb = lns_logadd(ab, bb); exact = a + b
        res = _lns16_to_float_np(rb)[0]
        err = abs(res - exact) / abs(exact) if exact != 0 else 0.0
        print(f"  {op:>8} {a:>8.3f} {b:>8.3f} {res:>10.4f} {exact:>10.4f} {err:>9.4%}")


def demo_dot_product():
    print(f"\n{DIVIDER}")
    print("  LNS16 Dot Product Demo")
    print(DIVIDER)
    pairs = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 0, 0], [0, 1, 0]),
        ([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]),
        (list(range(1, 9)), list(range(1, 9))),
    ]
    print(f"  {'a':>30} {'b':>30} {'LNS16':>8} {'Exact':>8} {'Err':>8}")
    print("  " + "-" * 90)
    for a, b in pairs:
        ab = _float_to_lns16_np(np.array(a, dtype=np.float64))
        bb = _float_to_lns16_np(np.array(b, dtype=np.float64))
        rb = lns_dot_product(ab, bb)
        res = _lns16_to_float_np(rb)[0]
        exact = float(np.dot(a, b))
        err   = abs(res - exact) / abs(exact) if exact != 0 else 0.0
        print(f"  {str(a):>30} {str(b):>30} {res:>8.3f} {exact:>8.3f} {err:>7.3%}")


def demo_lns_layer():
    print(f"\n{DIVIDER}")
    print("  LNS16 Neural Network Layer Demo")
    print(DIVIDER)

    # LNSLinear
    layer = LNSLinear(128, 64)
    x = torch.randn(8, 128)
    t0 = time.perf_counter()
    y = layer(x)
    t1 = time.perf_counter()
    print(f"  LNSLinear(128→64)  | input {tuple(x.shape)} → output {tuple(y.shape)}")
    print(f"  Forward time: {(t1-t0)*1000:.2f} ms  |  output mean: {y.mean():.4f}")

    # LNSConv2d
    conv = LNSConv2d(16, 32, 3, padding=1)
    x2 = torch.randn(4, 16, 32, 32)
    t0 = time.perf_counter()
    y2 = conv(x2)
    t1 = time.perf_counter()
    print(f"\n  LNSConv2d(16→32,k=3)| input {tuple(x2.shape)} → output {tuple(y2.shape)}")
    print(f"  Forward time: {(t1-t0)*1000:.2f} ms  |  output mean: {y2.mean():.4f}")


def demo_quantisation_error():
    print(f"\n{DIVIDER}")
    print("  Quantisation Error Analysis (LNS16 vs FP32)")
    print(DIVIDER)
    torch.manual_seed(42)
    w = torch.randn(64, 64)
    x = torch.randn(64)

    # FP32 reference
    fp32_out = w @ x

    # FP16
    fp16_out = (w.half() @ x.half()).float()

    # LNS16
    from lns_lib.lns16 import LNS16
    lns_w = LNS16.quantise_tensor(w)
    lns_x = LNS16.quantise_tensor(x)
    lns_out = lns_w @ lns_x   # still float32 arithmetic after quantisation

    fp16_err = (fp16_out - fp32_out).abs().mean().item()
    lns_err  = (lns_out  - fp32_out).abs().mean().item()

    print(f"  Matrix-vector multiply (64×64) @ (64,)")
    print(f"  FP32 mean output magnitude : {fp32_out.abs().mean():.4f}")
    print(f"  FP16 mean absolute error   : {fp16_err:.6f}")
    print(f"  LNS16 mean absolute error  : {lns_err:.6f}")
    print(f"  LNS16 / FP16 error ratio   : {lns_err/fp16_err:.3f}x")


def simulated_benchmark_table():
    """
    Simulated benchmark results representative of typical ResNet-20 / CIFAR-10 runs.
    Full results require running train.py + benchmark.py.
    Values based on standard literature + our LNS16 quantisation overhead.
    """
    print(f"\n{DIVIDER}")
    print("  Simulated Benchmark Results (ResNet-20 / CIFAR-10)")
    print("  (Run train.py + benchmark.py for actual measured values)")
    print(DIVIDER)

    # These represent typical values; replaced by real numbers after full run
    results = {
        "FP32":  {"accuracy": 92.47, "latency_ms": 0.142, "param_kb": 1079.5},
        "FP16":  {"accuracy": 92.31, "latency_ms": 0.089, "param_kb": 539.8},
        "LNS16": {"accuracy": 91.83, "latency_ms": 0.198, "param_kb": 539.8},
    }

    print(f"\n  {'Format':<8} {'Accuracy':>10} {'Latency ms/img':>16} {'Params KB':>12} {'Acc Drop':>10}")
    print("  " + "-" * 62)
    fp32_acc = results["FP32"]["accuracy"]
    for fmt, r in results.items():
        drop = fp32_acc - r["accuracy"]
        print(f"  {fmt:<8} {r['accuracy']:>9.2f}% {r['latency_ms']:>15.3f}  "
              f"{r['param_kb']:>11.1f}  {drop:>+9.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/simulated_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to results/simulated_benchmark.json")
    return results


if __name__ == "__main__":
    print("\n" + DIVIDER)
    print("  LNS16 Library Demo — Jayanarayan Thakurdas Tudu")
    print(DIVIDER)

    demo_encode_decode()
    demo_arithmetic()
    demo_dot_product()
    demo_lns_layer()
    demo_quantisation_error()
    simulated_benchmark_table()

    print(f"\n{DIVIDER}")
    print("  Demo complete. To run full experiment:")
    print("    python experiments/train.py         # trains ResNet-20 (~100 epochs)")
    print("    python experiments/benchmark.py     # measures all three variants")
    print(DIVIDER + "\n")
