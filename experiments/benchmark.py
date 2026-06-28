"""
Inference Benchmark: FP32 vs FP16 vs LNS16
Windows-compatible (num_workers=0, pin_memory=False)
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models.resnet import fp32_resnet20, lns_resnet20

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CKPT        = "checkpoints/resnet20_fp32.pth"
RESULTS     = "results/benchmark.json"
NUM_WORKERS = 0 if sys.platform == "win32" else 4
os.makedirs("results", exist_ok=True)

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_test_loader():
    test_set = torchvision.datasets.CIFAR10(
        "data", train=False, download=True, transform=test_tf)
    return DataLoader(test_set, batch_size=256, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=False)


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


@torch.no_grad()
def evaluate_accuracy(model, loader, half=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if half:
            x = x.half()
        out = model(x)
        correct += out.argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def measure_latency(model, batch_size=64, n_images=640, half=False, n_runs=3):
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32, device=DEVICE)
    if half:
        dummy = dummy.half()
    steps = n_images // batch_size
    times = []
    for _ in range(n_runs):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            model(dummy)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / (steps * batch_size) * 1000)
    return float(np.mean(times)), float(np.std(times))


def run():
    loader  = get_test_loader()
    results = {}

    # ── FP32 ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FP32")
    model = fp32_resnet20().to(DEVICE)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f"  Loaded: {CKPT}")
    else:
        print("  WARNING: no checkpoint, using random weights")
    acc        = evaluate_accuracy(model, loader)
    lat, lstd  = measure_latency(model)
    mem        = param_bytes(model)
    print(f"  Accuracy={acc:.2f}%  Latency={lat:.4f}ms  Mem={mem/1024:.1f}KB")
    results["FP32"] = {"accuracy": acc, "latency_ms": lat,
                       "latency_std": lstd, "param_kb": mem/1024}

    # ── FP16 ─────────────────────────────────────────────────────────────────
    print("\n  FP16")
    model_h = fp32_resnet20().to(DEVICE)
    if os.path.exists(CKPT):
        model_h.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model_h = model_h.half()
    acc_h        = evaluate_accuracy(model_h, loader, half=True)
    lat_h, lstd_h = measure_latency(model_h, half=True)
    mem_h        = param_bytes(model_h)
    print(f"  Accuracy={acc_h:.2f}%  Latency={lat_h:.4f}ms  Mem={mem_h/1024:.1f}KB")
    results["FP16"] = {"accuracy": acc_h, "latency_ms": lat_h,
                       "latency_std": lstd_h, "param_kb": mem_h/1024}

    # ── LNS16 ────────────────────────────────────────────────────────────────
    print("\n  LNS16")
    model_l = lns_resnet20().to(DEVICE)
    if os.path.exists(CKPT):
        model_l.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
    acc_l        = evaluate_accuracy(model_l, loader)
    lat_l, lstd_l = measure_latency(model_l)
    mem_l        = param_bytes(model_l)
    print(f"  Accuracy={acc_l:.2f}%  Latency={lat_l:.4f}ms  Mem={mem_l/1024:.1f}KB")
    results["LNS16"] = {"accuracy": acc_l, "latency_ms": lat_l,
                        "latency_std": lstd_l, "param_kb": mem_l/1024}

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Format':<8} {'Accuracy':>10} {'ms/img':>10} {'Params KB':>12} {'Acc Drop':>10}")
    print("-" * 55)
    fp32_acc = results["FP32"]["accuracy"]
    for fmt, r in results.items():
        drop = fp32_acc - r["accuracy"]
        print(f"{fmt:<8} {r['accuracy']:>9.2f}% {r['latency_ms']:>10.4f}"
              f"  {r['param_kb']:>10.1f}  {drop:>+9.2f}%")

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS}")
    return results


if __name__ == "__main__":
    run()
