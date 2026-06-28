"""
Train ResNet-20 on CIFAR-10 (FP32 baseline).
Saves weights to checkpoints/resnet20_fp32.pth

Windows-compatible: num_workers=0, pin_memory=False
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models.resnet import fp32_resnet20

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS    = 30
BATCH     = 128
LR        = 0.1
MOMENTUM  = 0.9
WD        = 1e-4
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR  = "checkpoints"

# Windows: always use num_workers=0 (no forking)
NUM_WORKERS = 0 if sys.platform == "win32" else 4

# ── Data ──────────────────────────────────────────────────────────────────────
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_loaders():
    train_set = torchvision.datasets.CIFAR10(
        "data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        "data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(
        train_set, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False)
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += y.size(0)
    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        correct += out.argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return 100 * correct / total


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}  |  Workers: {NUM_WORKERS}")

    train_loader, test_loader = get_loaders()

    model     = fp32_resnet20().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc        = evaluate(model, test_loader)
        scheduler.step()

        print(f"[{epoch:3d}/{EPOCHS}] loss={loss:.4f}  "
              f"train={train_acc:.2f}%  test={test_acc:.2f}%  "
              f"({time.time()-t0:.1f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       f"{CKPT_DIR}/resnet20_fp32.pth")
            print(f"           ↑ new best — saved checkpoint")

    print(f"\nBest FP32 test accuracy: {best_acc:.2f}%")


# ── Windows multiprocessing guard ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
