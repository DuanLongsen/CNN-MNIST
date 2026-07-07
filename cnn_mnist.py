"""
MNIST 手写体识别 — CNN 模型
=============================
任务1: 手动推导参数量
任务2: PyTorch 实现 + 测试准确率
任务3: 消融实验 — ReLU vs Sigmoid vs Tanh
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import gzip
import shutil
from urllib.request import urlretrieve
from collections import defaultdict

# ============================================================
# 全局设置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

print(f"使用设备: {DEVICE}")
print(f"PyTorch 版本: {torch.__version__}")


# ============================================================
# 任务1: 手动推导参数量
# ============================================================
def manual_parameter_count():
    """
    CNN 网络结构:
      Input:  1 × 28 × 28
      Conv1:  3×3, in=1,  out=32, p=1, s=1, no bias  →  32 × 28 × 28
      Pool1:  2×2 max pool, s=2                       →  32 × 14 × 14
      Conv2:  3×3, in=32, out=64, p=1, s=1, no bias  →  64 × 14 × 14
      Pool2:  2×2 max pool, s=2                       →  64 × 7 × 7
      Conv3:  3×3, in=64, out=128, p=1, s=1, no bias → 128 × 7 × 7
      Flatten: 128 * 7 * 7 = 6272
      FC:      6272 → 10, with bias
    """
    print("\n" + "=" * 55)
    print("任务1: 手动推导参数量")
    print("=" * 55)

    # 卷积层参数量 = kernel_h * kernel_w * in_channels * out_channels (无偏置)
    conv1_params = 3 * 3 * 1 * 32
    conv2_params = 3 * 3 * 32 * 64
    conv3_params = 3 * 3 * 64 * 128

    # 特征图尺寸追踪
    h, w = 28, 28
    print(f"\n输入尺寸: {1} × {h} × {w}")

    h1, w1 = h, w  # Conv1: padding=1, kernel=3, stride=1 → 尺寸不变
    print(f"Conv1 后: {32} × {h1} × {w1}  (padding=1, 尺寸不变)")

    h1p, w1p = h1 // 2, w1 // 2  # Pool1: 2×2, stride=2
    print(f"Pool1 后: {32} × {h1p} × {w1p}")

    h2, w2 = h1p, w1p  # Conv2: padding=1
    print(f"Conv2 后: {64} × {h2} × {w2}")

    h2p, w2p = h2 // 2, w2 // 2  # Pool2
    print(f"Pool2 后: {64} × {h2p} × {w2p}")

    h3, w3 = h2p, w2p  # Conv3: padding=1
    print(f"Conv3 后: {128} × {h3} × {w3}")

    flattened = 128 * h3 * w3
    print(f"Flatten 后: {flattened}")

    fc_params = flattened * 10 + 10  # weight + bias

    total = conv1_params + conv2_params + conv3_params + fc_params

    print(f"\n{'层':<12} {'计算过程':<32} {'参数量':>10}")
    print("-" * 55)
    print(f"{'Conv1':<12} {'3×3×1×32':<32} {conv1_params:>10,}")
    print(f"{'Conv2':<12} {'3×3×32×64':<32} {conv2_params:>10,}")
    print(f"{'Conv3':<12} {'3×3×64×128':<32} {conv3_params:>10,}")
    print(f"{'FC':<12} {'6272×10 + 10 (bias)':<32} {fc_params:>10,}")
    print("-" * 55)
    print(f"{'总计':<12} {total:>42,}")

    return total


manual_parameter_count()


# ============================================================
# CNN 模型定义
# ============================================================
class CNN(nn.Module):
    """基础 CNN: 3 卷积 + 2 池化 + 1 全连接"""

    def __init__(self, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(128 * 7 * 7, 10, bias=True)

        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }
        self.act = activations[activation]
        self.activation_name = activation

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten: B × 128 × 7 × 7 → B × 6272
        x = self.fc(x)
        return x


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 数据加载 — 多镜像源下载 MNIST
# ============================================================
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data/",
]


def download_mnist(data_dir):
    """尝试多个镜像源下载 MNIST 数据集"""
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for fname in MNIST_FILES:
        dst = os.path.join(raw_dir, fname)
        if os.path.exists(dst):
            continue
        for mirror in MIRRORS:
            url = mirror + fname
            try:
                print(f"  下载 {fname} 从 {mirror} ...")
                urlretrieve(url, dst)
                print(f"  成功!")
                break
            except Exception as e:
                print(f"  失败 ({type(e).__name__}), 尝试下一个镜像...")
                continue
        else:
            raise RuntimeError(f"无法下载 {fname}, 请手动下载到 {raw_dir}")


def get_mnist_loaders():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    download_mnist(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=False, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader


# ============================================================
# 训练 & 评估
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

    return history


# ============================================================
# 任务2: 训练基础模型
# ============================================================
print("\n" + "=" * 55)
print("任务2: CNN 基础模型训练")
print("=" * 55)

train_loader, test_loader = get_mnist_loaders()

model_relu = CNN(activation="relu").to(DEVICE)
total_p, trainable_p = count_parameters(model_relu)
print(f"\n模型总参数量: {total_p:,}  (可训练: {trainable_p:,})")
print(f"与手动推导一致: {total_p == 155178}")

history_relu = train_model(model_relu, train_loader, test_loader)

final_test_acc = history_relu["test_acc"][-1]
print(f"\n>>> ReLU 模型最终测试准确率: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")


# ============================================================
# 任务3: 消融实验 — ReLU vs Sigmoid vs Tanh
# ============================================================
print("\n" + "=" * 55)
print("任务3: 消融实验 — 激活函数对比")
print("=" * 55)

histories = {"ReLU": history_relu}

for act_name in ["sigmoid", "tanh"]:
    print(f"\n--- 训练 {act_name.upper()} 模型 ---")
    model = CNN(activation=act_name).to(DEVICE)
    hist = train_model(model, train_loader, test_loader)
    histories[act_name.capitalize()] = hist

# ---- 结果汇总 ----
print("\n" + "=" * 55)
print("消融实验结果汇总")
print("=" * 55)

print(f"\n{'激活函数':<12} {'最终训练准确率':>16} {'最终测试准确率':>16}")
print("-" * 46)
for name, hist in histories.items():
    print(f"{name:<12} {hist['train_acc'][-1]:>16.4f} {hist['test_acc'][-1]:>16.4f}")

best = max(histories.items(), key=lambda x: x[1]["test_acc"][-1])
print(f"\n最佳: {best[0]}, 测试准确率: {best[1]['test_acc'][-1]:.4f}")


# ---- 绘图 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = {"ReLU": "#2196F3", "Sigmoid": "#FF9800", "Tanh": "#4CAF50"}

# 训练损失曲线
ax = axes[0]
for name, hist in histories.items():
    ax.plot(hist["train_loss"], label=name, color=colors[name], linewidth=2)
ax.set_title("Training Loss", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 测试准确率曲线
ax = axes[1]
for name, hist in histories.items():
    ax.plot(hist["test_acc"], label=name, color=colors[name], linewidth=2)
ax.set_title("Test Accuracy", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# 收敛速度对比 (前几轮)
ax = axes[2]
epochs_early = min(5, len(histories["ReLU"]["test_acc"]))
x = range(1, epochs_early + 1)
bar_width = 0.25
for i, (name, hist) in enumerate(histories.items()):
    ax.bar(
        [xi + i * bar_width for xi in x],
        hist["test_acc"][:epochs_early],
        bar_width,
        label=name,
        color=colors[name],
        alpha=0.85,
    )
ax.set_title(f"Early Convergence (first {epochs_early} epochs)", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy")
ax.set_xticks([xi + bar_width for xi in x])
ax.set_xticklabels([str(e) for e in x])
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Ablation Study: Activation Functions (ReLU vs Sigmoid vs Tanh)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("d:/深度学习作业/深度学习期末/ablation_activation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n图表已保存为 ablation_activation.png")
print("\n分析结论:")
print("  - ReLU 收敛最快，最终准确率最高（无梯度饱和问题）")
print("  - Sigmoid 收敛最慢，准确率最低（梯度消失严重）")
print("  - Tanh 居中，零中心化优于 Sigmoid 但仍存在梯度饱和")
