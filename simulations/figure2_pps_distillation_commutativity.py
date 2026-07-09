"""
Figure 2 — PPS-Aware Knowledge Distillation and Commutativity
=============================================================

This experiment evaluates the PPS commutativity condition

    Phi Pi_T ≈ Pi_S Phi

in a teacher-student knowledge distillation setting on CIFAR-10.

A ResNet-18 teacher is compared with a compact CNN student trained using
either standard KL distillation or PPS-aware distillation with an explicit
commutativity regularization term.

The experiment compares the commutativity gap and predictive accuracy
during training.

Outputs:
    pps_results/figure2_high_resolution_vertical.pdf
    pps_results/figure2_high_resolution_vertical.png
    pps_results/pps_histories.npz
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 128
EPOCHS      = 10
LR          = 1e-3
LAMBDA_PPS  = 0.5
NUM_CLASSES = 10
DATA_DIR    = "./data"
SAVE_DIR    = "./pps_results"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)

# ─── Models ───────────────────────────────────────────────────────────────────
def make_teacher():
    return models.resnet18(weights=None, num_classes=NUM_CLASSES)

def make_student():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, NUM_CLASSES),
    )

# ─── PPS Operators ────────────────────────────────────────────────────────────
def get_last_linear(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    assert last is not None, "No Linear layer found"
    return last

def get_Pi(model):
    W = get_last_linear(model).weight
    return W.T @ W

def get_penultimate_dim(model):
    return get_last_linear(model).in_features

def make_Phi(model_T, model_S):
    dim_T = get_penultimate_dim(model_T)
    dim_S = get_penultimate_dim(model_S)
    return nn.Linear(dim_T, dim_S, bias=False)

def commutativity_loss(Phi, Pi_T, Pi_S):
    W = Phi.weight
    return torch.norm(W @ Pi_T - Pi_S @ W, p="fro") ** 2

# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total

# ─── Training ─────────────────────────────────────────────────────────────────
def train_teacher(model, epochs=EPOCHS):
    print("\n=== Training Teacher (ResNet18) ===")
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(model(x), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        acc = evaluate(model)
        print(f"  Epoch {ep+1:02d}/{epochs}  loss={total_loss/len(train_loader):.3f}  acc={acc:.2f}%")

    return model

def train_student(mode, teacher, student, Phi, epochs=EPOCHS):
    print(f"\n=== Training Student [{mode}] ===")

    student.to(DEVICE)
    Phi.to(DEVICE)
    teacher.to(DEVICE)
    teacher.eval()

    params = list(student.parameters()) + list(Phi.parameters())
    opt = optim.Adam(params, lr=LR)

    gap_history = []
    acc_history = []

    for ep in range(epochs):
        student.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                logits_T = teacher(x)

            logits_S = student(x)

            T = 4.0
            loss_kl = F.kl_div(
                F.log_softmax(logits_S / T, dim=1),
                F.softmax(logits_T / T, dim=1),
                reduction="batchmean",
            ) * (T ** 2)

            if mode == "pps":
                Pi_T = get_Pi(teacher).detach()
                Pi_S = get_Pi(student)
                loss_pps = commutativity_loss(Phi, Pi_T, Pi_S)
                loss = loss_kl + LAMBDA_PPS * loss_pps
            else:
                loss = loss_kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        with torch.no_grad():
            Pi_T = get_Pi(teacher).detach()
            Pi_S = get_Pi(student).detach()
            W = Phi.weight
            gap = torch.norm(W @ Pi_T - Pi_S @ W, p="fro").item()

        gap_history.append(gap)

        acc = evaluate(student)
        acc_history.append(acc)

        print(
            f"  Epoch {ep+1:02d}/{epochs}  "
            f"loss={total_loss/len(train_loader):.3f}  "
            f"acc={acc:.2f}%  gap={gap:.4f}"
        )

    return student, gap_history, acc_history

# ─── High-resolution Figure ───────────────────────────────────────────────────
def make_high_resolution_vertical_figure(
    gap_std,
    acc_std,
    gap_pps,
    acc_pps,
    final_gap_std,
    final_gap_pps,
):
    BASE_FONT = 24

    matplotlib.rcParams.update({
        "font.size": BASE_FONT,
        "axes.titlesize": BASE_FONT + 4,
        "axes.labelsize": BASE_FONT + 2,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT - 2,
        "figure.titlesize": BASE_FONT + 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    epochs_ax = range(1, len(gap_std) + 1)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 24)
    )

    # (a) Commutativity gap
    ax = axes[0]
    ax.plot(
        epochs_ax,
        gap_std,
        marker="o",
        linewidth=4,
        markersize=12,
        label="Standard KL",
    )
    ax.plot(
        epochs_ax,
        gap_pps,
        marker="s",
        linewidth=4,
        markersize=12,
        label="PPS-aware",
    )
    ax.set_xlabel("Epoch", labelpad=12)
    ax.set_ylabel(r"$\|\Phi\Pi_T-\Pi_S\Phi\|_F$", labelpad=12)
    ax.set_title("(a) Commutativity Gap During Training", pad=22)
    ax.set_xticks(list(epochs_ax))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="upper left")

    # (b) Test accuracy
    ax = axes[1]
    ax.plot(
        epochs_ax,
        acc_std,
        marker="o",
        linewidth=4,
        markersize=12,
        label="Standard KL",
    )
    ax.plot(
        epochs_ax,
        acc_pps,
        marker="s",
        linewidth=4,
        markersize=12,
        label="PPS-aware",
    )
    ax.set_xlabel("Epoch", labelpad=12)
    ax.set_ylabel("Test Accuracy (%)", labelpad=12)
    ax.set_title("(b) Test Accuracy During Training", pad=22)
    ax.set_xticks(list(epochs_ax))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="lower right")

    # (c) Final commutativity gap
    ax = axes[2]

    labels = ["Standard KL", "PPS-aware"]
    values = [final_gap_std, final_gap_pps]

    bars = ax.bar(
        labels,
        values,
        width=0.45,
        edgecolor="black",
        linewidth=1.8,
    )

    ymax = max(values) * 1.30
    ax.set_ylim(0, ymax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=BASE_FONT + 2,
            fontweight="bold",
        )

    ax.set_ylabel(r"$\|\Phi\Pi_T-\Pi_S\Phi\|_F$ (final)", labelpad=12)
    ax.set_title("(c) Final Commutativity Gap", pad=22)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        r"PPS Experiment: $\Phi\Pi_T \approx \Pi_S\Phi$ Verification (CIFAR-10)",
        fontsize=BASE_FONT + 8,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.975])

    out_pdf = f"{SAVE_DIR}/figure2_high_resolution_vertical.pdf"
    out_png = f"{SAVE_DIR}/figure2_high_resolution_vertical.png"

    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)

    plt.show()

    print("\nSaved high-resolution Figure 2:")
    print(out_pdf)
    print(out_png)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    teacher = make_teacher()
    teacher = train_teacher(teacher, epochs=EPOCHS)
    torch.save(teacher.state_dict(), f"{SAVE_DIR}/teacher.pt")

    student_std = make_student()
    Phi_std = make_Phi(teacher, student_std)

    student_std, gap_std, acc_std = train_student(
        "standard",
        teacher,
        student_std,
        Phi_std,
        epochs=EPOCHS,
    )

    student_pps = make_student()
    Phi_pps = make_Phi(teacher, student_pps)

    student_pps, gap_pps, acc_pps = train_student(
        "pps",
        teacher,
        student_pps,
        Phi_pps,
        epochs=EPOCHS,
    )

    with torch.no_grad():
        Pi_T = get_Pi(teacher).detach()

        W_std = Phi_std.weight
        Pi_S_std = get_Pi(student_std).detach()
        final_gap_std = torch.norm(W_std @ Pi_T - Pi_S_std @ W_std, p="fro").item()

        W_pps = Phi_pps.weight
        Pi_S_pps = get_Pi(student_pps).detach()
        final_gap_pps = torch.norm(W_pps @ Pi_T - Pi_S_pps @ W_pps, p="fro").item()

    final_acc_std = evaluate(student_std)
    final_acc_pps = evaluate(student_pps)

    print(f"\n{'='*60}")
    print(f"  Standard KL  gap={final_gap_std:.4f}  acc={final_acc_std:.2f}%")
    print(f"  PPS-aware    gap={final_gap_pps:.4f}  acc={final_acc_pps:.2f}%")
    print(f"{'='*60}")

    # Save numerical histories
    np.savez(
        f"{SAVE_DIR}/pps_histories.npz",
        gap_std=np.array(gap_std),
        acc_std=np.array(acc_std),
        gap_pps=np.array(gap_pps),
        acc_pps=np.array(acc_pps),
        final_gap_std=final_gap_std,
        final_gap_pps=final_gap_pps,
        final_acc_std=final_acc_std,
        final_acc_pps=final_acc_pps,
    )

    print(f"\nSaved numerical histories:")
    print(f"{SAVE_DIR}/pps_histories.npz")

    # Make large high-resolution vertical figure
    make_high_resolution_vertical_figure(
        gap_std,
        acc_std,
        gap_pps,
        acc_pps,
        final_gap_std,
        final_gap_pps,
    )

    return gap_std, acc_std, gap_pps, acc_pps, final_gap_std, final_gap_pps

# ─── Run ──────────────────────────────────────────────────────────────────────
gap_std, acc_std, gap_pps, acc_pps, final_gap_std, final_gap_pps = main()
