"""
Minimal Knowledge Distillation Reproduction
=============================================
Reproduces core ideas from knowledge distillation literature:
1. Hinton et al. (2015): soft targets with temperature
2. Teacher-student training on MNIST
3. Compare: hard labels vs soft labels vs combined
4. Effect of temperature on knowledge transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Models ──

class TeacherNet(nn.Module):
    """Larger teacher network."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


class StudentNet(nn.Module):
    """Smaller student network."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


class TinyStudent(nn.Module):
    """Even smaller student."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


# ── Distillation Loss ──

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """Knowledge distillation loss.

    L = α * T² * KL(σ(z_t/T) || σ(z_s/T)) + (1-α) * CE(z_s, y)

    The T² factor compensates for the softmax softening — the gradients
    from the soft targets are scaled by 1/T², so multiplying by T² ensures
    the relative contribution of hard and soft targets is balanced.
    """
    # Soft target loss (KL divergence)
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    # Use KL divergence with teacher as target
    kl_loss = F.kl_div(soft_student, F.softmax(teacher_logits / temperature, dim=-1),
                        reduction='batchmean') * (temperature ** 2)

    # Hard target loss
    ce_loss = F.cross_entropy(student_logits, labels)

    return alpha * kl_loss + (1 - alpha) * ce_loss


# ── Training Functions ──

def train_teacher(model, train_loader, n_epochs=5, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_student_hard(student, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    """Train student with hard labels only (standard training)."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = student(bx)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    return losses


def train_student_distill(student, teacher, train_loader, n_epochs=10,
                          lr=1e-3, temperature=4.0, alpha=0.7, device='cpu'):
    """Train student with knowledge distillation."""
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                teacher_logits = teacher(bx)
            student_logits = student(bx)
            loss = distillation_loss(student_logits, teacher_logits, by,
                                      temperature=temperature, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    return losses


def evaluate(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += by.shape[0]
    return correct / total


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "14-distillation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load MNIST
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Train teacher
    print("=== Training Teacher ===")
    teacher = TeacherNet().to(device)
    train_teacher(teacher, train_loader, n_epochs=5, device=device)
    teacher_acc = evaluate(teacher, test_loader, device)
    print(f"  Teacher accuracy: {teacher_acc:.4f}")

    # Train students with different strategies
    n_epochs = 10
    n_runs = 1  # Single run for speed

    strategies = {
        'Hard Labels': lambda s: train_student_hard(s, train_loader, n_epochs, device=device),
        'KD (T=2)': lambda s: train_student_distill(s, teacher, train_loader, n_epochs,
                                                      temperature=2.0, device=device),
        'KD (T=4)': lambda s: train_student_distill(s, teacher, train_loader, n_epochs,
                                                      temperature=4.0, device=device),
        'KD (T=8)': lambda s: train_student_distill(s, teacher, train_loader, n_epochs,
                                                      temperature=8.0, device=device),
        'KD (T=16)': lambda s: train_student_distill(s, teacher, train_loader, n_epochs,
                                                       temperature=16.0, device=device),
    }

    student_types = {
        'Student (64h)': StudentNet,
        'Tiny (16h)': TinyStudent,
    }

    all_results = {}

    for sname, SClass in student_types.items():
        print(f"\n=== {sname} ===")
        all_results[sname] = {}

        for strat_name, train_fn in strategies.items():
            accs = []
            for run in range(n_runs):
                student = SClass().to(device)
                train_fn(student)
                acc = evaluate(student, test_loader, device)
                accs.append(acc)
            avg_acc = np.mean(accs)
            all_results[sname][strat_name] = avg_acc
            print(f"  {strat_name}: {avg_acc:.4f} (±{np.std(accs):.4f})")

    # ── Temperature sensitivity analysis ──
    print("\n=== Temperature Sensitivity ===")
    temperatures = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 16.0, 20.0]
    temp_results = {}

    for temp in temperatures:
        accs = []
        for run in range(n_runs):
            student = StudentNet().to(device)
            train_student_distill(student, teacher, train_loader, n_epochs,
                                   temperature=temp, device=device)
            acc = evaluate(student, test_loader, device)
            accs.append(acc)
        temp_results[temp] = np.mean(accs)
        print(f"  T={temp:.0f}: {temp_results[temp]:.4f}")

    # ── Alpha sensitivity analysis ──
    print("\n=== Alpha Sensitivity (T=4) ===")
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alpha_results = {}

    for alpha in alphas:
        accs = []
        for run in range(n_runs):
            student = StudentNet().to(device)
            train_student_distill(student, teacher, train_loader, n_epochs,
                                   temperature=4.0, alpha=alpha, device=device)
            acc = evaluate(student, test_loader, device)
            accs.append(acc)
        alpha_results[alpha] = np.mean(accs)
        print(f"  α={alpha:.1f}: {alpha_results[alpha]:.4f}")

    # ── Visualization ──

    # 1. Strategy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (sname, strats) in enumerate(all_results.items()):
        ax = axes[idx]
        names = list(strats.keys())
        values = list(strats.values())
        colors = ['red' if 'Hard' in n else f'C{i}' for i, n in enumerate(names)]
        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.7)
        ax.axhline(y=teacher_acc, color='green', linestyle='--', alpha=0.5, label=f'Teacher ({teacher_acc:.3f})')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{sname}: Hard Labels vs Knowledge Distillation")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.9, 1.0)

    plt.tight_layout()
    plt.savefig(results_dir / "strategy_comparison.png", dpi=150)
    plt.close()

    # 2. Temperature sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(temp_results.keys()), list(temp_results.values()),
            'o-', color='blue', linewidth=2, markersize=8)
    ax.axhline(y=teacher_acc, color='green', linestyle='--', alpha=0.5, label='Teacher')
    ax.axhline(y=all_results['Student (64h)']['Hard Labels'], color='red',
               linestyle='--', alpha=0.5, label='Student (hard labels)')
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Knowledge Distillation: Effect of Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "temperature_sensitivity.png", dpi=150)
    plt.close()

    # 3. Alpha sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(alpha_results.keys()), list(alpha_results.values()),
            'o-', color='purple', linewidth=2, markersize=8)
    ax.axhline(y=teacher_acc, color='green', linestyle='--', alpha=0.5, label='Teacher')
    ax.set_xlabel("α (soft target weight)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Knowledge Distillation: Effect of α (T=4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "alpha_sensitivity.png", dpi=150)
    plt.close()

    # 4. Soft target visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Get a sample and show teacher's soft predictions at different temperatures
    sample_x, sample_y = next(iter(test_loader))
    sample_x, sample_y = sample_x[:1].to(device), sample_y[:1].to(device)

    with torch.no_grad():
        teacher_logits = teacher(sample_x)[0]

    for ax, temp in zip(axes, [1.0, 4.0, 20.0]):
        soft_probs = F.softmax(teacher_logits / temp, dim=-1).cpu().numpy()
        hard_probs = F.softmax(teacher_logits, dim=-1).cpu().numpy()

        ax.bar(range(10), soft_probs, alpha=0.7, color='blue', label=f'Soft (T={temp:.0f})')
        ax.bar(range(10), hard_probs, alpha=0.3, color='red', label='Hard (T=1)')
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_title(f"Teacher Output (T={temp:.0f}), True={sample_y.item()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Dark Knowledge: Soft Targets at Different Temperatures", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "soft_targets.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
