"""
Minimal HuggingGPT-Style Task Planning Reproduction
=====================================================
Reproduces core ideas from HuggingGPT (2303.17580, Shen et al.):
1. LLM acts as planner: decompose user request into subtasks
2. Select appropriate AI models for each subtask from a model hub
3. Execute subtasks in dependency order (DAG scheduling)
4. Summarize results into a coherent response
5. Key insight: LLM as orchestrator connecting specialized models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json
from collections import defaultdict


# ── Toy Model Hub ──

MODELS = {
    "image_generator":  {"input": "text",    "output": "image",  "cost": 2.0, "accuracy": 0.9},
    "image_captioner":  {"input": "image",   "output": "text",   "cost": 1.0, "accuracy": 0.85},
    "object_detector":  {"input": "image",   "output": "boxes",  "cost": 1.5, "accuracy": 0.88},
    "translator":       {"input": "text",    "output": "text",   "cost": 0.5, "accuracy": 0.92},
    "summarizer":       {"input": "text",    "output": "text",   "cost": 0.8, "accuracy": 0.87},
    "sentiment":        {"input": "text",    "output": "label",  "cost": 0.3, "accuracy": 0.91},
    "tts":              {"input": "text",    "output": "audio",  "cost": 1.0, "accuracy": 0.80},
    "asr":              {"input": "audio",   "output": "text",   "cost": 1.0, "accuracy": 0.83},
    "face_detector":    {"input": "image",   "output": "boxes",  "cost": 1.2, "accuracy": 0.90},
    "ocr":              {"input": "image",   "output": "text",   "cost": 1.0, "accuracy": 0.86},
    "qa":               {"input": "text",    "output": "text",   "cost": 0.6, "accuracy": 0.88},
    "style_transfer":   {"input": "image",   "output": "image",  "cost": 2.5, "accuracy": 0.78},
}

TYPE_COMPAT = {
    "text": ["text"], "image": ["image"], "audio": ["audio"],
    "boxes": ["boxes"], "label": ["label"],
}

MODEL_NAMES = list(MODELS.keys())


def execute_model(model_name, input_data, rng=None):
    """Simulate model execution with accuracy-based success rate."""
    if rng is None:
        rng = random.Random()
    acc = MODELS[model_name]["accuracy"]
    success = rng.random() < acc
    output_type = MODELS[model_name]["output"]
    if success:
        return {"status": "success", "output_type": output_type, "data": f"{output_type}_result"}
    else:
        return {"status": "error", "output_type": output_type, "data": None}


# ── Task Decomposition ──

# Define request templates that require multi-step planning
REQUEST_TEMPLATES = [
    {
        "request": "Generate an image of a cat and describe what you see",
        "plan": [
            {"task": "generate cat image",  "model": "image_generator", "dep": []},
            {"task": "describe the image",   "model": "image_captioner", "dep": [0]},
        ],
        "difficulty": 1,
    },
    {
        "request": "Translate 'Hello World' to French and read it aloud",
        "plan": [
            {"task": "translate text",       "model": "translator",     "dep": []},
            {"task": "read aloud",           "model": "tts",            "dep": [0]},
        ],
        "difficulty": 1,
    },
    {
        "request": "Analyze the sentiment of this review and summarize it",
        "plan": [
            {"task": "sentiment analysis",   "model": "sentiment",      "dep": []},
            {"task": "summarize text",       "model": "summarizer",     "dep": []},
        ],
        "difficulty": 1,
    },
    {
        "request": "Generate an image of a sunset, detect objects in it, and describe it",
        "plan": [
            {"task": "generate sunset",      "model": "image_generator","dep": []},
            {"task": "detect objects",       "model": "object_detector","dep": [0]},
            {"task": "describe image",       "model": "image_captioner","dep": [0]},
        ],
        "difficulty": 2,
    },
    {
        "request": "Read the text in this photo and translate it to English",
        "plan": [
            {"task": "extract text from image", "model": "ocr",         "dep": []},
            {"task": "translate to English",    "model": "translator",  "dep": [0]},
        ],
        "difficulty": 2,
    },
    {
        "request": "Detect faces, generate descriptions, and translate to Spanish",
        "plan": [
            {"task": "detect faces",         "model": "face_detector",  "dep": []},
            {"task": "describe faces",       "model": "image_captioner","dep": [0]},
            {"task": "translate to Spanish",  "model": "translator",    "dep": [1]},
        ],
        "difficulty": 2,
    },
    {
        "request": "Transcribe audio, summarize it, and answer questions about it",
        "plan": [
            {"task": "transcribe audio",     "model": "asr",            "dep": []},
            {"task": "summarize text",       "model": "summarizer",     "dep": [0]},
            {"task": "answer questions",     "model": "qa",             "dep": [0]},
        ],
        "difficulty": 2,
    },
    {
        "request": "Generate a landscape, apply style transfer, detect objects, and caption",
        "plan": [
            {"task": "generate landscape",   "model": "image_generator","dep": []},
            {"task": "apply style",          "model": "style_transfer", "dep": [0]},
            {"task": "detect objects",       "model": "object_detector","dep": [1]},
            {"task": "caption image",        "model": "image_captioner","dep": [1]},
        ],
        "difficulty": 3,
    },
]


# ── Planner Network ──

class TaskEncoder(nn.Module):
    """Encode a task description into an embedding."""
    def __init__(self, vocab_size=50, d_model=64, max_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        h = h.mean(dim=1)
        return self.proj(h)


class PlannerModel(nn.Module):
    """HuggingGPT-style planner: given request, output task sequence.

    Three heads:
    1. n_tasks: predict how many subtasks (1-5)
    2. model_selection: for each subtask, which model to use
    3. dependency: for each subtask, which previous tasks it depends on
    """
    def __init__(self, vocab_size=50, d_model=64, n_models=len(MODEL_NAMES), max_tasks=5):
        super().__init__()
        self.n_models = n_models
        self.max_tasks = max_tasks
        self.task_encoder = TaskEncoder(vocab_size, d_model)
        self.request_proj = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, d_model)
        )

        # Head 1: number of tasks
        self.n_tasks_head = nn.Linear(d_model, max_tasks)

        # Head 2: model selection per task (shared weight, conditioned on task index)
        self.model_head = nn.Sequential(
            nn.Linear(d_model + max_tasks, 128), nn.ReLU(), nn.Linear(128, n_models)
        )

        # Head 3: dependency prediction (for task i, which of 0..i-1 it depends on)
        self.dep_head = nn.Sequential(
            nn.Linear(d_model + max_tasks, 64), nn.ReLU(), nn.Linear(64, max_tasks)
        )

    def forward(self, request_tokens):
        """request_tokens: (B, T)"""
        req_emb = self.task_encoder(request_tokens)  # (B, D)
        req_emb = self.request_proj(req_emb)

        n_tasks_logits = self.n_tasks_head(req_emb)  # (B, max_tasks)

        # For each task position, predict model and dependencies
        model_logits_list = []
        dep_logits_list = []
        for t in range(self.max_tasks):
            # One-hot task index
            task_idx = F.one_hot(torch.tensor([t]), self.max_tasks).float().to(req_emb.device)
            task_idx = task_idx.expand(req_emb.size(0), -1)  # (B, max_tasks)
            combined = torch.cat([req_emb, task_idx], dim=-1)  # (B, D + max_tasks)

            model_logits_list.append(self.model_head(combined))   # (B, n_models)
            dep_logits_list.append(self.dep_head(combined))       # (B, max_tasks)

        model_logits = torch.stack(model_logits_list, dim=1)  # (B, max_tasks, n_models)
        dep_logits = torch.stack(dep_logits_list, dim=1)      # (B, max_tasks, max_tasks)

        return n_tasks_logits, model_logits, dep_logits


# ── Tokenizer ──

WORD_VOCAB = {}
VOCAB_SIZE = 50


def build_vocab():
    """Build simple vocabulary from request templates."""
    idx = 1  # 0 = PAD
    words = set()
    for tmpl in REQUEST_TEMPLATES:
        for w in tmpl["request"].lower().split():
            words.add(w)
        for task in tmpl["plan"]:
            for w in task["task"].lower().split():
                words.add(w)
    # Add common words
    for w in ["generate", "detect", "describe", "translate", "summarize", "analyze",
              "read", "apply", "extract", "transcribe", "caption"]:
        words.add(w)
    for w in sorted(words):
        WORD_VOCAB[w] = idx
        idx += 1
    return idx


def tokenize_request(text, max_len=16):
    """Tokenize request text."""
    tokens = []
    for w in text.lower().split():
        tokens.append(WORD_VOCAB.get(w, 0))
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return tokens


# ── DAG Execution ──

def execute_plan(plan, rng=None):
    """Execute a task plan as a DAG. Returns execution results and metrics."""
    if rng is None:
        rng = random.Random()

    n_tasks = len(plan)
    results = [None] * n_tasks
    execution_order = []
    total_cost = 0
    successes = 0

    # Topological sort (simple: tasks with no deps first, then dependent)
    completed = set()
    while len(completed) < n_tasks:
        ready = []
        for i, task in enumerate(plan):
            if i not in completed and all(d in completed for d in task["dep"]):
                ready.append(i)
        if not ready:
            break  # cycle or error
        for i in ready:
            model_name = task.get("model", plan[i].get("model", "qa"))
            result = execute_model(model_name, None, rng=rng)
            results[i] = result
            total_cost += MODELS[model_name]["cost"]
            if result["status"] == "success":
                successes += 1
            completed.add(i)
            execution_order.append(i)

    return {
        "results": results,
        "execution_order": execution_order,
        "total_cost": total_cost,
        "success_rate": successes / max(n_tasks, 1),
        "completed": len(completed) == n_tasks,
    }


# ── Training ──

def generate_training_data(n_samples=500):
    """Generate (request, plan) training pairs."""
    data = []
    rng = random.Random(42)

    for _ in range(n_samples):
        tmpl = rng.choice(REQUEST_TEMPLATES)
        request_tokens = tokenize_request(tmpl["request"])
        n_tasks = len(tmpl["plan"])
        model_indices = [MODEL_NAMES.index(t["model"]) for t in tmpl["plan"]]
        # Dependencies as binary vector per task
        deps = []
        for t in tmpl["plan"]:
            dep_vec = [0] * 5  # max_tasks
            for d in t["dep"]:
                if d < 5:
                    dep_vec[d] = 1
            deps.append(dep_vec)

        # Pad models and deps to max_tasks=5
        padded_models = model_indices + [0] * (5 - len(model_indices))
        while len(deps) < 5:
            deps.append([0] * 5)

        data.append({
            "request": request_tokens,
            "n_tasks": n_tasks,
            "models": padded_models,
            "deps": deps,
            "difficulty": tmpl["difficulty"],
        })

    return data


def train(model, data, n_epochs=30, batch_size=16, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    all_requests = torch.tensor([d["request"] for d in data], dtype=torch.long, device=device)
    all_n_tasks = torch.tensor([d["n_tasks"] - 1 for d in data], dtype=torch.long, device=device)  # 0-indexed
    all_models = torch.tensor([d["models"] for d in data], dtype=torch.long, device=device)
    all_deps = torch.tensor([d["deps"] for d in data], dtype=torch.float, device=device)

    n_samples = len(data)
    losses_total = []
    losses_ntasks = []
    losses_model = []
    losses_dep = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0
        epoch_nt = 0
        epoch_ml = 0
        epoch_dl = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            req = all_requests[idx]
            nt_target = all_n_tasks[idx]
            ml_target = all_models[idx]
            dep_target = all_deps[idx]

            nt_logits, ml_logits, dep_logits = model(req)

            # Loss 1: number of tasks
            nt_loss = F.cross_entropy(nt_logits, nt_target)

            # Loss 2: model selection per task
            B = req.size(0)
            ml_loss = torch.tensor(0.0, device=device)
            for t in range(5):
                # Only compute loss for tasks that exist in ground truth
                mask = (nt_target >= t).float()  # tasks where t < n_tasks
                if mask.sum() == 0:
                    continue
                per_sample_loss = F.cross_entropy(ml_logits[:, t], ml_target[:, t], reduction='none')
                ml_loss += (per_sample_loss * mask).sum() / mask.sum()

            # Loss 3: dependency prediction
            dep_loss = F.binary_cross_entropy_with_logits(
                dep_logits.view(-1), dep_target.view(-1)
            )

            loss = nt_loss + ml_loss + 0.5 * dep_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_nt += nt_loss.item()
            epoch_ml += ml_loss.item()
            epoch_dl += dep_loss.item()
            n_batches += 1

        losses_total.append(epoch_loss / n_batches)
        losses_ntasks.append(epoch_nt / n_batches)
        losses_model.append(epoch_ml / n_batches)
        losses_dep.append(epoch_dl / n_batches)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Total: {losses_total[-1]:.4f} | "
                  f"NT: {losses_ntasks[-1]:.4f} | Model: {losses_model[-1]:.4f} | "
                  f"Dep: {losses_dep[-1]:.4f}")

    return losses_total, losses_ntasks, losses_model, losses_dep


# ── Evaluation ──

def evaluate_planning(model, data, device='cpu'):
    """Evaluate planning accuracy: correct n_tasks, correct model selection, correct deps."""
    model.eval()
    n_correct_nt = 0
    n_correct_model = 0
    n_total_model = 0
    n_correct_dep = 0
    n_total_dep = 0
    n_samples = len(data)

    execution_results = []

    with torch.no_grad():
        for d in data:
            req = torch.tensor([d["request"]], dtype=torch.long, device=device)
            nt_logits, ml_logits, dep_logits = model(req)

            # Predict n_tasks
            pred_nt = nt_logits[0].argmax().item() + 1
            true_nt = d["n_tasks"]
            if pred_nt == true_nt:
                n_correct_nt += 1

            # Predict models
            for t in range(min(pred_nt, true_nt)):
                pred_model = ml_logits[0, t].argmax().item()
                true_model = d["models"][t] if t < len(d["models"]) else -1
                n_total_model += 1
                if pred_model == true_model:
                    n_correct_model += 1

            # Predict dependencies
            for t in range(min(pred_nt, true_nt)):
                pred_deps = (dep_logits[0, t].sigmoid() > 0.5).int().tolist()
                true_deps = d["deps"][t] if t < len(d["deps"]) else [0]*5
                for j in range(5):
                    n_total_dep += 1
                    if pred_deps[j] == true_deps[j]:
                        n_correct_dep += 1

            # Execute the predicted plan
            predicted_plan = []
            for t in range(pred_nt):
                model_name = MODEL_NAMES[ml_logits[0, t].argmax().item()]
                pred_deps_t = (dep_logits[0, t].sigmoid() > 0.5).int().tolist()
                dep_indices = [j for j, v in enumerate(pred_deps_t) if v and j < t]
                predicted_plan.append({"model": model_name, "dep": dep_indices})

            exec_result = execute_plan(predicted_plan, rng=random.Random(42))
            execution_results.append({
                "true_nt": true_nt,
                "pred_nt": pred_nt,
                "cost": exec_result["total_cost"],
                "success_rate": exec_result["success_rate"],
                "completed": exec_result["completed"],
                "difficulty": d["difficulty"],
            })

    model.train()

    nt_acc = n_correct_nt / n_samples
    model_acc = n_correct_model / max(n_total_model, 1)
    dep_acc = n_correct_dep / max(n_total_dep, 1)

    return {
        'n_tasks_acc': nt_acc,
        'model_acc': model_acc,
        'dep_acc': dep_acc,
        'execution_results': execution_results,
    }


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "86-hugginggpt"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build vocab
    actual_vocab = build_vocab()
    print(f"Vocab size: {actual_vocab}")

    # Generate data
    print("\n=== Generating Training Data ===")
    train_data = generate_training_data(n_samples=500)
    eval_data = generate_training_data(n_samples=200)
    print(f"  Training samples: {len(train_data)}")
    print(f"  Eval samples: {len(eval_data)}")

    # Show some examples
    print("\n  Example requests:")
    for tmpl in REQUEST_TEMPLATES[:3]:
        print(f"    '{tmpl['request']}'")
        for task in tmpl["plan"]:
            print(f"      -> {task['task']} [{task['model']}] deps={task['dep']}")

    # Train
    print("\n=== Training Planner Model ===")
    planner = PlannerModel(vocab_size=actual_vocab).to(device)
    losses_total, losses_nt, losses_ml, losses_dep = train(
        planner, train_data, n_epochs=30, batch_size=16, lr=1e-3, device=device
    )

    # Evaluate
    print("\n=== Evaluation ===")
    eval_results = evaluate_planning(planner, eval_data, device=device)
    print(f"  N-tasks accuracy:   {eval_results['n_tasks_acc']:.3f}")
    print(f"  Model selection:    {eval_results['model_acc']:.3f}")
    print(f"  Dependency accuracy: {eval_results['dep_acc']:.3f}")

    exec_res = eval_results['execution_results']
    avg_cost = np.mean([r['cost'] for r in exec_res])
    avg_success = np.mean([r['success_rate'] for r in exec_res])
    completion_rate = np.mean([r['completed'] for r in exec_res])
    print(f"  Avg execution cost: {avg_cost:.2f}")
    print(f"  Avg task success:   {avg_success:.3f}")
    print(f"  Plan completion:    {completion_rate:.3f}")

    # Analyze by difficulty
    diff_costs = defaultdict(list)
    diff_success = defaultdict(list)
    for r in exec_res:
        diff_costs[r['difficulty']].append(r['cost'])
        diff_success[r['difficulty']].append(r['success_rate'])

    # ── Visualization ──

    # 1. Training losses
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(losses_total, color='blue')
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(losses_nt, color='red')
    axes[0, 1].set_title("N-Tasks Prediction Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(losses_ml, color='green')
    axes[1, 0].set_title("Model Selection Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(losses_dep, color='orange')
    axes[1, 1].set_title("Dependency Prediction Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("HuggingGPT Planner: Training Losses", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_losses.png", dpi=150)
    plt.close()

    # 2. Planning accuracy breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ['N-Tasks\nAccuracy', 'Model\nSelection', 'Dependency\nPrediction']
    values = [eval_results['n_tasks_acc'], eval_results['model_acc'], eval_results['dep_acc']]
    colors = ['red', 'green', 'orange']
    bars = axes[0].bar(metrics, values, color=colors, alpha=0.7)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Planning Accuracy Breakdown")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                     f'{v:.3f}', ha='center', fontweight='bold')

    # Execution by difficulty
    diff_levels = sorted(diff_costs.keys())
    costs_by_diff = [np.mean(diff_costs[d]) for d in diff_levels]
    success_by_diff = [np.mean(diff_success[d]) for d in diff_levels]

    x = np.arange(len(diff_levels))
    width = 0.35
    axes[1].bar(x - width/2, costs_by_diff, width, label='Avg Cost', color='blue', alpha=0.7)
    ax2 = axes[1].twinx()
    ax2.bar(x + width/2, success_by_diff, width, label='Success Rate', color='green', alpha=0.7)
    axes[1].set_xlabel("Difficulty Level")
    axes[1].set_ylabel("Avg Cost", color='blue')
    ax2.set_ylabel("Success Rate", color='green')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Level {d}' for d in diff_levels])
    axes[1].set_title("Execution Cost & Success by Difficulty")
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "planning_accuracy.png", dpi=150)
    plt.close()

    # 3. Model hub usage distribution
    model_usage = defaultdict(int)
    for d in train_data:
        for m in d["models"]:
            model_usage[MODEL_NAMES[m]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(model_usage.keys())
    counts = list(model_usage.values())
    sorted_idx = np.argsort(counts)[::-1]
    axes[0].barh([names[i] for i in sorted_idx], [counts[i] for i in sorted_idx],
                 color='teal', alpha=0.7)
    axes[0].set_xlabel("Usage Count in Training Data")
    axes[0].set_title("Model Hub Usage Distribution")
    axes[0].grid(True, alpha=0.3, axis='x')

    # Model accuracy vs cost scatter
    model_names_short = list(MODELS.keys())
    accuracies = [MODELS[m]["accuracy"] for m in model_names_short]
    costs = [MODELS[m]["cost"] for m in model_names_short]
    axes[1].scatter(costs, accuracies, s=100, c='purple', alpha=0.7)
    for i, name in enumerate(model_names_short):
        axes[1].annotate(name, (costs[i], accuracies[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].set_xlabel("Cost")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Model Hub: Cost vs Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "model_hub.png", dpi=150)
    plt.close()

    # 4. DAG execution example
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Show a 4-task plan example
    example = REQUEST_TEMPLATES[-1]
    plan = example["plan"]
    n = len(plan)

    # Draw nodes
    x_positions = [0.15, 0.35, 0.6, 0.8]
    y_base = 0.5

    for i, task in enumerate(plan):
        x = x_positions[i]
        color = plt.cm.Set2(i / n)
        rect = plt.Rectangle((x - 0.08, y_base - 0.12), 0.16, 0.24,
                              facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y_base + 0.05, f"Task {i+1}", fontsize=11, fontweight='bold',
                ha='center', va='center')
        ax.text(x, y_base - 0.02, task["task"], fontsize=8, ha='center', va='center')
        ax.text(x, y_base - 0.08, f"[{task['model']}]", fontsize=7, ha='center',
                va='center', fontstyle='italic', color='gray')

    # Draw edges
    for i, task in enumerate(plan):
        for dep in task["dep"]:
            ax.annotate('', xy=(x_positions[i] - 0.08, y_base),
                        xytext=(x_positions[dep] + 0.08, y_base),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.text(0.5, 0.88, f"Request: \"{example['request']}\"", fontsize=11,
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # Show execution order
    ax.text(0.5, 0.12, "Execution Order: Task1 → Task2 → Task3 || Task4   (parallel where possible)",
            fontsize=10, ha='center', va='center', color='purple',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.5))

    ax.set_title("HuggingGPT: Task Decomposition & DAG Execution", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(results_dir / "dag_example.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    steps = [
        ("1. User\nRequest", "Natural language\ninstruction\nfrom user\n\n\"Generate an image\nand describe it\"", 0.1, 'purple'),
        ("2. Task\nPlanning", "LLM decomposes\ninto subtasks\n\nSelect models\nfrom hub\nBuild DAG", 0.32, 'orange'),
        ("3. Model\nExecution", "Execute subtasks\nin dependency order\n\nHandle I/O types\nbetween models\nCollect results", 0.56, 'teal'),
        ("4. Response\nGeneration", "LLM summarizes\nall results into\ncoherent response\n\nGrounded in\nmodel outputs", 0.82, 'green'),
    ]

    for name, desc, x_pos, color in steps:
        ax.text(x_pos, 0.78, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.21, 0.44, 0.69]:
        ax.annotate('→', xy=(x, 0.55), fontsize=24, ha='center', va='center', color='gray')

    ax.set_title("HuggingGPT: LLM as Orchestrator for AI Models", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "hugginggpt_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
