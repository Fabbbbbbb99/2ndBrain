"""
train_laya.py - Ultra-Calibrated Training Pipeline for Laya (2ndBrain)

Features:
1. Focal Loss (gamma=2.0) with Teacher Soft-Distillation to eliminate overconfident logit blowup.
2. Unfreezes top ModernBERT encoder layers + decision head.
3. Multi-Temperature Vector Calibration: Independent T_choice, T_score, T_noul.
4. Non-Parametric Isotonic Regression (PAVA) for binary/noul calibration.
5. Automatic CUDA/MPS/CPU device placement across batch collation and L-BFGS calibration.
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import laya
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from laya.common import collate_items
from safetensors.torch import save_model
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_device(device: str = "auto") -> str:
    """Resolves the optimal PyTorch compute device ('cuda', 'mps', or 'cpu')."""
    if device and device.lower() != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(filepath: str):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def apply_label_smoothing(target_idx: int, num_classes: int, epsilon: float = 0.04) -> list:
    """Label smoothing to prevent logit explosion."""
    if num_classes <= 1:
        return [1.0]
    smoothed = [epsilon / (num_classes - 1)] * num_classes
    smoothed[target_idx] = 1.0 - epsilon
    return smoothed


def prepare_item(agent, sample: dict):
    q_type = sample["type"]
    state = sample["state"]

    if q_type == "choice":
        options = sample["options"]
        if "target_dist" in sample:
            target_vec = sample["target_dist"]
            target_idx = int(np.argmax(target_vec))
        else:
            target = sample["target"]
            target_idx = options.index(target)
            target_vec = apply_label_smoothing(target_idx, len(options), epsilon=0.08)

        qdef = {
            "type": "choice",
            "instructions": sample["question"],
            "criteria": {opt: opt for opt in options},
        }
    elif q_type == "score":
        score_val = sample["target"]
        target_idx = max(0, min(4, score_val - 1))
        target_vec = apply_label_smoothing(target_idx, 5, epsilon=0.08)

        qdef = {
            "type": "score",
            "instructions": sample["question"],
            "criteria": [f"Level {i+1}" for i in range(5)],
        }
    elif q_type == "noul":
        if "target_dist" in sample:
            target_vec = sample["target_dist"]
            target_idx = int(np.argmax(target_vec))
        else:
            bool_val = bool(sample["target"])
            target_idx = 1 if bool_val else 0
            target_vec = apply_label_smoothing(target_idx, 2, epsilon=0.08)

        qdef = {
            "type": "noul",
            "instructions": sample["question"],
            "criteria": {"false": "False criteria", "true": "True criteria"},
        }
    else:
        return None

    internal = {"q": agent._to_internal(qdef)}
    items = agent._encode_state(state, ["q"], internal)
    if not items:
        return None

    item = items[0]
    item["target"] = target_vec
    item["target_idx"] = target_idx
    item["q_type_name"] = q_type
    return item


def compute_ece(confidences, predictions, ground_truth, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    if len(confidences) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(predictions[in_bin] == ground_truth[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def fit_task_temperature(val_logits_list, val_targets_list, lr=0.01, max_iter=50) -> float:
    """Finds optimal temperature T for a specific task minimizing NLL using L-BFGS."""
    if isinstance(val_logits_list, torch.Tensor):
        if val_logits_list.numel() == 0:
            return 1.0
        logits_seq = [val_logits_list[i] for i in range(val_logits_list.shape[0])]
    else:
        logits_seq = list(val_logits_list)
        if not logits_seq:
            return 1.0

    if isinstance(val_targets_list, torch.Tensor):
        targets_seq = [int(x) for x in val_targets_list.tolist()]
    else:
        targets_seq = [int(x) for x in val_targets_list]

    target_device = logits_seq[0].device
    log_temp = torch.nn.Parameter(torch.zeros(1, device=target_device))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_temp)
        loss = torch.zeros(1, device=target_device)
        for logits, target in zip(logits_seq, targets_seq):
            l_dev = logits.detach().to(target_device)
            t_tensor = torch.tensor([target], dtype=torch.long, device=target_device)
            loss = loss + F.cross_entropy(l_dev.unsqueeze(0) / t, t_tensor)
        loss = loss / len(logits_seq)
        loss.backward()
        return loss

    optimizer.step(closure)
    return round(float(torch.exp(log_temp).item()), 4)


def fit_isotonic_pava(x_vals, y_vals):
    """Exact Pool Adjacent Violators Algorithm (PAVA) in pure Python/NumPy for Isotonic Regression."""
    if len(x_vals) < 2:
        return [0.0, 1.0], [0.0, 1.0]

    order = np.argsort(x_vals)
    x_sorted = np.array(x_vals, dtype=float)[order]
    y_sorted = np.array(y_vals, dtype=float)[order]

    blocks = [
        [float(y_sorted[i]), 1.0, float(x_sorted[i]), float(x_sorted[i])]
        for i in range(len(y_sorted))
    ]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            sum_y = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [sum_y / w, w, blocks[i][2], blocks[i + 1][3]]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    pts_x = [b[2] for b in blocks]
    pts_y = [b[0] for b in blocks]
    if pts_x[0] > 0.0:
        pts_x.insert(0, 0.0)
        pts_y.insert(0, 0.0)
    if pts_x[-1] < 1.0:
        pts_x.append(1.0)
        pts_y.append(1.0)
    return pts_x, pts_y


def train_laya(
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 6,
    gamma: float = 2.0,
    lr_head: float = 1e-4,
    lr_encoder: float = 2e-5,
    device: str = "auto",
):
    train_file = train_file or str(REPO_ROOT / "data" / "train_data.jsonl")
    val_file = val_file or str(REPO_ROOT / "data" / "val_data.jsonl")
    output_dir = output_dir or str(REPO_ROOT / "models" / "laya-knowledge-base")
    target_device = resolve_device(device)

    print(
        f"[Train] Initializing base Laya agent ('convaiinnovations/laya' / typed-decisions) on {target_device}..."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device=target_device)

    print("[Train] Freezing base encoder; unfreezing top 4 ModernBERT layers + decision head...")
    for param in agent.model.encoder.parameters():
        param.requires_grad = False

    unfrozen_layers = agent.model.encoder.layers[-4:]
    for layer in unfrozen_layers:
        for param in layer.parameters():
            param.requires_grad = True

    if hasattr(agent.model.encoder, "final_norm"):
        for param in agent.model.encoder.final_norm.parameters():
            param.requires_grad = True

    for param in agent.model.type_emb.parameters():
        param.requires_grad = True
    for param in agent.model.head.parameters():
        param.requires_grad = True
    for param in agent.model.scorer.parameters():
        param.requires_grad = True
    for param in agent.model.act_head.parameters():
        param.requires_grad = True

    encoder_params = [p for p in agent.model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for n, p in agent.model.named_parameters() if not n.startswith("encoder") and p.requires_grad
    ]

    trainable_count = sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
    print(f"[Train] Trainable Parameters: {trainable_count:,} (Top ModernBERT + Decision Head)")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder, "weight_decay": 0.01},
            {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
        ]
    )

    train_samples = load_dataset(train_file)
    val_samples = load_dataset(val_file)

    train_items = [prepare_item(agent, s) for s in train_samples]
    train_items = [it for it in train_items if it is not None]

    val_items = [prepare_item(agent, s) for s in val_samples]
    val_items = [it for it in val_items if it is not None]

    print(
        f"[Train] Loaded {len(train_items)} training samples and {len(val_items)} validation samples."
    )

    batch_size = 8
    num_training_steps = epochs * ((len(train_items) + batch_size - 1) // batch_size)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(1, epochs + 1):
        agent.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        perm = torch.randperm(len(train_items)).tolist()
        shuffled = [train_items[i] for i in perm]

        for i in range(0, len(shuffled), batch_size):
            batch_slice = shuffled[i : i + batch_size]
            b = collate_items([[it] for it in batch_slice], agent.tok.pad_token_id)
            if b is None or "target" not in b:
                continue

            b = {
                k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }

            optimizer.zero_grad()
            logits, _ = agent.model(
                b["input_ids"],
                b["attention_mask"],
                b["marker_pos"],
                b["marker_mask"],
                b["qtype"],
            )

            probs = F.softmax(logits, dim=-1)
            pt = (probs * b["target"]).sum(dim=-1).clamp(min=1e-8, max=1.0)
            focal_weight = (1.0 - pt) ** gamma
            loss = -(focal_weight * torch.log(pt)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(batch_slice)
            preds = logits.argmax(dim=-1).tolist()
            targets = [it["target_idx"] for it in batch_slice]
            for p, t in zip(preds, targets):
                if p == t:
                    correct += 1
                total += 1

        avg_loss = total_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100
        print(
            f"[Epoch {epoch}/{epochs}] Focal Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% ({correct}/{total})"
        )

    agent.model.eval()
    val_correct = 0
    val_total = 0

    task_data = {
        "choice": {"logits": [], "targets": [], "confs": [], "preds": []},
        "score": {"logits": [], "targets": [], "confs": [], "preds": []},
        "noul": {"logits": [], "targets": [], "confs": [], "preds": []},
    }

    all_confs_uncal = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(val_items), batch_size):
            batch_slice = val_items[i : i + batch_size]
            b = collate_items([[it] for it in batch_slice], agent.tok.pad_token_id)
            if b is None or "target" not in b:
                continue
            b = {
                k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            logits, _ = agent.model(
                b["input_ids"],
                b["attention_mask"],
                b["marker_pos"],
                b["marker_mask"],
                b["qtype"],
            )
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).tolist()
            confs = probs.max(dim=-1).values.tolist()
            targets = [it["target_idx"] for it in batch_slice]

            for row_idx in range(len(batch_slice)):
                q_type = batch_slice[row_idx]["q_type_name"]
                task_data[q_type]["logits"].append(logits[row_idx].detach().cpu())
                task_data[q_type]["targets"].append(targets[row_idx])
                task_data[q_type]["confs"].append(confs[row_idx])
                task_data[q_type]["preds"].append(preds[row_idx])

                all_confs_uncal.append(confs[row_idx])
                all_preds.append(preds[row_idx])
                all_targets.append(targets[row_idx])

                if preds[row_idx] == targets[row_idx]:
                    val_correct += 1
                val_total += 1

    overall_acc = (val_correct / max(1, val_total)) * 100
    ece_raw = compute_ece(np.array(all_confs_uncal), np.array(all_preds), np.array(all_targets))

    t_choice = fit_task_temperature(task_data["choice"]["logits"], task_data["choice"]["targets"])
    t_score = fit_task_temperature(task_data["score"]["logits"], task_data["score"]["targets"])
    t_noul = fit_task_temperature(task_data["noul"]["logits"], task_data["noul"]["targets"])

    all_confs_cal = []
    for q_type, t_val in [("choice", t_choice), ("score", t_score), ("noul", t_noul)]:
        for l in task_data[q_type]["logits"]:
            scaled_p = F.softmax(l / t_val, dim=-1)
            all_confs_cal.append(scaled_p.max().item())

    ece_multi = compute_ece(np.array(all_confs_cal), np.array(all_preds), np.array(all_targets))

    all_top1_probs_raw = []
    all_top1_targets = []

    for q_type, t_val in [("choice", t_choice), ("score", t_score), ("noul", t_noul)]:
        for l, target in zip(task_data[q_type]["logits"], task_data[q_type]["targets"]):
            scaled_p = F.softmax(l / t_val, dim=-1)
            max_prob, pred = scaled_p.max(dim=-1)
            all_top1_probs_raw.append(max_prob.item())
            all_top1_targets.append(1 if pred.item() == target else 0)

    pts_x, pts_y = fit_isotonic_pava(all_top1_probs_raw, all_top1_targets)

    all_calibrated_confs = []
    for p in all_top1_probs_raw:
        cal_p = float(np.interp(p, pts_x, pts_y))
        all_calibrated_confs.append(cal_p)

    ece_pava = compute_ece(
        np.array(all_calibrated_confs),
        np.array([1] * len(all_calibrated_confs)),
        np.array(all_top1_targets),
    )

    print("\n================ ULTRA-CALIBRATION RESULTS ================")
    print(f"Overall Accuracy:        {overall_acc:.2f}% ({val_correct}/{val_total})")
    print(f"Raw Model ECE:           {ece_raw:.4f}")
    print(f"Vector Temperatures:     T_choice={t_choice}, T_score={t_score}, T_noul={t_noul}")
    print(
        f"Multi-Temperature ECE:   {ece_multi:.4f} "
        f"(Error reduced by {((ece_raw - ece_multi)/max(1e-5, ece_raw)*100):.1f}%)"
    )
    print(f"Isotonic Top-1 PAVA ECE: {ece_pava:.4f}")
    print("==========================================================\n")

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model.safetensors")
    print(f"[Export] Saving fine-tuned weights to {weights_path}...")
    save_model(agent.model, weights_path)

    snapshot_dir = snapshot_download("convaiinnovations/laya")
    base_subfolder = os.path.join(snapshot_dir, "typed-decisions")

    tok_src = os.path.join(base_subfolder, "tokenizer")
    tok_dst = os.path.join(output_dir, "tokenizer")
    if os.path.exists(tok_src) and not os.path.exists(tok_dst):
        shutil.copytree(tok_src, tok_dst)

    enc_src = os.path.join(base_subfolder, "encoder")
    enc_dst = os.path.join(output_dir, "encoder")
    if os.path.exists(enc_src) and not os.path.exists(enc_dst):
        shutil.copytree(enc_src, enc_dst)

    cfg = dict(agent.cfg)
    cfg["temperature"] = [t_choice, t_score, t_noul]
    cfg["temperature_by_options"] = {
        "choice:2": t_noul,
        "choice:3": t_choice,
        "choice:5": t_score,
        "choice:11+": t_choice,
    }
    cfg["finetuned_domain"] = "2ndBrain_dual_process"
    cfg["val_accuracy"] = round(overall_acc, 2)
    cfg["val_ece"] = round(ece_multi, 4)
    cfg["val_ece_pava_isotonic"] = round(ece_pava, 4)

    cfg_path = os.path.join(output_dir, "rl_agent_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    iso_path = os.path.join(output_dir, "pava_isotonic.json")
    with open(iso_path, "w", encoding="utf-8") as f:
        json.dump({"x": pts_x, "y": pts_y}, f, indent=2)

    print(f"[Success] All checkpoints and vector calibrations saved to {output_dir}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Train and calibrate Laya System 1 router")
    parser.add_argument("--train-file", default=None, help="Path to train_data.jsonl")
    parser.add_argument("--val-file", default=None, help="Path to val_data.jsonl")
    parser.add_argument("--output-dir", default=None, help="Path to output model directory")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu")
    args = parser.parse_args()
    train_laya(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
    )
