"""
Ablation runner for the DICE-Net V2 pipeline.

Runs LOSO leave-one-subject-out cross-validation.
Available configs:
  Exp 3:  Dual-branch additive fusion, d=24, mean aggregation  [BEST]
  Exp 11: Dual-branch additive fusion, d=24, median aggregation [2nd]

Supports feature-level mixup augmentation (training only).

Usage:
    python -m pipeline.ablation [--experiments 3,11]
"""
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader

from . import config as cfg
from .data_loading import build_label_map
from .dataset import EEGWindowDataset, build_fold_datasets
from .evaluate import compute_metrics
from .features import load_cached_features
from .model import DICENetV2
from .utils import EarlyStopping, get_class_weights, get_device, set_seed


@dataclass
class ExperimentConfig:
    """Configuration for one ablation experiment."""
    name: str
    # Model architecture
    d_model: int = 24
    nhead: int = 2
    dim_ff: int = 48
    dropout: float = 0.3
    classifier_dropout: float = 0.5
    # Training
    lr: float = 5e-4
    weight_decay: float = 5e-2
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 32
    # Mixup
    mixup_alpha: float = 0.2       # 0 = disabled
    # Aggregation
    aggregation: str = "mean"      # 'mean' or 'median'


def build_model(ec: ExperimentConfig, device: torch.device) -> nn.Module:
    """Instantiate DICENetV2 from experiment config."""
    model = DICENetV2(
        d_model=ec.d_model, nhead=ec.nhead, dim_ff=ec.dim_ff,
        dropout=ec.dropout, classifier_dropout=ec.classifier_dropout,
    )
    return model.to(device)


# -- Mixup helpers --

def mixup_batch(rbp, scc, labels, alpha=0.2):
    """Apply feature-level mixup to a training batch.

    Draws lam ~ Beta(alpha, alpha), shuffles the batch, and blends
    features and one-hot labels.
    """
    if alpha <= 0:
        y_onehot = F.one_hot(labels, num_classes=2).float()
        return rbp, scc, y_onehot

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5

    B = rbp.size(0)
    perm = torch.randperm(B, device=rbp.device)

    rbp_mix = lam * rbp + (1 - lam) * rbp[perm]
    scc_mix = lam * scc + (1 - lam) * scc[perm]

    y_onehot = F.one_hot(labels, num_classes=2).float()
    y_perm = y_onehot[perm]
    y_mix = lam * y_onehot + (1 - lam) * y_perm

    return rbp_mix, scc_mix, y_mix


def soft_cross_entropy(logits, soft_targets, weight=None):
    """Cross-entropy with soft (mixup) targets."""
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -(soft_targets * log_probs).sum(dim=1)

    if weight is not None:
        hard_labels = soft_targets.argmax(dim=1)
        sample_weights = weight[hard_labels]
        per_sample = per_sample * sample_weights

    return per_sample.mean()


# -- Training loop --

def train_one_epoch(model, loader, optimizer, criterion_weight, device,
                    grad_clip=1.0, mixup_alpha=0.0):
    """Train one epoch with optional mixup."""
    model.train()
    total_loss = 0.0
    n = 0
    for rbp, scc, labels, _ in loader:
        rbp, scc, labels = rbp.to(device), scc.to(device), labels.to(device)

        # Mixup (training-only augmentation)
        rbp_in, scc_in, y_soft = mixup_batch(rbp, scc, labels, mixup_alpha)

        optimizer.zero_grad()
        logits = model(rbp_in, scc_in)
        loss = soft_cross_entropy(logits, y_soft, weight=criterion_weight)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_windows(model, loader, criterion, device):
    """Validate without mixup. Uses standard CE loss."""
    model.eval()
    all_probs, all_labels, all_sids = [], [], []
    total_loss = 0.0
    n = 0
    for rbp, scc, labels, sids in loader:
        rbp, scc, labels = rbp.to(device), scc.to(device), labels.to(device)
        logits = model(rbp, scc)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item()
        n += 1
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_sids.append(sids.cpu().numpy())
    return (
        total_loss / max(n, 1),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_sids),
    )


def aggregate_predictions(probs, sids, labels, method="mean"):
    """Subject-level aggregation. Returns (sid_list, pred, label, p_ad)."""
    unique_sids = sorted(set(sids))
    out_sids, out_preds, out_labels, out_probs = [], [], [], []
    for sid in unique_sids:
        mask = sids == sid
        s_probs = probs[mask]  # (W, 2)
        if method == "median":
            p_ad = float(np.median(s_probs[:, 1]))
        else:
            p_ad = float(s_probs[:, 1].mean())
        out_sids.append(sid)
        out_preds.append(1 if p_ad >= 0.5 else 0)
        out_labels.append(int(labels[mask][0]))
        out_probs.append(p_ad)
    return out_sids, np.array(out_preds), np.array(out_labels), np.array(out_probs)


def train_fold_v2(
    ec: ExperimentConfig,
    train_ds: EEGWindowDataset,
    val_ds: EEGWindowDataset,
    device: torch.device,
) -> dict:
    """Train one LOSO fold. Returns val predictions."""
    model = build_model(ec, device)

    train_labels = train_ds.labels.numpy()
    cw = get_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)  # for validation only

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ec.lr, weight_decay=ec.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ec.max_epochs,
    )
    stopper = EarlyStopping(patience=ec.patience)

    train_loader = DataLoader(train_ds, batch_size=ec.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ec.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, ec.max_epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, cw, device,
            mixup_alpha=ec.mixup_alpha,
        )
        val_loss, _, _, _ = validate_windows(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stopper(val_loss):
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    _, val_probs, val_labels, val_sids = validate_windows(
        model, val_loader, criterion, device
    )
    return {"probs": val_probs, "labels": val_labels, "sids": val_sids}


def run_loso_experiment(
    ec: ExperimentConfig,
    features: Dict[int, dict],
    seed: int = cfg.SEED,
) -> dict:
    """Full LOSO evaluation for one experiment config."""
    set_seed(seed)
    device = get_device()

    sids = np.array(sorted(features.keys()))
    labels = np.array([features[s]["label"] for s in sids])
    logo = LeaveOneGroupOut()

    all_sids, all_preds, all_labels_out, all_probs = [], [], [], []
    n_total = len(sids)

    for fold, (train_idx, val_idx) in enumerate(
        logo.split(np.zeros(len(sids)), labels, groups=sids), 1
    ):
        train_sids = sids[train_idx].tolist()
        val_sids = sids[val_idx].tolist()
        val_sid = val_sids[0]

        t0 = time.time()
        train_ds, val_ds = build_fold_datasets(features, train_sids, val_sids)
        result = train_fold_v2(ec, train_ds, val_ds, device)

        s_ids, s_preds, s_labels, s_probs = aggregate_predictions(
            result["probs"], result["sids"], result["labels"],
            method=ec.aggregation,
        )
        all_sids.extend(s_ids)
        all_preds.extend(s_preds)
        all_labels_out.extend(s_labels)
        all_probs.extend(s_probs)

        elapsed = time.time() - t0
        correct = "+" if s_preds[0] == s_labels[0] else "X"
        actual = cfg.INT_TO_LABEL[s_labels[0]]
        predicted = cfg.INT_TO_LABEL[s_preds[0]]
        print(f"  [{fold:2d}/{n_total}] Subj {val_sid:>3} ({actual}) -> "
              f"{predicted} P(AD)={s_probs[0]:.3f} {correct} [{elapsed:.1f}s]")

    all_preds = np.array(all_preds)
    all_labels_out = np.array(all_labels_out)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_preds, all_labels_out, all_probs)
    return {
        "metrics": metrics,
        "sids": all_sids,
        "preds": all_preds,
        "labels": all_labels_out,
        "probs": all_probs,
    }


def run_multi_seed(
    ec: ExperimentConfig,
    features: Dict[int, dict],
    seeds: List[int] = None,
) -> dict:
    """Run LOSO with multiple seeds and average predictions."""
    if seeds is None:
        seeds = [42, 123, 7]

    sids_sorted = sorted(features.keys())
    n = len(sids_sorted)
    all_probs = np.zeros((len(seeds), n))

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{len(seeds)}) ---")
        result = run_loso_experiment(ec, features, seed=seed)
        sid_to_idx = {s: j for j, s in enumerate(sids_sorted)}
        for j, sid in enumerate(result["sids"]):
            all_probs[i, sid_to_idx[sid]] = result["probs"][j]

    avg_probs = all_probs.mean(axis=0)
    labels = np.array([features[s]["label"] for s in sids_sorted])
    preds = (avg_probs >= 0.5).astype(int)

    metrics = compute_metrics(preds, labels, avg_probs)
    return {
        "metrics": metrics,
        "sids": sids_sorted,
        "preds": preds,
        "labels": labels,
        "probs": avg_probs,
        "per_seed_probs": all_probs,
    }


def print_experiment_result(name: str, result: dict):
    """Print compact summary of one experiment."""
    m = result["metrics"]
    n_correct = (result["preds"] == result["labels"]).sum()
    n_total = len(result["preds"])
    print(f"\n{'='*60}")
    print(f"RESULT: {name}")
    print(f"  Accuracy:     {n_correct}/{n_total} = {m['accuracy']:.1%}")
    print(f"  Balanced Acc: {m['balanced_accuracy']:.1%}")
    print(f"  Macro F1:     {m['macro_f1']:.3f}")
    print(f"  ROC-AUC:      {m['roc_auc']:.3f}")
    cm = m["confusion_matrix"]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  Sensitivity:  {tp}/{tp+fn} = {m['sensitivity']:.1%}")
        print(f"  Specificity:  {tn}/{tn+fp} = {m['specificity']:.1%}")

    missed = []
    for i, (s, p, l) in enumerate(zip(result["sids"], result["preds"], result["labels"])):
        if p != l:
            missed.append(f"{s}({cfg.INT_TO_LABEL[l]}->{cfg.INT_TO_LABEL[p]})")
    if missed:
        print(f"  Misclassified: {', '.join(missed)}")
    print(f"{'='*60}")


# -- Experiment definitions --

EXPERIMENTS = {
    3: ExperimentConfig(
        name="3. Dual V2 additive (d=24) + mixup 0.2 -- mean agg",
        d_model=24, nhead=2, dim_ff=48,
        mixup_alpha=0.2,
    ),
    11: ExperimentConfig(
        name="11. Dual V2 additive (d=24) + mixup 0.2 -- median agg",
        d_model=24, nhead=2, dim_ff=48,
        mixup_alpha=0.2,
        aggregation="median",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated experiment numbers (e.g. '3,11'). Default: all.",
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Run 3-seed averaging on best config.",
    )
    args = parser.parse_args()

    # Load cached features
    label_map = build_label_map()
    train_sids = sorted(label_map.keys())
    features = load_cached_features(train_sids)
    n_ad = sum(1 for f in features.values() if f["label"] == 1)
    n_cn = sum(1 for f in features.values() if f["label"] == 0)
    print(f"Loaded {len(features)} subjects (AD={n_ad}, CN={n_cn})")

    if args.experiments:
        exp_nums = [int(x.strip()) for x in args.experiments.split(",")]
    else:
        exp_nums = sorted(EXPERIMENTS.keys())

    results = {}
    for num in exp_nums:
        if num not in EXPERIMENTS:
            print(f"Unknown experiment {num}, skipping")
            continue
        ec = EXPERIMENTS[num]
        print(f"\n{'#'*60}")
        print(f"# Experiment {ec.name}")
        print(f"# d_model={ec.d_model}, nhead={ec.nhead}, ff={ec.dim_ff}")
        print(f"# dropout={ec.dropout}/{ec.classifier_dropout}, "
              f"mixup_alpha={ec.mixup_alpha}, agg={ec.aggregation}")
        nparams = sum(p.numel() for p in build_model(ec, torch.device("cpu")).parameters())
        print(f"# Params: {nparams:,}")
        print(f"{'#'*60}")

        result = run_loso_experiment(ec, features)
        print_experiment_result(ec.name, result)
        results[num] = result

    if args.multi_seed and results:
        best_num = max(results, key=lambda k: results[k]["metrics"]["balanced_accuracy"])
        best_ec = EXPERIMENTS[best_num]
        print(f"\n{'#'*60}")
        print(f"# 3-seed averaging on best config: Exp {best_num}")
        print(f"{'#'*60}")
        ms_result = run_multi_seed(best_ec, features, seeds=[42, 123, 7])
        print_experiment_result(f"3-seed avg of Exp {best_num}", ms_result)

    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'Exp':<5} {'Name':<50} {'Acc':>5} {'BAcc':>5} "
              f"{'AUC':>5} {'F1':>5} {'Sens':>5} {'Spec':>5}")
        print("-" * 80)
        for num in sorted(results.keys()):
            m = results[num]["metrics"]
            n_c = (results[num]["preds"] == results[num]["labels"]).sum()
            n_t = len(results[num]["preds"])
            print(f"{num:<5} {EXPERIMENTS[num].name:<50} "
                  f"{n_c}/{n_t:>2}  "
                  f"{m['balanced_accuracy']:.1%} "
                  f"{m['roc_auc']:.3f} "
                  f"{m['macro_f1']:.3f} "
                  f"{m['sensitivity']:.1%} "
                  f"{m['specificity']:.1%}")
        print("=" * 80)


if __name__ == "__main__":
    main()
