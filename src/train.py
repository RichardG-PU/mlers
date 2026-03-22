"""
LOOCV training loop for AD vs CN classification.

For each held-out subject:
  1. Train DICENet on all other subjects (with augmentation).
  2. Early-stop on held-out subject's epoch-level loss.
  3. Load best checkpoint, aggregate epoch probabilities → subject prediction.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.config as cfg
from src.dataset import EEGDataset
from src.model import DICENet


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.DEVICE.type == "mps":
        torch.mps.manual_seed(seed)
    elif cfg.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _compute_pos_weight(subject_ids: list[str], manifest: dict) -> torch.Tensor:
    """pos_weight = n_CN_epochs / n_AD_epochs on the training fold."""
    n_ad = sum(manifest[s]["n_epochs"] for s in subject_ids if manifest[s]["label"] == "A")
    n_cn = sum(manifest[s]["n_epochs"] for s in subject_ids if manifest[s]["label"] == "C")
    weight = n_cn / max(n_ad, 1)
    return torch.tensor([weight], dtype=torch.float32)


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for rbp, scc, labels in loader:
        rbp    = rbp.to(device)
        scc    = scc.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(rbp, scc)
        # Label smoothing
        eps = cfg.LABEL_SMOOTH
        labels = labels * (1 - eps) + (1 - labels) * eps
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for rbp, scc, labels in loader:
        rbp    = rbp.to(device)
        scc    = scc.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        logits = model(rbp, scc)
        total_loss += criterion(logits, labels).item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _subject_predict(model, loader, device) -> tuple[int, float]:
    """Mean-probability aggregation across all epochs of one subject."""
    model.eval()
    probs = []
    for rbp, scc, _ in loader:
        logits = model(rbp.to(device), scc.to(device))
        p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
    mean_prob = float(np.mean(probs))
    pred_label = int(mean_prob >= 0.5)
    return pred_label, mean_prob


def run_loocv(manifest: dict) -> list[tuple]:
    """
    Run leave-one-out cross-validation over all subjects in manifest.

    Returns:
        List of (subject_id, true_int_label, pred_int_label, mean_prob).
    """
    set_seed()

    all_subjects = list(manifest.keys())
    label_map    = {"A": 1, "C": 0}
    device       = cfg.DEVICE

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold_idx, val_sid in enumerate(all_subjects):
        ckpt_path = cfg.CKPT_DIR / f"best_fold_{fold_idx}.pt"
        train_sids = [s for s in all_subjects if s != val_sid]
        true_label = label_map[manifest[val_sid]["label"]]

        print(f"\nFold {fold_idx+1:>2}/{len(all_subjects)}  "
              f"val={val_sid} ({manifest[val_sid]['label']})  "
              f"train={len(train_sids)} subjects")

        # Datasets + loaders
        train_ds = EEGDataset(train_sids, manifest, augment=True)
        val_ds   = EEGDataset([val_sid],  manifest, augment=False)

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True,  drop_last=False, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                                  shuffle=False, drop_last=False, num_workers=0)

        # Model + optimiser + scheduler
        set_seed(cfg.SEED + fold_idx)
        model = DICENet().to(device)
        pos_weight = _compute_pos_weight(train_sids, manifest).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.AdamW(model.parameters(),
                                       lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.N_EPOCHS, eta_min=1e-5
        )

        best_val_loss    = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.N_EPOCHS + 1):
            train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss   = _eval_loss(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                print(f"  ep {epoch:>3}  train={train_loss:.4f}  val={val_loss:.4f}"
                      f"  patience={patience_counter}/{cfg.PATIENCE}")

            if patience_counter >= cfg.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        # Load best weights → subject-level prediction
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        pred_label, mean_prob = _subject_predict(model, val_loader, device)

        fold_results.append((val_sid, true_label, pred_label, mean_prob))
        print(f"  → pred={pred_label}  true={true_label}  P(AD)={mean_prob:.3f}"
              f"  {'✓' if pred_label == true_label else '✗'}")

    return fold_results


def train_final_model(manifest: dict, n_epochs: int | None = None, seed: int = cfg.SEED) -> str:
    """
    Train a single model on ALL labeled subjects (no held-out).

    Args:
        manifest:  Dict from manifest.json.
        n_epochs:  Training epochs.  If None, uses cfg.N_EPOCHS.
        seed:      Random seed for reproducibility.

    Returns:
        Path to the saved checkpoint.
    """
    set_seed(seed)
    device = cfg.DEVICE
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.CKPT_DIR / f"final_model_seed{seed}.pt"

    all_sids = list(manifest.keys())
    n_train = n_epochs if n_epochs is not None else cfg.N_EPOCHS

    print(f"\nTraining final model on all {len(all_sids)} subjects for {n_train} epochs...")

    train_ds = EEGDataset(all_sids, manifest, augment=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=0)

    model = DICENet().to(device)
    pos_weight = _compute_pos_weight(all_sids, manifest).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_train, eta_min=1e-5
    )

    for epoch in range(1, n_train + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  ep {epoch:>3}  train={train_loss:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"  Final model saved → {ckpt_path}")
    return str(ckpt_path)
