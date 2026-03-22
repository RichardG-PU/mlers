"""
Test inference with DICENet V2.

Trains on ALL 38 labeled subjects (128 Hz, mixup),
then predicts the 15 unseen CN test subjects.

Configs:
  Exp 3:  d=24, add, mean   + mixup(0.2)  [BEST]
  Exp 11: d=24, add, median + mixup(0.2)  [2nd]

Usage:
    python -m pipeline.inference_v2
"""
import csv as csv_mod
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as cfg
from .ablation import (
    ExperimentConfig,
    build_model,
    train_one_epoch,
)
from .data_loading import _find_subject_file, build_label_map
from .dataset import EEGWindowDataset
from .features import load_cached_features, precompute_subject_features
from .utils import get_class_weights, get_device, set_seed


# -- Test subject discovery --

def _get_all_csv_sids(csv_path: str = cfg.LABEL_CSV) -> set[int]:
    """Get ALL subject IDs from the CSV (any label: A, C, F)."""
    sids = set()
    with open(csv_path, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            sids.add(int(row["anonymized_id"]))
    return sids


def _find_test_subjects(data_root: str = cfg.DATA_ROOT) -> list[dict]:
    """Find test subjects: present in data folders but NOT in the CSV."""
    csv_sids = _get_all_csv_sids()
    test_subjects = []
    for folder, label_name in [("AD", "AD"), ("CN", "CN")]:
        d = os.path.join(data_root, folder)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".npy"):
                sid = int(fn.replace(".npy", ""))
                if sid not in csv_sids:
                    test_subjects.append({"sid": sid, "folder": label_name})
    return sorted(test_subjects, key=lambda x: x["sid"])


def _load_and_extract_test_subject(sid: int, data_root: str = cfg.DATA_ROOT) -> dict | None:
    """Load, preprocess (500->128 Hz downsample), window, extract features."""
    path = _find_subject_file(data_root, sid)
    if path is None:
        print(f"  WARNING: no .npy file for subject {sid}")
        return None
    data = np.load(path)
    if data.shape[0] != cfg.N_CHANNELS:
        print(f"  WARNING: subject {sid} has {data.shape[0]} channels, skipping")
        return None
    subj = {"data": data, "sid": sid, "label": -1}
    return precompute_subject_features(subj)


# -- Configs --

CONFIGS = {
    "Exp3_d24_add_mean": ExperimentConfig(
        name="Exp3: d=24, add, mean + mixup(0.2) [BEST]",
        d_model=24, nhead=2, dim_ff=48,
        mixup_alpha=0.2,
    ),
    "Exp11_d24_add_median": ExperimentConfig(
        name="Exp11: d=24, add, median + mixup(0.2) [2nd]",
        d_model=24, nhead=2, dim_ff=48,
        mixup_alpha=0.2,
        aggregation="median",
    ),
}


def build_full_dataset(features):
    """Stack all labeled subjects into one training dataset."""
    rbp_all, scc_all, labels_all, sids_all = [], [], [], []
    for sid in sorted(features.keys()):
        f = features[sid]
        n = f["n_windows"]
        rbp_all.append(f["rbp"])
        scc_all.append(f["scc"])
        labels_all.append(np.full(n, f["label"], dtype=np.int64))
        sids_all.append(np.full(n, sid, dtype=np.int64))
    return EEGWindowDataset(
        np.concatenate(rbp_all),
        np.concatenate(scc_all),
        np.concatenate(labels_all),
        np.concatenate(sids_all),
    )


def train_final_v2(ec, train_ds, device, seed=42):
    """Train a V2 model on all training data. Fixed epochs, no validation."""
    set_seed(seed)
    model = build_model(ec, device)

    train_labels = train_ds.labels.numpy()
    cw = get_class_weights(train_labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ec.lr, weight_decay=ec.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ec.max_epochs,
    )

    train_loader = DataLoader(train_ds, batch_size=ec.batch_size, shuffle=True)

    for epoch in range(1, ec.max_epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, cw, device,
            mixup_alpha=ec.mixup_alpha,
        )
        scheduler.step()

    return model


@torch.no_grad()
def predict_test_subjects(model, test_features, device, aggregation="mean"):
    """Predict test subjects. Returns list of (sid, p_ad, pred_label)."""
    model.eval()
    results = []
    for tf in test_features:
        sid = tf["sid"]
        rbp_t = torch.tensor(tf["rbp"], dtype=torch.float32)
        scc_t = torch.tensor(tf["scc"], dtype=torch.float32)

        all_p_ad = []
        bs = 64
        for i in range(0, len(rbp_t), bs):
            rbp_b = rbp_t[i:i + bs].to(device)
            scc_b = scc_t[i:i + bs].to(device)
            logits = model(rbp_b, scc_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_p_ad.append(probs[:, 1])

        p_ad_all = np.concatenate(all_p_ad)
        if aggregation == "median":
            p_ad = float(np.median(p_ad_all))
        else:
            p_ad = float(p_ad_all.mean())
        pred = 1 if p_ad >= 0.5 else 0
        results.append((sid, p_ad, pred))
    return results


def main():
    device = get_device()
    N_SEEDS = 5

    print("=" * 70)
    print("DICE-NET V2 -- TEST SET INFERENCE (128 Hz, mixup)")
    print("  Train on all 38 labeled subjects, predict 15 unseen CN subjects")
    print(f"  {N_SEEDS}-seed ensemble per config")
    print("=" * 70)

    # 1. Load training features
    print("\n[1] Loading training features...")
    label_map = build_label_map()
    train_sids = sorted(label_map.keys())
    features = load_cached_features(train_sids)
    n_ad = sum(1 for f in features.values() if f["label"] == 1)
    n_cn = sum(1 for f in features.values() if f["label"] == 0)
    print(f"    {len(features)} subjects (AD={n_ad}, CN={n_cn})")

    train_ds = build_full_dataset(features)
    print(f"    {len(train_ds)} total windows")

    # 2. Load test subjects
    print("\n[2] Loading test subjects...")
    test_info = _find_test_subjects()
    test_sids = [t["sid"] for t in test_info]
    folder_label = {t["sid"]: t["folder"] for t in test_info}
    print(f"    {len(test_info)} test subjects: {test_sids}")

    test_features = []
    for i, sid in enumerate(test_sids):
        t0 = time.time()
        result = _load_and_extract_test_subject(sid)
        if result is not None:
            test_features.append(result)
            print(f"    [{i+1}/{len(test_sids)}] Subject {sid}: "
                  f"{result['n_windows']} windows [{time.time()-t0:.1f}s]")
    print(f"    {len(test_features)} subjects extracted successfully")

    # 3. Run each config
    print("\n[3] Running inference for each config...")
    all_results = {}

    for config_name, ec in CONFIGS.items():
        print(f"\n{'-'*70}")
        print(f"  CONFIG: {ec.name}")
        print(f"  Params: {sum(p.numel() for p in build_model(ec, torch.device('cpu')).parameters()):,}")
        print(f"{'-'*70}")

        seeds = [42, 123, 7, 2024, 999][:N_SEEDS]
        seed_preds = {tf["sid"]: [] for tf in test_features}

        for si, seed in enumerate(seeds):
            t0 = time.time()
            model = train_final_v2(ec, train_ds, device, seed=seed)
            preds = predict_test_subjects(model, test_features, device, ec.aggregation)
            elapsed = time.time() - t0
            n_cn_pred = sum(1 for _, _, p in preds if p == 0)
            print(f"    Seed {seed}: {n_cn_pred}/{len(preds)} CN [{elapsed:.1f}s]")
            for sid, p_ad, _ in preds:
                seed_preds[sid].append(p_ad)

        config_results = []
        for tf in test_features:
            sid = tf["sid"]
            avg_p = np.mean(seed_preds[sid])
            pred = 1 if avg_p >= 0.5 else 0
            config_results.append({
                "sid": sid,
                "folder": folder_label[sid],
                "p_ad": avg_p,
                "pred": "AD" if pred == 1 else "CN",
                "correct": (pred == 0),
            })

        all_results[config_name] = config_results

        n_correct = sum(1 for r in config_results if r["correct"])
        print(f"\n  Per-subject results ({n_correct}/{len(config_results)} correct):")
        for r in config_results:
            status = "+" if r["correct"] else "X"
            print(f"    Subject {r['sid']:>3} (true={r['folder']}) -> "
                  f"{r['pred']}  P(AD)={r['p_ad']:.3f}  {status}")

    # 4. Summary
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON -- TEST SET (15 unseen CN subjects)")
    print(f"{'='*70}")
    print(f"\n{'Config':<30} {'Correct':>8} {'Accuracy':>10} {'Avg P(AD)':>10}")
    print("-" * 60)

    for config_name, results in all_results.items():
        n_correct = sum(1 for r in results if r["correct"])
        avg_p = np.mean([r["p_ad"] for r in results])
        print(f"{config_name:<30} {n_correct:>3}/{len(results):<4}  "
              f"{n_correct/len(results):>8.1%}    {avg_p:>8.3f}")

    print(f"\n{'Config':<30} {'Misclassified (predicted AD)'}")
    print("-" * 60)
    for config_name, results in all_results.items():
        wrong = [str(r["sid"]) for r in results if not r["correct"]]
        print(f"{config_name:<30} {', '.join(wrong) if wrong else 'None'}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
