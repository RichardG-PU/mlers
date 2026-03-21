"""
One-time feature precomputation.

For each subject in the chosen task (default: AD + CN):
  1. Load raw (19, T) recording.
  2. Z-score normalise per channel over the full recording.
  3. Segment into 30-second epochs.
  4. Call extract_features on each epoch.
  5. Stack and save as (n_epochs, N_WINDOWS, N_BANDS, N_CHANNELS) arrays.
  6. Write manifest.json with metadata.

Run via:  python scripts/precompute_features.py
"""

import json
import time

import numpy as np
import pandas as pd

import src.config as cfg
from src.features import segment_recording, extract_features


def precompute_all(label_filter: tuple[str, ...] = ("A", "C")) -> None:
    """
    Precompute and cache RBP + SCC features for all subjects whose label is in
    *label_filter*.

    Args:
        label_filter: Labels to include.  Default ("A","C") = AD vs CN task.
    """
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.LABEL_CSV)
    df = df[df["label"].isin(label_filter)].reset_index(drop=True)

    manifest: dict = {}

    for _, row in df.iterrows():
        subject_id = str(row["anonymized_id"])
        label      = row["label"]
        class_dir  = cfg.CLASS_DIR[label]
        npy_path   = cfg.DATA_ROOT / class_dir / f"{subject_id}.npy"

        if not npy_path.exists():
            print(f"  [WARN] {npy_path} not found — skipping subject {subject_id}")
            continue

        print(f"  Processing subject {subject_id:>4s} ({class_dir}) … ", end="", flush=True)
        t0 = time.time()

        # Load + normalise
        eeg = np.load(npy_path)                                   # (19, T)
        mu    = eeg.mean(axis=1, keepdims=True)
        sigma = eeg.std(axis=1,  keepdims=True)
        eeg_norm = (eeg - mu) / (sigma + 1e-8)

        # Epoch (with overlapping windows)
        epochs = segment_recording(eeg_norm, stride_sec=cfg.EPOCH_STRIDE_SEC)
        n_epochs = len(epochs)

        if n_epochs == 0:
            print(f"SKIP (recording too short: {eeg.shape[1]} samples)")
            continue

        # Extract features for each epoch
        all_rbp, all_scc = [], []
        for epoch in epochs:
            rbp, scc = extract_features(epoch, sfreq=cfg.SFREQ,
                                        target_time_steps=cfg.N_WINDOWS)
            all_rbp.append(rbp)
            all_scc.append(scc)

        rbp_stack = np.stack(all_rbp).astype(np.float32)  # (n_epochs,30,5,19)
        scc_stack = np.stack(all_scc).astype(np.float32)

        np.save(cfg.CACHE_DIR / f"{subject_id}_rbp.npy", rbp_stack)
        np.save(cfg.CACHE_DIR / f"{subject_id}_scc.npy", scc_stack)

        manifest[subject_id] = {
            "label":     label,
            "class_dir": class_dir,
            "n_epochs":  n_epochs,
            "norm_mu":   mu.flatten().tolist(),
            "norm_sigma": sigma.flatten().tolist(),
        }

        elapsed = time.time() - t0
        print(f"{n_epochs} epochs  ({elapsed:.1f}s)")

    with open(cfg.CACHE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} subjects cached → {cfg.CACHE_DIR}")


def load_manifest() -> dict:
    manifest_path = cfg.CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cache manifest not found at {manifest_path}.\n"
            "Run:  python scripts/precompute_features.py"
        )
    with open(manifest_path) as f:
        return json.load(f)
