"""
Inference utilities for predicting on unlabeled test data.

1.  precompute_test_features()  – same pipeline as training cache, no labels.
2.  predict_subjects()          – load a trained model, predict on test subjects.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import src.config as cfg
from src.features import segment_recording, extract_features
from src.model import DICENet


def precompute_test_features(
    test_dir: Path = cfg.TEST_DIR,
    cache_dir: Path = cfg.TEST_CACHE_DIR,
) -> dict:
    """
    Compute RBP + PLV features for every .npy file found under test_dir/{AD,CN,FTD}/.

    Returns:
        test_manifest: {subject_id: {class_dir, n_epochs, norm_mu, norm_sigma}}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}

    for class_dir_name in ("AD", "CN", "FTD"):
        folder = test_dir / class_dir_name
        if not folder.exists():
            continue
        for npy_path in sorted(folder.glob("*.npy")):
            sid = npy_path.stem
            print(f"  [test] Processing {class_dir_name}/{sid} … ", end="", flush=True)
            t0 = time.time()

            eeg = np.load(npy_path)
            mu = eeg.mean(axis=1, keepdims=True)
            sigma = eeg.std(axis=1, keepdims=True)
            eeg_norm = (eeg - mu) / (sigma + 1e-8)

            epochs = segment_recording(eeg_norm, stride_sec=cfg.EPOCH_STRIDE_SEC)
            n_epochs = len(epochs)
            if n_epochs == 0:
                print(f"SKIP (recording too short: {eeg.shape[1]} samples)")
                continue

            all_rbp, all_scc = [], []
            for epoch in epochs:
                rbp, scc = extract_features(epoch, sfreq=cfg.SFREQ,
                                            target_time_steps=cfg.N_WINDOWS)
                all_rbp.append(rbp)
                all_scc.append(scc)

            rbp_stack = np.stack(all_rbp).astype(np.float32)
            scc_stack = np.stack(all_scc).astype(np.float32)

            np.save(cache_dir / f"{sid}_rbp.npy", rbp_stack)
            np.save(cache_dir / f"{sid}_scc.npy", scc_stack)

            manifest[sid] = {
                "class_dir": class_dir_name,
                "n_epochs": n_epochs,
                "norm_mu": mu.flatten().tolist(),
                "norm_sigma": sigma.flatten().tolist(),
            }
            elapsed = time.time() - t0
            print(f"{n_epochs} epochs  ({elapsed:.1f}s)")

    manifest_path = cache_dir / "test_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  {len(manifest)} test subjects cached → {cache_dir}")
    return manifest


@torch.no_grad()
def predict_subjects(
    model_path: str | Path,
    test_manifest: dict,
    cache_dir: Path = cfg.TEST_CACHE_DIR,
) -> list[tuple[str, float, str]]:
    """
    Load a trained model and predict on all test subjects.

    Returns:
        List of (subject_id, probability, predicted_label_str).
    """
    device = cfg.DEVICE
    model = DICENet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []
    for sid, info in sorted(test_manifest.items(), key=lambda x: int(x[0])):
        rbp_all = np.load(cache_dir / f"{sid}_rbp.npy")
        scc_all = np.load(cache_dir / f"{sid}_scc.npy")

        probs = []
        for ep_idx in range(info["n_epochs"]):
            rbp_t = torch.from_numpy(rbp_all[ep_idx]).unsqueeze(0).to(device)
            scc_t = torch.from_numpy(scc_all[ep_idx]).unsqueeze(0).to(device)
            logit = model(rbp_t, scc_t)
            p = torch.sigmoid(logit).item()
            probs.append(p)

        mean_prob = float(np.mean(probs))
        pred_label = "A" if mean_prob >= 0.5 else "C"
        results.append((sid, mean_prob, pred_label))

    return results
