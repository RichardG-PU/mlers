"""
Multi-seed DICE-net ensemble: LOOCV evaluation + final model training + test predictions.

Usage:
    python scripts/predict.py

Prerequisites:
    python scripts/precompute_features.py   (training cache must exist)
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import load_manifest
from src.train import run_loocv, train_final_model
from src.inference import precompute_test_features, predict_subjects
from src.evaluate import print_results_table
import src.config as cfg

SEEDS = [42, 43, 44]  # Multi-seed ensemble
OUTPUT_DIR = cfg.ROOT / "output"


def save_loocv_csv(fold_results: list[tuple], path: Path) -> None:
    """Save LOOCV per-subject results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "true_label", "pred_label", "probability"])
        for sid, true_lbl, pred_lbl, prob in fold_results:
            true_str = "AD" if true_lbl == 1 else "CN"
            pred_str = "AD" if pred_lbl == 1 else "CN"
            writer.writerow([sid, true_str, pred_str, f"{prob:.4f}"])
    print(f"  LOOCV results saved → {path}")


if __name__ == "__main__":
    print(f"Device: {cfg.DEVICE}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Load manifest ──────────────────────────────────────
    manifest = load_manifest()
    n_ad = sum(1 for v in manifest.values() if v["label"] == "A")
    n_cn = sum(1 for v in manifest.values() if v["label"] == "C")
    print(f"Training manifest: {len(manifest)} subjects (AD={n_ad}, CN={n_cn})\n")

    # ── Phase 2: LOOCV evaluation ───────────────────────────────────
    print("=" * 60)
    print("PHASE 1: LOOCV Evaluation")
    print("=" * 60)
    fold_results = run_loocv(manifest)
    metrics = print_results_table(fold_results)

    # Save DICE-net LOOCV results to CSV (for meta-ensemble analysis)
    save_loocv_csv(fold_results, OUTPUT_DIR / "dicenet_loocv.csv")

    # ── Phase 3: Precompute test features ───────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Precomputing Test Features")
    print("=" * 60)
    test_manifest = precompute_test_features()

    if not test_manifest:
        print("No test subjects found. Exiting.")
        sys.exit(0)

    # ── Phase 4: Multi-seed final model training ────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 3: Multi-Seed Final Model Training ({len(SEEDS)} seeds)")
    print("=" * 60)

    # Benchmark first seed to decide seed count
    t0 = time.time()
    ckpt_0 = train_final_model(manifest, seed=SEEDS[0])
    t_one = time.time() - t0
    print(f"\n  Seed {SEEDS[0]} took {t_one:.0f}s")

    if t_one > 1800:  # >30 min → use 2 seeds
        SEEDS_ACTUAL = SEEDS[:2]
        print(f"  >30 min per seed → using 2 seeds: {SEEDS_ACTUAL}")
    elif t_one < 900:  # <15 min → use 5 seeds
        SEEDS_ACTUAL = [42, 43, 44, 45, 46]
        print(f"  <15 min per seed → using 5 seeds: {SEEDS_ACTUAL}")
    else:
        SEEDS_ACTUAL = SEEDS
        print(f"  Using 3 seeds: {SEEDS_ACTUAL}")

    # Train remaining seeds
    ckpt_paths = [ckpt_0]
    for seed in SEEDS_ACTUAL[1:]:
        ckpt = train_final_model(manifest, seed=seed)
        ckpt_paths.append(ckpt)

    # ── Phase 5: Multi-seed ensemble prediction ─────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 4: Ensemble Prediction ({len(ckpt_paths)} models)")
    print("=" * 60)

    # Collect predictions from each seed
    all_seed_results = []
    for i, ckpt in enumerate(ckpt_paths):
        results = predict_subjects(ckpt, test_manifest)
        all_seed_results.append(results)
        print(f"  Seed {SEEDS_ACTUAL[i]}: {len(results)} subjects predicted")

    # Average probabilities across seeds
    # all_seed_results[i] = [(sid, prob, label), ...]
    subject_ids = [r[0] for r in all_seed_results[0]]
    ensemble_probs = {}
    for sid in subject_ids:
        probs = []
        for seed_results in all_seed_results:
            for s, p, _ in seed_results:
                if s == sid:
                    probs.append(p)
                    break
        ensemble_probs[sid] = float(np.mean(probs))

    # Assertions (from eng review: inline assertions for ensemble)
    for sid, prob in ensemble_probs.items():
        assert 0.0 <= prob <= 1.0, f"Ensemble P(AD) for {sid} = {prob} outside [0,1]"

    # Print ensemble results
    print(f"\n{'Subject':>8}  {'P(AD)':>7}  {'Pred':>5}  {'Seeds':>6}")
    print("─" * 35)
    ensemble_results = []
    for sid in sorted(ensemble_probs.keys(), key=int):
        prob = ensemble_probs[sid]
        pred = "A" if prob >= 0.5 else "C"
        pred_str = "AD" if pred == "A" else "CN"
        n_seeds = len(ckpt_paths)
        print(f"{sid:>8}  {prob:>7.3f}  {pred_str:>5}  {n_seeds:>6}")
        ensemble_results.append((sid, prob, pred))

    # Save ensemble predictions CSV
    out_path = cfg.ROOT / "predictions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anonymized_id", "predicted_label", "probability"])
        for sid, prob, label in ensemble_results:
            writer.writerow([sid, label, f"{prob:.4f}"])
    print(f"\nEnsemble predictions saved → {out_path}")

    # Also save per-seed predictions for transparency
    seed_csv = OUTPUT_DIR / "dicenet_test_per_seed.csv"
    with open(seed_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["subject"] + [f"seed_{s}" for s in SEEDS_ACTUAL] + ["ensemble_prob", "predicted"]
        writer.writerow(header)
        for sid in sorted(ensemble_probs.keys(), key=int):
            row = [sid]
            for seed_results in all_seed_results:
                for s, p, _ in seed_results:
                    if s == sid:
                        row.append(f"{p:.4f}")
                        break
            row.append(f"{ensemble_probs[sid]:.4f}")
            row.append("AD" if ensemble_probs[sid] >= 0.5 else "CN")
            writer.writerow(row)
    print(f"Per-seed predictions saved → {seed_csv}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  LOOCV results:      {OUTPUT_DIR / 'dicenet_loocv.csv'}")
    print(f"  Test predictions:   {out_path}")
    print(f"  Per-seed detail:    {seed_csv}")
    print(f"  Models: {len(ckpt_paths)} seeds × final model")
