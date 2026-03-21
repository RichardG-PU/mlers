"""
Train a final model on all labeled data and predict on unlabeled test subjects.

Usage:
    python scripts/predict.py

Prerequisites:
    python scripts/precompute_features.py   (training cache must exist)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import load_manifest
from src.train import run_loocv, train_final_model
from src.inference import precompute_test_features, predict_subjects
from src.evaluate import print_results_table
import src.config as cfg


if __name__ == "__main__":
    print(f"Device: {cfg.DEVICE}\n")

    # 1. Load training manifest
    manifest = load_manifest()
    print(f"Training manifest: {len(manifest)} subjects  "
          f"(AD={sum(1 for v in manifest.values() if v['label']=='A')}, "
          f"CN={sum(1 for v in manifest.values() if v['label']=='C')})\n")

    # 2. LOOCV evaluation
    print("=" * 60)
    print("PHASE 1: LOOCV Evaluation")
    print("=" * 60)
    fold_results = run_loocv(manifest)
    metrics = print_results_table(fold_results)

    # 3. Train final model on all subjects
    print("\n" + "=" * 60)
    print("PHASE 2: Training Final Model (all subjects)")
    print("=" * 60)
    final_ckpt = train_final_model(manifest)

    # 4. Precompute test features
    print("\n" + "=" * 60)
    print("PHASE 3: Precomputing Test Features")
    print("=" * 60)
    test_manifest = precompute_test_features()

    if not test_manifest:
        print("No test subjects found. Exiting.")
        sys.exit(0)

    # 5. Predict
    print("\n" + "=" * 60)
    print("PHASE 4: Test Predictions")
    print("=" * 60)
    results = predict_subjects(final_ckpt, test_manifest)

    print(f"\n{'Subject':>8}  {'P(AD)':>7}  {'Pred':>5}")
    print("─" * 25)
    for sid, prob, label in results:
        label_str = "AD" if label == "A" else "CN"
        print(f"{sid:>8}  {prob:>7.3f}  {label_str:>5}")

    # 6. Save predictions CSV
    out_path = cfg.ROOT / "predictions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anonymized_id", "predicted_label", "probability"])
        for sid, prob, label in results:
            writer.writerow([sid, label, f"{prob:.4f}"])
    print(f"\nPredictions saved → {out_path}")
