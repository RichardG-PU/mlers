"""
Meta-ensemble: combine DICE-net and XGBoost LOOCV predictions to find optimal weighting.

Usage:
    python scripts/ensemble.py

Prerequisites:
    - python scripts/predict.py  (generates output/dicenet_loocv.csv)
    - xgboost_predictions/loocv.csv must exist (from lasmar branch)
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as cfg

OUTPUT_DIR = cfg.ROOT / "output"


def load_dicenet_loocv(path: Path) -> dict:
    """Load DICE-net LOOCV: {subject_id: (true_label, prob)}"""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject"]
            true_lbl = 1 if row["true_label"] == "AD" else 0
            prob = float(row["probability"])
            results[sid] = (true_lbl, prob)
    return results


def load_xgboost_loocv(path: Path) -> dict:
    """Load XGBoost LOOCV: {subject_id: (true_label, prob)}"""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject"]
            true_lbl = 1 if row["true"] == "AD" else 0
            prob = float(row["avg_prob"])
            results[sid] = (true_lbl, prob)
    return results


def ensemble_accuracy(dicenet: dict, xgboost: dict, dice_weight: float) -> tuple:
    """
    Compute accuracy of weighted ensemble on shared subjects.

    Returns: (accuracy, n_correct, n_total, per_subject_details)
    """
    shared_sids = sorted(set(dicenet.keys()) & set(xgboost.keys()), key=int)
    assert len(shared_sids) > 0, "No shared subjects between DICE-net and XGBoost!"

    correct = 0
    details = []
    for sid in shared_sids:
        true_lbl, dice_prob = dicenet[sid]
        _, xgb_prob = xgboost[sid]

        ens_prob = dice_weight * dice_prob + (1 - dice_weight) * xgb_prob
        ens_pred = 1 if ens_prob >= 0.5 else 0

        is_correct = ens_pred == true_lbl
        if is_correct:
            correct += 1
        details.append({
            "subject": sid,
            "true": true_lbl,
            "dice_prob": dice_prob,
            "xgb_prob": xgb_prob,
            "ens_prob": ens_prob,
            "ens_pred": ens_pred,
            "correct": is_correct,
        })

    return correct / len(shared_sids), correct, len(shared_sids), details


if __name__ == "__main__":
    dice_path = OUTPUT_DIR / "dicenet_loocv.csv"
    xgb_path = cfg.ROOT / "xgboost_predictions" / "loocv.csv"

    # Check prerequisites
    if not dice_path.exists():
        print(f"ERROR: {dice_path} not found. Run: python scripts/predict.py")
        sys.exit(1)
    if not xgb_path.exists():
        print(f"ERROR: {xgb_path} not found.")
        sys.exit(1)

    dicenet = load_dicenet_loocv(dice_path)
    xgboost = load_xgboost_loocv(xgb_path)

    print(f"DICE-net LOOCV: {len(dicenet)} subjects")
    print(f"XGBoost LOOCV:  {len(xgboost)} subjects")
    shared = set(dicenet.keys()) & set(xgboost.keys())
    print(f"Shared subjects: {len(shared)}")
    assert len(shared) == len(dicenet) == len(xgboost), \
        f"Subject mismatch! DICE-net has {set(dicenet.keys()) - shared}, XGBoost has {set(xgboost.keys()) - shared}"

    # ── Individual model performance ────────────────────────────────
    print("\n── Individual Model Performance (LOOCV) ──────────────────")

    dice_correct = sum(1 for sid in shared
                       if (1 if dicenet[sid][1] >= 0.5 else 0) == dicenet[sid][0])
    xgb_correct = sum(1 for sid in shared
                      if (1 if xgboost[sid][1] >= 0.5 else 0) == xgboost[sid][0])

    print(f"  DICE-net:  {dice_correct}/{len(shared)} = {dice_correct/len(shared):.1%}")
    print(f"  XGBoost:   {xgb_correct}/{len(shared)} = {xgb_correct/len(shared):.1%}")

    # ── Error comparison ────────────────────────────────────────────
    print("\n── Error Comparison ──────────────────────────────────────")
    dice_errors = set()
    xgb_errors = set()
    for sid in sorted(shared, key=int):
        dice_pred = 1 if dicenet[sid][1] >= 0.5 else 0
        xgb_pred = 1 if xgboost[sid][1] >= 0.5 else 0
        true = dicenet[sid][0]

        if dice_pred != true:
            dice_errors.add(sid)
        if xgb_pred != true:
            xgb_errors.add(sid)

    both_wrong = dice_errors & xgb_errors
    only_dice_wrong = dice_errors - xgb_errors
    only_xgb_wrong = xgb_errors - dice_errors

    print(f"  DICE-net errors ({len(dice_errors)}): {sorted(dice_errors, key=int)}")
    print(f"  XGBoost errors  ({len(xgb_errors)}): {sorted(xgb_errors, key=int)}")
    print(f"  Both wrong      ({len(both_wrong)}): {sorted(both_wrong, key=int)}")
    print(f"  Only DICE wrong ({len(only_dice_wrong)}): {sorted(only_dice_wrong, key=int)}")
    print(f"  Only XGB wrong  ({len(only_xgb_wrong)}): {sorted(only_xgb_wrong, key=int)}")

    if len(both_wrong) < len(dice_errors) or len(both_wrong) < len(xgb_errors):
        print("\n  → Models make DIFFERENT errors — ensemble has high upside!")
    else:
        print("\n  → Models make the SAME errors — ensemble upside is limited.")

    # ── Grid search for optimal DICE-net weight ─────────────────────
    print("\n── Meta-Ensemble Weight Grid Search ──────────────────────")
    weights = np.arange(0.0, 1.05, 0.1)
    best_acc, best_w = 0, 0.5

    print(f"  {'Weight':>8}  {'Accuracy':>10}  {'Correct':>8}")
    print("  " + "─" * 30)
    for w in weights:
        acc, n_correct, n_total, _ = ensemble_accuracy(dicenet, xgboost, w)
        marker = " ◄" if acc > best_acc else ""
        print(f"  {w:>8.1f}  {acc:>10.1%}  {n_correct:>4}/{n_total}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print(f"\n  Optimal weight: DICE-net={best_w:.1f}, XGBoost={1-best_w:.1f}")
    print(f"  Best ensemble accuracy: {best_acc:.1%}")

    # ── Per-subject detail at optimal weight ────────────────────────
    _, _, _, details = ensemble_accuracy(dicenet, xgboost, best_w)

    print(f"\n── Per-Subject Detail (weight={best_w:.1f}) ────────────────")
    print(f"  {'Subject':>8}  {'True':>5}  {'DICE':>7}  {'XGB':>7}  {'Ens':>7}  {'Pred':>5}  {'OK':>3}")
    print("  " + "─" * 50)
    for d in details:
        true_str = "AD" if d["true"] == 1 else "CN"
        pred_str = "AD" if d["ens_pred"] == 1 else "CN"
        ok = "✓" if d["correct"] else "✗"
        print(f"  {d['subject']:>8}  {true_str:>5}  {d['dice_prob']:>7.3f}  "
              f"{d['xgb_prob']:>7.3f}  {d['ens_prob']:>7.3f}  {pred_str:>5}  {ok:>3}")

    # ── Save ensemble LOOCV results ─────────────────────────────────
    ens_csv = OUTPUT_DIR / "ensemble_loocv.csv"
    with open(ens_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "true_label", "dice_prob", "xgb_prob",
                          "ensemble_prob", "pred_label", "correct",
                          "dice_weight", "xgb_weight"])
        for d in details:
            writer.writerow([
                d["subject"],
                "AD" if d["true"] == 1 else "CN",
                f"{d['dice_prob']:.4f}",
                f"{d['xgb_prob']:.4f}",
                f"{d['ens_prob']:.4f}",
                "AD" if d["ens_pred"] == 1 else "CN",
                d["correct"],
                f"{best_w:.1f}",
                f"{1-best_w:.1f}",
            ])
    print(f"\n  Ensemble LOOCV saved → {ens_csv}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  DICE-net alone:     {dice_correct}/{len(shared)} ({dice_correct/len(shared):.1%})")
    print(f"  XGBoost alone:      {xgb_correct}/{len(shared)} ({xgb_correct/len(shared):.1%})")
    print(f"  Meta-ensemble:      {int(best_acc * len(shared))}/{len(shared)} ({best_acc:.1%})")
    print(f"  Optimal weights:    DICE={best_w:.1f}, XGB={1-best_w:.1f}")
