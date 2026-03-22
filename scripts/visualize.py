"""
Interpretability visualizations for AD vs CN EEG classification.

Generates:
    1. Per-frequency-band power comparison (AD vs CN boxplots)
    2. Confusion matrices for each model + ensemble
    3. ROC curves for each model
    4. Per-subject confidence comparison (DICE-net vs XGBoost)
    5. Model agreement/disagreement chart

Usage:
    python scripts/visualize.py

Prerequisites:
    - python scripts/predict.py   (generates output/dicenet_loocv.csv)
    - python scripts/ensemble.py  (generates output/ensemble_loocv.csv)
    - xgboost_predictions/loocv.csv must exist
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as cfg

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib required. Install: pip install matplotlib")
    sys.exit(1)

OUTPUT_DIR = cfg.ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ────────────────────────────────────────────────────

def load_dicenet_loocv() -> dict:
    path = OUTPUT_DIR / "dicenet_loocv.csv"
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            results[row["subject"]] = {
                "true": 1 if row["true_label"] == "AD" else 0,
                "prob": float(row["probability"]),
                "pred": 1 if row["pred_label"] == "AD" else 0,
            }
    return results


def load_xgboost_loocv() -> dict:
    path = cfg.ROOT / "xgboost_predictions" / "loocv.csv"
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            results[row["subject"]] = {
                "true": 1 if row["true"] == "AD" else 0,
                "prob": float(row["avg_prob"]),
                "pred": 1 if row["pred"] == "AD" else 0,
            }
    return results


def load_ensemble_loocv() -> dict:
    path = OUTPUT_DIR / "ensemble_loocv.csv"
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            results[row["subject"]] = {
                "true": 1 if row["true_label"] == "AD" else 0,
                "prob": float(row["ensemble_prob"]),
                "pred": 1 if row["pred_label"] == "AD" else 0,
            }
    return results


# ── Plot 1: Confusion matrices ─────────────────────────────────────

def plot_confusion_matrices(dice: dict, xgb: dict, ens: dict):
    """Side-by-side confusion matrices for all models."""
    models = [("DICE-net", dice), ("XGBoost", xgb)]
    if ens:
        models.append(("Meta-Ensemble", ens))

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, models):
        sids = sorted(data.keys(), key=int)
        y_true = [data[s]["true"] for s in sids]
        y_pred = [data[s]["pred"] for s in sids]

        # Compute confusion matrix
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        cm = np.array([[tn, fp], [fn, tp]])
        acc = (tp + tn) / len(y_true)

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.flat))
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["CN", "AD"])
        ax.set_yticklabels(["CN", "AD"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\nAcc={acc:.1%}")

    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 2: ROC curves ─────────────────────────────────────────────

def compute_roc(y_true: list, y_prob: list) -> tuple:
    """Compute ROC curve points."""
    order = np.argsort(-np.array(y_prob))
    y_true_s = np.array(y_true)[order]

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for label in y_true_s:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def plot_roc_curves(dice: dict, xgb: dict, ens: dict):
    """Overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(6, 6))

    models = [("DICE-net", dice, "#2196F3"), ("XGBoost", xgb, "#FF9800")]
    if ens:
        models.append(("Meta-Ensemble", ens, "#4CAF50"))

    for name, data, color in models:
        sids = sorted(data.keys(), key=int)
        y_true = [data[s]["true"] for s in sids]
        y_prob = [data[s]["prob"] for s in sids]
        fpr, tpr, auc = compute_roc(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — LOOCV")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    path = OUTPUT_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: Per-subject confidence comparison ───────────────────────

def plot_subject_confidence(dice: dict, xgb: dict):
    """Per-subject P(AD) from both models, colored by true label."""
    shared = sorted(set(dice.keys()) & set(xgb.keys()), key=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(shared))
    width = 0.35

    dice_probs = [dice[s]["prob"] for s in shared]
    xgb_probs = [xgb[s]["prob"] for s in shared]
    true_labels = [dice[s]["true"] for s in shared]

    # Color bars by true label
    dice_colors = ["#EF5350" if t == 1 else "#42A5F5" for t in true_labels]
    xgb_colors = ["#C62828" if t == 1 else "#1565C0" for t in true_labels]

    ax.bar(x - width / 2, dice_probs, width, color=dice_colors, alpha=0.7, label="DICE-net")
    ax.bar(x + width / 2, xgb_probs, width, color=xgb_colors, alpha=0.7, label="XGBoost")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("P(AD)")
    ax.set_title("Per-Subject Confidence: DICE-net vs XGBoost (LOOCV)")
    ax.set_xticks(x)
    ax.set_xticklabels(shared, fontsize=7, rotation=45)
    ax.set_ylim(0, 1)

    # Legend
    ad_patch = mpatches.Patch(color="#EF5350", alpha=0.7, label="True AD")
    cn_patch = mpatches.Patch(color="#42A5F5", alpha=0.7, label="True CN")
    ax.legend(handles=[ad_patch, cn_patch], loc="upper right")

    path = OUTPUT_DIR / "subject_confidence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 4: Model agreement/disagreement ───────────────────────────

def plot_agreement(dice: dict, xgb: dict):
    """Show which subjects models agree/disagree on."""
    shared = sorted(set(dice.keys()) & set(xgb.keys()), key=int)

    categories = {"Both correct": 0, "Both wrong": 0,
                  "Only DICE correct": 0, "Only XGB correct": 0}
    subject_cats = {}

    for sid in shared:
        true = dice[sid]["true"]
        d_ok = dice[sid]["pred"] == true
        x_ok = xgb[sid]["pred"] == true
        if d_ok and x_ok:
            categories["Both correct"] += 1
            subject_cats[sid] = "Both correct"
        elif not d_ok and not x_ok:
            categories["Both wrong"] += 1
            subject_cats[sid] = "Both wrong"
        elif d_ok:
            categories["Only DICE correct"] += 1
            subject_cats[sid] = "Only DICE correct"
        else:
            categories["Only XGB correct"] += 1
            subject_cats[sid] = "Only XGB correct"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]
    labels = list(categories.keys())
    sizes = [categories[l] for l in labels]
    nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    ax1.pie([s for _, s, _ in nonzero],
            labels=[f"{l}\n({s})" for l, s, _ in nonzero],
            colors=[c for _, _, c in nonzero],
            autopct="%1.0f%%", startangle=90)
    ax1.set_title("Model Agreement (LOOCV)")

    # Subject-level detail
    cat_colors = {"Both correct": "#4CAF50", "Both wrong": "#F44336",
                  "Only DICE correct": "#2196F3", "Only XGB correct": "#FF9800"}
    bar_colors = [cat_colors[subject_cats[s]] for s in shared]
    ax2.bar(range(len(shared)), [1] * len(shared), color=bar_colors)
    ax2.set_xticks(range(len(shared)))
    ax2.set_xticklabels(shared, fontsize=6, rotation=45)
    ax2.set_yticks([])
    ax2.set_title("Per-Subject Agreement")
    ax2.set_xlabel("Subject ID")

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()
               if categories[l] > 0]
    ax2.legend(handles=patches, loc="upper right", fontsize=8)

    path = OUTPUT_DIR / "model_agreement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: Band power analysis (AD vs CN) ─────────────────────────

def plot_band_power():
    """Per-frequency-band relative power comparison between AD and CN subjects."""
    from src.cache import load_manifest

    manifest = load_manifest()
    band_names = ["Delta\n(0.5-4)", "Theta\n(4-8)", "Alpha\n(8-13)",
                  "Beta\n(13-25)", "Gamma\n(25-45)"]

    ad_power = {b: [] for b in range(5)}
    cn_power = {b: [] for b in range(5)}

    for sid, info in manifest.items():
        rbp = np.load(cfg.CACHE_DIR / f"{sid}_rbp.npy")  # (n_epochs, 30, 5, 19)
        # Average across epochs, time windows, and channels → per-band power
        mean_rbp = rbp.mean(axis=(0, 1, 3))  # (5,) — one value per band

        target = ad_power if info["label"] == "A" else cn_power
        for b in range(5):
            target[b].append(mean_rbp[b])

    fig, ax = plt.subplots(figsize=(10, 5))

    positions_ad = np.arange(5) * 2 - 0.3
    positions_cn = np.arange(5) * 2 + 0.3

    bp_ad = ax.boxplot([ad_power[b] for b in range(5)], positions=positions_ad,
                       widths=0.5, patch_artist=True, showfliers=False)
    bp_cn = ax.boxplot([cn_power[b] for b in range(5)], positions=positions_cn,
                       widths=0.5, patch_artist=True, showfliers=False)

    for box in bp_ad["boxes"]:
        box.set_facecolor("#EF5350")
        box.set_alpha(0.7)
    for box in bp_cn["boxes"]:
        box.set_facecolor("#42A5F5")
        box.set_alpha(0.7)

    ax.set_xticks(np.arange(5) * 2)
    ax.set_xticklabels(band_names)
    ax.set_ylabel("Relative Band Power")
    ax.set_title("Frequency Band Power: AD vs CN")

    ad_patch = mpatches.Patch(color="#EF5350", alpha=0.7, label="AD")
    cn_patch = mpatches.Patch(color="#42A5F5", alpha=0.7, label="CN")
    ax.legend(handles=[ad_patch, cn_patch])

    path = OUTPUT_DIR / "band_power.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating interpretability visualizations...\n")

    dice = load_dicenet_loocv()
    xgb = load_xgboost_loocv()
    ens = load_ensemble_loocv()

    print(f"  DICE-net: {len(dice)} subjects")
    print(f"  XGBoost:  {len(xgb)} subjects")
    print(f"  Ensemble: {len(ens)} subjects\n")

    print("1. Confusion matrices")
    plot_confusion_matrices(dice, xgb, ens)

    print("2. ROC curves")
    plot_roc_curves(dice, xgb, ens)

    print("3. Per-subject confidence comparison")
    plot_subject_confidence(dice, xgb)

    print("4. Model agreement/disagreement")
    plot_agreement(dice, xgb)

    print("5. Frequency band power (AD vs CN)")
    plot_band_power()

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
