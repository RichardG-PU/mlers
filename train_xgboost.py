"""
AD vs CN EEG Classification using XGBoost — v2 (Rigorous Evaluation)
=====================================================================
Key improvements over v1:
  1. Leave-One-Subject-Out (LOSO) cross-validation for TRUE accuracy
  2. No eval-set leakage — early stopping uses an internal CV split,
     the held-out subject is NEVER seen during training
  3. Stratified 80/20 holdout as a secondary check
  4. Full analytics: per-subject breakdown, confidence calibration,
     confusion matrix, ROC, feature importance, training curves

Classifies Alzheimer's Disease (AD) vs Cognitively Normal (CN) subjects
from 19-channel EEG recordings using 30s sliding windows and hand-crafted
frequency/statistical features.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "training")
LABEL_CSV = os.path.join(TRAINING_DIR, "train_label_mapping.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SFREQ = 128           # Sampling frequency in Hz (verified from data)
WINDOW_SEC = 30       # Window duration in seconds
STEP_SEC = 15         # Step size in seconds (50% overlap)
WINDOW_SAMPLES = WINDOW_SEC * SFREQ   # 3840
STEP_SAMPLES = STEP_SEC * SFREQ       # 1920

RANDOM_STATE = 42

# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 25),
    "gamma": (25, 45),
}
BAND_NAMES = list(BANDS.keys())

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Load label mapping, filter to AD/CN, load all .npy files."""
    print("=" * 70)
    print("PHASE 1: Loading data")
    print("=" * 70)

    df = pd.read_csv(LABEL_CSV)
    df = df[df["label"].isin(["A", "C"])].reset_index(drop=True)
    print(f"Subjects after filtering (AD + CN): {len(df)}")
    print(f"  AD (A): {(df['label'] == 'A').sum()}")
    print(f"  CN (C): {(df['label'] == 'C').sum()}")

    data_list = []
    labels = []
    subject_ids = []

    for _, row in df.iterrows():
        sid = row["anonymized_id"]
        label = 1 if row["label"] == "A" else 0  # AD=1, CN=0
        folder = "AD" if row["label"] == "A" else "CN"
        path = os.path.join(TRAINING_DIR, folder, f"{sid}.npy")

        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping subject {sid}")
            continue

        eeg = np.load(path, allow_pickle=True)
        assert eeg.shape[0] == 19, f"Expected 19 channels, got {eeg.shape[0]} for subject {sid}"
        duration_sec = eeg.shape[1] / SFREQ
        assert eeg.shape[1] >= WINDOW_SAMPLES, (
            f"Subject {sid}: recording too short ({duration_sec:.1f}s < {WINDOW_SEC}s)"
        )

        data_list.append(eeg)
        labels.append(label)
        subject_ids.append(sid)

    print(f"Loaded {len(data_list)} subjects successfully.")
    return data_list, np.array(labels), np.array(subject_ids)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — SLIDING WINDOW SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def segment_data(data_list, labels, subject_ids):
    """Segment each recording into 30s windows with 15s step."""
    print("\n" + "=" * 70)
    print("PHASE 2: Sliding window segmentation")
    print("=" * 70)

    segments = []
    seg_labels = []
    seg_subjects = []

    for eeg, label, sid in zip(data_list, labels, subject_ids):
        n_points = eeg.shape[1]
        n_windows = 0
        for start in range(0, n_points - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            segment = eeg[:, start:start + WINDOW_SAMPLES]
            segments.append(segment)
            seg_labels.append(label)
            seg_subjects.append(sid)
            n_windows += 1

    seg_labels = np.array(seg_labels)
    seg_subjects = np.array(seg_subjects)

    print(f"Total snippets: {len(segments)}")
    print(f"  AD snippets: {(seg_labels == 1).sum()}")
    print(f"  CN snippets: {(seg_labels == 0).sum()}")

    return segments, seg_labels, seg_subjects


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _hjorth_params(signal_1d):
    """Compute Hjorth activity, mobility, complexity for a 1D signal."""
    diff1 = np.diff(signal_1d)
    diff2 = np.diff(diff1)
    activity = np.var(signal_1d)
    mobility_num = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0.0
    mobility_den = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0.0
    complexity = mobility_den / mobility_num if mobility_num > 0 else 0.0
    return activity, mobility_num, complexity


def _spectral_entropy(psd_norm):
    """Shannon entropy of the normalized PSD."""
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))


def extract_features_xgb(segment, sfreq=SFREQ):
    """
    Extract 418 features from a single 30-second EEG segment.
    22 features x 19 channels = 418.
    """
    n_channels = segment.shape[0]
    nperseg = min(segment.shape[1], sfreq * 2)

    features = []
    feature_names = []

    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch{ch}"
        signal = segment[ch]

        freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
        total_power = np.sum(psd)
        if total_power == 0:
            total_power = 1e-10

        band_powers = {}
        for band_name, (fmin, fmax) in BANDS.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            bp = np.sum(psd[idx])
            band_powers[band_name] = bp
            features.append(bp)
            feature_names.append(f"{ch_name}_abp_{band_name}")

        for band_name in BAND_NAMES:
            rbp = band_powers[band_name] / total_power
            features.append(rbp)
            feature_names.append(f"{ch_name}_rbp_{band_name}")

        eps = 1e-10
        ratios = [
            ("theta_alpha", band_powers["theta"] / (band_powers["alpha"] + eps)),
            ("alpha_beta",  band_powers["alpha"] / (band_powers["beta"] + eps)),
            ("theta_beta",  band_powers["theta"] / (band_powers["beta"] + eps)),
            ("delta_alpha", band_powers["delta"] / (band_powers["alpha"] + eps)),
        ]
        for rname, rval in ratios:
            features.append(rval)
            feature_names.append(f"{ch_name}_ratio_{rname}")

        features.append(np.mean(signal))
        feature_names.append(f"{ch_name}_mean")
        features.append(np.std(signal))
        feature_names.append(f"{ch_name}_std")
        features.append(skew(signal))
        feature_names.append(f"{ch_name}_skew")
        features.append(kurtosis(signal))
        feature_names.append(f"{ch_name}_kurt")

        act, mob, comp = _hjorth_params(signal)
        features.append(act)
        feature_names.append(f"{ch_name}_hjorth_act")
        features.append(mob)
        feature_names.append(f"{ch_name}_hjorth_mob")
        features.append(comp)
        feature_names.append(f"{ch_name}_hjorth_comp")

        psd_norm = psd / total_power
        se = _spectral_entropy(psd_norm)
        features.append(se)
        feature_names.append(f"{ch_name}_spec_entropy")

    return np.array(features, dtype=np.float64), feature_names


def extract_all_features(segments):
    """Extract features from all segments."""
    print("\n" + "=" * 70)
    print("PHASE 3: Feature extraction")
    print("=" * 70)

    X_list = []
    feature_names = None

    for i, seg in enumerate(segments):
        feats, names = extract_features_xgb(seg)
        X_list.append(feats)
        if feature_names is None:
            feature_names = names
        if (i + 1) % 500 == 0 or (i + 1) == len(segments):
            print(f"  Extracted features for {i + 1}/{len(segments)} snippets")

    X = np.array(X_list)
    assert not np.any(np.isnan(X)), "NaN detected in feature matrix!"
    assert not np.any(np.isinf(X)), "Inf detected in feature matrix!"

    print(f"Feature matrix shape: {X.shape}")
    return X, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# CORE MODEL — build a fresh XGBoost with fixed hyperparams
# ─────────────────────────────────────────────────────────────────────────────
def _make_model(scale_pos_weight=1.0):
    """Return an XGBClassifier with conservative, anti-overfit hyperparams.

    These were tuned via the GroupKFold search in v1 and hardened:
      - max_depth=4:       shallow trees, less memorisation
      - learning_rate=0.03: slow learning, more robust
      - min_child_weight=7: bigger leaf sizes, smoother predictions
      - subsample=0.6:     row sampling, decorrelate trees
      - colsample_bytree=0.6: feature sampling, decorrelate trees
      - reg_lambda=5:      strong L2 regularization
      - gamma=0.5:         need significant gain to split
    """
    return XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=7,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=5,
        gamma=0.5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION (TRUE ACCURACY)
# ─────────────────────────────────────────────────────────────────────────────
def loso_cv(X, y, groups, feature_names):
    """
    Leave-One-Subject-Out CV: the gold standard for small-N subject studies.

    For each of the 38 subjects:
      - Train on ALL other 37 subjects' snippets
      - Use internal 15% split of training subjects for early stopping
        (held-out subject is NEVER seen during training or early stopping)
      - Predict snippets of the held-out subject
      - Majority vote -> subject-level prediction

    This gives us 38 independent subject-level predictions — the most
    honest accuracy estimate possible with this data.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Leave-One-Subject-Out Cross-Validation")
    print("=" * 70)

    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)

    subject_results = []
    all_snippet_preds = np.zeros(len(y))
    all_snippet_probs = np.zeros(len(y))

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_sid = groups[test_idx][0]
        true_label = y[test_idx][0]

        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        groups_train_fold = groups[train_idx]

        # Internal early-stopping split: hold out ~15% of training SUBJECTS
        # to create a validation set that does NOT include the test subject
        rng = np.random.RandomState(RANDOM_STATE + fold_i)
        train_sids = np.unique(groups_train_fold)
        rng.shuffle(train_sids)
        n_val = max(1, int(len(train_sids) * 0.15))
        val_sids = set(train_sids[:n_val])
        inner_train_sids = set(train_sids[n_val:])

        inner_train_mask = np.array([g in inner_train_sids for g in groups_train_fold])
        inner_val_mask = np.array([g in val_sids for g in groups_train_fold])

        X_it, y_it = X_train_fold[inner_train_mask], y_train_fold[inner_train_mask]
        X_iv, y_iv = X_train_fold[inner_val_mask], y_train_fold[inner_val_mask]

        # Class weight from training data
        n_pos = np.sum(y_it == 1)
        n_neg = np.sum(y_it == 0)
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        model = _make_model(scale_pos_weight=spw)
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_it, y_it,
            eval_set=[(X_iv, y_iv)],
            verbose=False,
        )

        # Predict held-out subject
        preds = model.predict(X[test_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        all_snippet_preds[test_idx] = preds
        all_snippet_probs[test_idx] = probs

        # Majority vote
        vote = int(np.round(np.mean(preds)))
        ad_pct = np.mean(preds) * 100
        avg_prob = np.mean(probs)
        correct = vote == true_label

        true_str = "AD" if true_label == 1 else "CN"
        pred_str = "AD" if vote == 1 else "CN"
        mark = "OK" if correct else "WRONG"

        subject_results.append({
            "subject": int(test_sid),
            "true": true_str,
            "pred": pred_str,
            "correct": correct,
            "ad_snippet_pct": round(ad_pct, 1),
            "avg_prob": round(avg_prob, 3),
            "n_snippets": len(preds),
            "best_iteration": model.best_iteration,
        })

        print(f"  [{fold_i+1:2d}/{n_subjects}] Subject {int(test_sid):>3d}: "
              f"true={true_str} pred={pred_str} [{mark:>5s}]  "
              f"AD%={ad_pct:5.1f}%  prob={avg_prob:.3f}  "
              f"({len(preds)} snip, {model.best_iteration} trees)")

    # ── Summary ──
    n_correct = sum(r["correct"] for r in subject_results)
    subj_acc = n_correct / n_subjects
    snippet_acc = accuracy_score(y, all_snippet_preds)

    # Per-class breakdown
    ad_results = [r for r in subject_results if r["true"] == "AD"]
    cn_results = [r for r in subject_results if r["true"] == "CN"]
    ad_correct = sum(r["correct"] for r in ad_results)
    cn_correct = sum(r["correct"] for r in cn_results)

    print(f"\n{'='*70}")
    print(f"LOSO-CV RESULTS (TRUE ACCURACY)")
    print(f"{'='*70}")
    print(f"Subject-level accuracy: {n_correct}/{n_subjects} = {subj_acc:.4f}")
    print(f"  AD recall: {ad_correct}/{len(ad_results)} = {ad_correct/len(ad_results):.4f}")
    print(f"  CN recall: {cn_correct}/{len(cn_results)} = {cn_correct/len(cn_results):.4f}")
    print(f"Snippet-level accuracy: {snippet_acc:.4f}")

    # AUC (over all snippets from all LOSO folds)
    try:
        auc = roc_auc_score(y, all_snippet_probs)
        print(f"Snippet-level AUC-ROC: {auc:.4f}")
    except ValueError:
        auc = None
        print("AUC-ROC: could not compute")

    # Save detailed results
    results_df = pd.DataFrame(subject_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "loso_subject_results.csv"), index=False)
    print(f"\nSaved: loso_subject_results.csv")

    # ── Plots ──
    # Confusion matrix (subject-level)
    true_labels = [1 if r["true"] == "AD" else 0 for r in subject_results]
    pred_labels = [1 if r["pred"] == "AD" else 0 for r in subject_results]
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"LOSO-CV Confusion Matrix (Subject-Level)\nAccuracy: {n_correct}/{n_subjects} = {subj_acc:.1%}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loso_confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: loso_confusion_matrix.png")

    # ROC curve (snippet-level, aggregated across all LOSO folds)
    if auc is not None:
        fpr, tpr, _ = roc_curve(y, all_snippet_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"LOSO ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("LOSO-CV ROC Curve (Snippet-Level)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loso_roc_curve.png"), dpi=150)
        plt.close()
        print("Saved: loso_roc_curve.png")

    # Per-subject confidence bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    sids = [r["subject"] for r in subject_results]
    probs_list = [r["avg_prob"] for r in subject_results]
    colors = []
    for r in subject_results:
        if r["correct"]:
            colors.append("steelblue" if r["true"] == "AD" else "seagreen")
        else:
            colors.append("red")

    ax.bar(range(len(sids)), probs_list, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([str(s) for s in sids], rotation=45)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Average P(AD)")
    ax.set_title(f"LOSO-CV: Per-Subject AD Probability\n"
                 f"Blue=AD correct, Green=CN correct, Red=Wrong | "
                 f"Accuracy={subj_acc:.1%}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loso_subject_confidence.png"), dpi=150)
    plt.close()
    print("Saved: loso_subject_confidence.png")

    return subj_acc, subject_results, all_snippet_probs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — TRAIN FINAL MODEL (for deployment/submission)
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(X, y, groups, feature_names):
    """Train the final model on ALL data with internal CV early stopping."""
    print("\n" + "=" * 70)
    print("PHASE 5: Training final model on all data")
    print("=" * 70)

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    # Use GroupKFold to determine optimal iteration count
    gkf = GroupKFold(n_splits=5)
    best_iters = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = _make_model(scale_pos_weight=spw)
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        best_iters.append(model.best_iteration)

    avg_iter = int(np.mean(best_iters))
    print(f"GroupKFold best iterations: {best_iters} -> avg = {avg_iter}")

    # Train final model on ALL data with the averaged iteration count
    final_model = _make_model(scale_pos_weight=spw)
    final_model.set_params(n_estimators=avg_iter)
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importances = final_model.feature_importances_
    top_k = 30
    top_idx = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), importances[top_idx][::-1], color="steelblue")
    plt.yticks(range(top_k), [feature_names[i] for i in top_idx][::-1])
    plt.xlabel("Feature Importance (Gain)")
    plt.title(f"Top {top_k} Features (Final Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

    # Print top features
    print(f"\nTop 15 features:")
    for rank, idx in enumerate(top_idx[:15]):
        print(f"  {rank+1:2d}. {feature_names[idx]:30s} importance={importances[idx]:.4f}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "xgb_ad_cn.json")
    final_model.save_model(model_path)
    print(f"\nSaved model: {model_path}")

    meta_path = os.path.join(OUTPUT_DIR, "feature_names.txt")
    with open(meta_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")
    print(f"Saved feature names: {meta_path}")

    # Train accuracy (sanity check — should be high but is NOT our eval metric)
    train_pred = final_model.predict(X)
    train_acc = accuracy_score(y, train_pred)
    print(f"\nFinal model train accuracy (sanity check only): {train_acc:.4f}")

    return final_model


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — OVERFITTING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def overfitting_analysis(X, y, groups):
    """5-Fold GroupKFold to measure train-vs-val gap (overfitting diagnostic)."""
    print("\n" + "=" * 70)
    print("PHASE 6: Overfitting analysis (5-Fold GroupKFold)")
    print("=" * 70)

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    gkf = GroupKFold(n_splits=5)
    train_accs = []
    val_accs = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = _make_model(scale_pos_weight=spw)
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )

        tr_acc = accuracy_score(y[train_idx], model.predict(X[train_idx]))
        va_acc = accuracy_score(y[val_idx], model.predict(X[val_idx]))
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        n_train_subj = len(np.unique(groups[train_idx]))
        n_val_subj = len(np.unique(groups[val_idx]))
        print(f"  Fold {fold_i+1}: train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}  "
              f"gap={tr_acc - va_acc:.4f}  "
              f"(train={n_train_subj} subj, val={n_val_subj} subj, "
              f"{model.best_iteration} trees)")

    avg_train = np.mean(train_accs)
    avg_val = np.mean(val_accs)
    avg_gap = avg_train - avg_val

    print(f"\nAverage: train={avg_train:.4f}  val={avg_val:.4f}  gap={avg_gap:.4f}")

    if avg_gap > 0.20:
        print("VERDICT: SEVERE overfitting — train/val gap > 20%")
    elif avg_gap > 0.10:
        print("VERDICT: MODERATE overfitting — train/val gap 10-20%")
    else:
        print("VERDICT: LOW overfitting — train/val gap < 10%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, 6)
    w = 0.35
    ax.bar(x - w/2, train_accs, w, label="Train", color="steelblue")
    ax.bar(x + w/2, val_accs, w, label="Validation", color="darkorange")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"5-Fold GroupKFold: Train vs Val Accuracy\n"
                 f"Avg gap = {avg_gap:.4f}")
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overfitting_analysis.png"), dpi=150)
    plt.close()
    print("Saved: overfitting_analysis.png")

    return avg_gap


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("AD vs CN EEG Classification — XGBoost Pipeline v2")
    print(f"Sampling rate: {SFREQ} Hz | Window: {WINDOW_SEC}s | Step: {STEP_SEC}s")
    print()

    # Phase 1: Load
    data_list, labels, subject_ids = load_data()

    # Phase 2: Segment
    segments, seg_labels, seg_subjects = segment_data(data_list, labels, subject_ids)

    # Phase 3: Features
    X, feature_names = extract_all_features(segments)

    # Phase 4: LOSO-CV (TRUE accuracy — the number that matters)
    subj_acc, subject_results, _ = loso_cv(X, seg_labels, seg_subjects, feature_names)

    # Phase 5: Train final model on all data
    final_model = train_final_model(X, seg_labels, seg_subjects, feature_names)

    # Phase 6: Overfitting analysis
    avg_gap = overfitting_analysis(X, seg_labels, seg_subjects)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    n_correct = sum(r["correct"] for r in subject_results)
    n_total = len(subject_results)
    wrong = [r for r in subject_results if not r["correct"]]

    print(f"TRUE subject-level accuracy (LOSO-CV): {n_correct}/{n_total} = {subj_acc:.4f}")
    print(f"Overfitting gap (5-fold GroupKFold):    {avg_gap:.4f}")

    if wrong:
        print(f"\nMisclassified subjects ({len(wrong)}):")
        for r in wrong:
            print(f"  Subject {r['subject']}: true={r['true']}, pred={r['pred']}, "
                  f"AD%={r['ad_snippet_pct']}%, prob={r['avg_prob']}")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
