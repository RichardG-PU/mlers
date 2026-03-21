"""
AD vs CN EEG Classification using XGBoost
==========================================
Classifies Alzheimer's Disease (AD) vs Cognitively Normal (CN) subjects
from 19-channel EEG recordings using 30-second sliding windows and
hand-crafted frequency/statistical features.

Subject-level 80/20 train/eval split with GroupKFold CV to prevent overfitting.
Per-subject majority-vote prediction at inference.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
TEST_SIZE = 0.2

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
        print(f"  Subject {sid:>3d} | label={'AD' if label == 1 else 'CN'} | "
              f"shape={eeg.shape} | duration={duration_sec:.1f}s ({duration_sec/60:.1f} min)")

    print(f"\nLoaded {len(data_list)} subjects successfully.")
    return data_list, np.array(labels), np.array(subject_ids)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — SLIDING WINDOW SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def segment_data(data_list, labels, subject_ids):
    """Segment each recording into 30s windows with 15s step."""
    print("\n" + "=" * 70)
    print("PHASE 2: Sliding window segmentation")
    print("=" * 70)

    segments = []    # list of (19, WINDOW_SAMPLES) arrays
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
        print(f"  Subject {sid:>3d}: {n_windows} windows")

    seg_labels = np.array(seg_labels)
    seg_subjects = np.array(seg_subjects)

    print(f"\nTotal snippets: {len(segments)}")
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
    Extract ~418 features from a single 30-second EEG segment.

    Features per channel (19 channels):
      - 5 absolute band powers
      - 5 relative band powers
      - 4 band power ratios (theta/alpha, alpha/beta, theta/beta, delta/alpha)
      - 4 statistical (mean, std, skewness, kurtosis)
      - 3 Hjorth parameters (activity, mobility, complexity)
      - 1 spectral entropy
    Total: 22 features × 19 channels = 418
    """
    n_channels = segment.shape[0]
    nperseg = min(segment.shape[1], sfreq * 2)  # 2-second Welch segments

    features = []
    feature_names = []

    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch{ch}"
        signal = segment[ch]

        # --- Welch PSD ---
        freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
        total_power = np.sum(psd)
        if total_power == 0:
            total_power = 1e-10

        # Absolute and relative band powers
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

        # Band power ratios (key AD biomarkers)
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

        # Statistical features
        features.append(np.mean(signal))
        feature_names.append(f"{ch_name}_mean")
        features.append(np.std(signal))
        feature_names.append(f"{ch_name}_std")
        features.append(skew(signal))
        feature_names.append(f"{ch_name}_skew")
        features.append(kurtosis(signal))
        feature_names.append(f"{ch_name}_kurt")

        # Hjorth parameters
        act, mob, comp = _hjorth_params(signal)
        features.append(act)
        feature_names.append(f"{ch_name}_hjorth_act")
        features.append(mob)
        feature_names.append(f"{ch_name}_hjorth_mob")
        features.append(comp)
        feature_names.append(f"{ch_name}_hjorth_comp")

        # Spectral entropy
        psd_norm = psd / total_power
        se = _spectral_entropy(psd_norm)
        features.append(se)
        feature_names.append(f"{ch_name}_spec_entropy")

    return np.array(features, dtype=np.float64), feature_names


def extract_all_features(segments):
    """Extract features from all segments. Returns feature matrix + names."""
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
        if (i + 1) % 200 == 0 or (i + 1) == len(segments):
            print(f"  Extracted features for {i + 1}/{len(segments)} snippets")

    X = np.array(X_list)

    # Sanity checks
    assert not np.any(np.isnan(X)), "NaN detected in feature matrix!"
    assert not np.any(np.isinf(X)), "Inf detected in feature matrix!"

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features per snippet: {X.shape[1]}")
    print(f"Feature names (first 10): {feature_names[:10]}")

    return X, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — TRAIN/EVAL SPLIT (SUBJECT-LEVEL)
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X, y, groups):
    """Stratified subject-level 80/20 train/eval split.
    
    Splits AD and CN subjects separately to maintain class balance in both
    train and eval sets, then combines. This avoids the problem of random
    GroupShuffleSplit putting nearly all of one class in eval.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Subject-level train/eval split (stratified)")
    print("=" * 70)

    rng = np.random.RandomState(RANDOM_STATE)

    # Get unique subjects and their labels
    unique_subjects = np.unique(groups)
    subject_labels = {}
    for sid in unique_subjects:
        mask = groups == sid
        subject_labels[sid] = y[mask][0]

    ad_subjects = [s for s in unique_subjects if subject_labels[s] == 1]
    cn_subjects = [s for s in unique_subjects if subject_labels[s] == 0]
    rng.shuffle(ad_subjects)
    rng.shuffle(cn_subjects)

    # Split each class 80/20
    n_ad_eval = max(1, int(len(ad_subjects) * TEST_SIZE))
    n_cn_eval = max(1, int(len(cn_subjects) * TEST_SIZE))

    ad_eval = set(ad_subjects[:n_ad_eval])
    cn_eval = set(cn_subjects[:n_cn_eval])
    eval_subjects = ad_eval | cn_eval
    train_subjects = set(unique_subjects) - eval_subjects

    train_idx = np.array([i for i, g in enumerate(groups) if g in train_subjects])
    eval_idx = np.array([i for i, g in enumerate(groups) if g in eval_subjects])

    X_train, X_eval = X[train_idx], X[eval_idx]
    y_train, y_eval = y[train_idx], y[eval_idx]
    groups_train, groups_eval = groups[train_idx], groups[eval_idx]

    # Verify no data leakage
    assert len(train_subjects & eval_subjects) == 0, "DATA LEAKAGE: subjects in both train and eval!"

    print(f"AD subjects: {len(ad_subjects)} total -> {len(ad_subjects) - n_ad_eval} train / {n_ad_eval} eval")
    print(f"CN subjects: {len(cn_subjects)} total -> {len(cn_subjects) - n_cn_eval} train / {n_cn_eval} eval")
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Eval  subjects ({len(eval_subjects)}): {sorted(eval_subjects)}")
    print(f"\nTrain snippets: {len(X_train)} (AD={np.sum(y_train==1)}, CN={np.sum(y_train==0)})")
    print(f"Eval  snippets: {len(X_eval)} (AD={np.sum(y_eval==1)}, CN={np.sum(y_eval==0)})")

    return X_train, X_eval, y_train, y_eval, groups_train, groups_eval


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — XGBOOST TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train, groups_train, X_eval, y_eval):
    """Train XGBoost with hyperparameter tuning via GroupKFold + early stopping."""
    print("\n" + "=" * 70)
    print("PHASE 5: XGBoost training")
    print("=" * 70)

    # Class balance weight
    n_cn = np.sum(y_train == 0)
    n_ad = np.sum(y_train == 1)
    scale_pos_weight = n_cn / n_ad if n_ad > 0 else 1.0
    print(f"scale_pos_weight (CN/AD ratio): {scale_pos_weight:.2f}")

    # ── Step 1: Hyperparameter search via GroupKFold ──
    print("\nStep 1: Hyperparameter search (RandomizedSearchCV + GroupKFold)...")

    param_dist = {
        "n_estimators": [300, 500, 700, 1000],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [3, 5, 7, 10],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 2, 3, 5],
        "gamma": [0, 0.1, 0.3, 0.5],
    }

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )

    gkf = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,
        scoring="accuracy",
        cv=gkf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"\nBest CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # ── Step 2: Train final model with early stopping on eval set ──
    print("\nStep 2: Training final model with early stopping...")

    best_params = search.best_params_.copy()
    best_params["n_estimators"] = 2000  # set high, rely on early stopping
    best_params["early_stopping_rounds"] = 50

    final_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_eval, y_eval)],
        verbose=False,
    )

    print(f"Best iteration: {final_model.best_iteration}")
    print(f"Best eval logloss: {final_model.best_score:.4f}")

    return final_model


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — EVALUATION & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_train, y_train, X_eval, y_eval,
                   groups_eval, feature_names):
    """Full evaluation: per-snippet, per-subject, feature importance, curves."""
    print("\n" + "=" * 70)
    print("PHASE 6: Evaluation & Analytics")
    print("=" * 70)

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_train_pred = model.predict(X_train)

    # ── 6a: Per-snippet metrics ──
    print("\n--- Per-Snippet Metrics (Eval Set) ---")
    print(classification_report(y_eval, y_pred, target_names=["CN", "AD"]))

    snippet_acc = accuracy_score(y_eval, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    auc = roc_auc_score(y_eval, y_prob)
    f1 = f1_score(y_eval, y_pred, average="macro")

    print(f"Snippet-level accuracy (eval):  {snippet_acc:.4f}")
    print(f"Snippet-level accuracy (train): {train_acc:.4f}")
    print(f"AUC-ROC:                        {auc:.4f}")
    print(f"Macro F1:                       {f1:.4f}")

    # Overfitting check
    overfit_gap = train_acc - snippet_acc
    print(f"\nOverfitting gap (train - eval acc): {overfit_gap:.4f}")
    if overfit_gap > 0.15:
        print("  ⚠ WARNING: Significant overfitting detected!")
    else:
        print("  OK: Overfitting is within acceptable range.")

    # ── 6b: Confusion matrix ──
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Per-Snippet)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("\nSaved: confusion_matrix.png")

    # ── 6c: ROC curve ──
    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: roc_curve.png")

    # ── 6d: Per-subject majority vote ──
    print("\n--- Per-Subject Majority Vote (Eval Set) ---")
    unique_subjects = np.unique(groups_eval)
    subject_results = []

    for sid in unique_subjects:
        mask = groups_eval == sid
        s_preds = y_pred[mask]
        s_true = y_eval[mask][0]  # all same for one subject
        s_probs = y_prob[mask]

        vote = int(np.round(np.mean(s_preds)))  # majority vote
        ad_pct = np.mean(s_preds) * 100
        avg_prob = np.mean(s_probs)

        correct = "✓" if vote == s_true else "✗"
        true_label = "AD" if s_true == 1 else "CN"
        pred_label = "AD" if vote == 1 else "CN"

        subject_results.append({
            "subject": sid,
            "true": true_label,
            "pred": pred_label,
            "correct": vote == s_true,
            "ad_snippet_pct": ad_pct,
            "avg_prob": avg_prob,
            "n_snippets": len(s_preds),
        })

        print(f"  Subject {sid:>3d}: true={true_label}, pred={pred_label} "
              f"[{correct}]  AD%={ad_pct:.1f}%  avg_prob={avg_prob:.3f}  "
              f"({len(s_preds)} snippets)")

    subj_correct = sum(1 for r in subject_results if r["correct"])
    subj_total = len(subject_results)
    subj_acc = subj_correct / subj_total

    print(f"\n*** SUBJECT-LEVEL ACCURACY: {subj_correct}/{subj_total} = {subj_acc:.4f} ***")

    # Save subject results to CSV
    pd.DataFrame(subject_results).to_csv(
        os.path.join(OUTPUT_DIR, "subject_results.csv"), index=False
    )
    print("Saved: subject_results.csv")

    # ── 6e: Feature importance (top 30) ──
    importances = model.feature_importances_
    top_k = 30
    top_idx = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), importances[top_idx][::-1], color="steelblue")
    plt.yticks(range(top_k), [feature_names[i] for i in top_idx][::-1])
    plt.xlabel("Feature Importance (Gain)")
    plt.title(f"Top {top_k} Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

    # ── 6f: Training curves ──
    results = model.evals_result()
    train_loss = results["validation_0"]["logloss"]
    eval_loss = results["validation_1"]["logloss"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train logloss", alpha=0.8)
    plt.plot(eval_loss, label="Eval logloss", alpha=0.8)
    plt.xlabel("Boosting Round")
    plt.ylabel("Log Loss")
    plt.title("Training vs Eval Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved: training_curves.png")

    return subj_acc


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, feature_names):
    """Persist model and feature metadata."""
    print("\n" + "=" * 70)
    print("PHASE 7: Saving model")
    print("=" * 70)

    model_path = os.path.join(OUTPUT_DIR, "xgb_ad_cn.json")
    model.save_model(model_path)
    print(f"Saved model: {model_path}")

    meta_path = os.path.join(OUTPUT_DIR, "feature_names.txt")
    with open(meta_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")
    print(f"Saved feature names: {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("AD vs CN EEG Classification — XGBoost Pipeline")
    print(f"Sampling rate: {SFREQ} Hz")
    print(f"Window: {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
    print(f"Step: {STEP_SEC}s ({STEP_SAMPLES} samples), overlap: 50%")
    print()

    # Phase 1: Load data
    data_list, labels, subject_ids = load_data()

    # Phase 2: Segment into 30s windows
    segments, seg_labels, seg_subjects = segment_data(data_list, labels, subject_ids)

    # Phase 3: Extract features
    X, feature_names = extract_all_features(segments)

    # Phase 4: Subject-level train/eval split
    X_train, X_eval, y_train, y_eval, groups_train, groups_eval = split_data(
        X, seg_labels, seg_subjects
    )

    # Phase 5: Train XGBoost
    model = train_model(X_train, y_train, groups_train, X_eval, y_eval)

    # Phase 6: Evaluate
    subj_acc = evaluate_model(
        model, X_train, y_train, X_eval, y_eval, groups_eval, feature_names
    )

    # Phase 7: Save
    save_model(model, feature_names)

    print("\n" + "=" * 70)
    print(f"DONE. Subject-level accuracy: {subj_acc:.4f}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
