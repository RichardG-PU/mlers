"""
AD vs CN EEG Classification using XGBoost — v3 (Enhanced Features + Optuna)
=============================================================================
Improvements over v2:
  1. Nonlinear complexity features per channel:
       - Sample entropy  (signal irregularity)
       - Permutation entropy (ordinal pattern complexity)
       - Detrended Fluctuation Analysis (long-range temporal correlations)
  2. Inter-channel spectral coherence (171 channel pairs × 5 bands = 855 features)
       Captures AD's hallmark reduction in long-range brain synchronization
  3. Optuna Bayesian hyperparameter tuning (50 trials, 5-fold GroupKFold)
       Replaces manually-set XGBoost hyperparameters

Feature count: 22×19 (existing) + 3×19 (nonlinear) + 171×5 (coherence) = 1330

Dependencies (pip install if missing):
  pip install antropy optuna
"""

import itertools
import json
import os
import warnings

import antropy as ant
import numpy as np
import optuna
import pandas as pd
from scipy.signal import coherence, welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "training")
LABEL_CSV    = os.path.join(TRAINING_DIR, "train_label_mapping.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")

SFREQ          = 128
WINDOW_SEC     = 30
STEP_SEC       = 15
WINDOW_SAMPLES = WINDOW_SEC * SFREQ   # 3840
STEP_SAMPLES   = STEP_SEC  * SFREQ   # 1920

RANDOM_STATE = 42
OPTUNA_TRIALS = 50

BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 25),
    "gamma": (25, 45),
}
BAND_NAMES = list(BANDS.keys())

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",
    "T3",  "C3",  "Cz",  "C4",  "T4",
    "T5",  "P3",  "Pz",  "P4",  "T6",
    "O1",  "O2",
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

    data_list  = []
    labels     = []
    subject_ids = []

    for _, row in df.iterrows():
        sid    = row["anonymized_id"]
        label  = 1 if row["label"] == "A" else 0
        folder = "AD" if row["label"] == "A" else "CN"
        path   = os.path.join(TRAINING_DIR, folder, f"{sid}.npy")

        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping subject {sid}")
            continue

        eeg = np.load(path, allow_pickle=True)
        assert eeg.shape[0] == 19, \
            f"Expected 19 channels, got {eeg.shape[0]} for subject {sid}"
        assert eeg.shape[1] >= WINDOW_SAMPLES, \
            f"Subject {sid}: recording too short ({eeg.shape[1]/SFREQ:.1f}s < {WINDOW_SEC}s)"

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

    segments    = []
    seg_labels  = []
    seg_subjects = []

    for eeg, label, sid in zip(data_list, labels, subject_ids):
        n_points  = eeg.shape[1]
        n_windows = 0
        for start in range(0, n_points - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            segments.append(eeg[:, start:start + WINDOW_SAMPLES])
            seg_labels.append(label)
            seg_subjects.append(sid)
            n_windows += 1

    seg_labels   = np.array(seg_labels)
    seg_subjects = np.array(seg_subjects)

    print(f"Total snippets: {len(segments)}")
    print(f"  AD snippets: {(seg_labels == 1).sum()}")
    print(f"  CN snippets: {(seg_labels == 0).sum()}")
    return segments, seg_labels, seg_subjects


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FEATURE EXTRACTION (1330 features per segment)
# ─────────────────────────────────────────────────────────────────────────────
def _hjorth_params(signal_1d):
    diff1    = np.diff(signal_1d)
    diff2    = np.diff(diff1)
    activity = np.var(signal_1d)
    mob_num  = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0.0
    mob_den  = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0.0
    complexity = mob_den / mob_num if mob_num > 0 else 0.0
    return activity, mob_num, complexity


def _spectral_entropy(psd_norm):
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))


def extract_features_v3(segment, sfreq=SFREQ):
    """
    Extract 1330 features from a single 30-second EEG segment.

    Block 1 — Per-channel (22 × 19 = 418):
        5 absolute band powers, 5 relative band powers,
        4 band-power ratios, mean/std/skew/kurt,
        3 Hjorth params, spectral entropy

    Block 2 — Nonlinear per-channel (3 × 19 = 57):
        sample entropy, permutation entropy, DFA

    Block 3 — Inter-channel coherence (171 pairs × 5 bands = 855):
        mean spectral coherence in each band for every channel pair
    """
    n_channels = segment.shape[0]
    nperseg    = min(segment.shape[1], sfreq * 2)
    eps        = 1e-10

    features     = []
    feature_names = []

    # ── Block 1 & 2: per-channel ──────────────────────────────────────────
    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch{ch}"
        signal  = segment[ch]

        freqs, psd  = welch(signal, fs=sfreq, nperseg=nperseg)
        total_power = np.sum(psd) or eps

        # Absolute band power
        band_powers = {}
        for band_name, (fmin, fmax) in BANDS.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            bp  = np.sum(psd[idx])
            band_powers[band_name] = bp
            features.append(bp)
            feature_names.append(f"{ch_name}_abp_{band_name}")

        # Relative band power
        for band_name in BAND_NAMES:
            features.append(band_powers[band_name] / total_power)
            feature_names.append(f"{ch_name}_rbp_{band_name}")

        # Band-power ratios
        for rname, rval in [
            ("theta_alpha", band_powers["theta"] / (band_powers["alpha"] + eps)),
            ("alpha_beta",  band_powers["alpha"] / (band_powers["beta"]  + eps)),
            ("theta_beta",  band_powers["theta"] / (band_powers["beta"]  + eps)),
            ("delta_alpha", band_powers["delta"] / (band_powers["alpha"] + eps)),
        ]:
            features.append(rval)
            feature_names.append(f"{ch_name}_ratio_{rname}")

        # Statistical moments
        features.extend([np.mean(signal), np.std(signal), skew(signal), kurtosis(signal)])
        feature_names.extend([f"{ch_name}_mean", f"{ch_name}_std",
                               f"{ch_name}_skew", f"{ch_name}_kurt"])

        # Hjorth parameters
        act, mob, comp = _hjorth_params(signal)
        features.extend([act, mob, comp])
        feature_names.extend([f"{ch_name}_hjorth_act", f"{ch_name}_hjorth_mob",
                               f"{ch_name}_hjorth_comp"])

        # Spectral entropy
        features.append(_spectral_entropy(psd / total_power))
        feature_names.append(f"{ch_name}_spec_entropy")

        # ── Block 2: nonlinear complexity ──
        features.append(ant.sample_entropy(signal))
        feature_names.append(f"{ch_name}_samp_entropy")

        features.append(ant.perm_entropy(signal, normalize=True))
        feature_names.append(f"{ch_name}_perm_entropy")

        features.append(ant.detrended_fluctuation(signal))
        feature_names.append(f"{ch_name}_dfa")

    # ── Block 3: inter-channel coherence ─────────────────────────────────
    for ch1, ch2 in itertools.combinations(range(n_channels), 2):
        n1 = CHANNEL_NAMES[ch1] if ch1 < len(CHANNEL_NAMES) else f"Ch{ch1}"
        n2 = CHANNEL_NAMES[ch2] if ch2 < len(CHANNEL_NAMES) else f"Ch{ch2}"
        f_coh, Cxy = coherence(segment[ch1], segment[ch2], fs=sfreq, nperseg=nperseg)
        for band_name, (fmin, fmax) in BANDS.items():
            idx = np.logical_and(f_coh >= fmin, f_coh <= fmax)
            features.append(Cxy[idx].mean() if idx.any() else 0.0)
            feature_names.append(f"{n1}_{n2}_coh_{band_name}")

    return np.array(features, dtype=np.float64), feature_names


def extract_all_features(segments):
    """Extract features from all segments."""
    print("\n" + "=" * 70)
    print("PHASE 3: Feature extraction (1330 features per segment)")
    print("=" * 70)

    X_list       = []
    feature_names = None

    for i, seg in enumerate(segments):
        feats, names = extract_features_v3(seg)
        X_list.append(feats)
        if feature_names is None:
            feature_names = names
        if (i + 1) % 100 == 0 or (i + 1) == len(segments):
            print(f"  Extracted features for {i + 1}/{len(segments)} snippets")

    X = np.array(X_list)

    # Replace any NaN/Inf with 0 (can arise from flat channels in entropy)
    bad = ~np.isfinite(X)
    if bad.any():
        print(f"  WARNING: {bad.sum()} non-finite values replaced with 0")
        X[bad] = 0.0

    print(f"Feature matrix shape: {X.shape}")
    return X, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3.5 — OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_hyperparams(X, y, groups, n_trials=OPTUNA_TRIALS):
    """
    Bayesian hyperparameter search using Optuna + 5-fold GroupKFold.

    Optimises mean snippet-level AUC-ROC across folds.
    The tuned params are then used in LOSO-CV and final model training,
    replacing the manually-set defaults from v2.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 3.5: Optuna hyperparameter tuning ({n_trials} trials)")
    print("=" * 70)

    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth",        3,    6),
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3,   15),
            "subsample":        trial.suggest_float("subsample",      0.5,  0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_lambda":       trial.suggest_float("reg_lambda",     1.0, 10.0),
            "gamma":            trial.suggest_float("gamma",          0.1,  2.0),
        }
        aucs = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            n_pos = (y[tr_idx] == 1).sum()
            n_neg = (y[tr_idx] == 0).sum()
            spw   = n_neg / n_pos if n_pos > 0 else 1.0
            model = XGBClassifier(
                n_estimators=2000,
                early_stopping_rounds=50,
                scale_pos_weight=spw,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
                **params,
            )
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[va_idx], y[va_idx])],
                verbose=False,
            )
            probs = model.predict_proba(X[va_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(y[va_idx], probs))
            except ValueError:
                aucs.append(0.5)
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest GroupKFold AUC: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Save for reproducibility
    out_path = os.path.join(OUTPUT_DIR, "optuna_best_params.json")
    with open(out_path, "w") as f:
        json.dump({"best_auc": study.best_value, "params": best}, f, indent=2)
    print(f"Saved: optuna_best_params.json")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# CORE MODEL — build a fresh XGBoost, optionally with tuned hyperparams
# ─────────────────────────────────────────────────────────────────────────────
def _make_model(scale_pos_weight=1.0, **kwargs):
    """Return an XGBClassifier.

    Default hyperparameters are the v2 conservative values.
    Any key in kwargs overrides the corresponding default (used with Optuna params).
    """
    defaults = dict(
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=7,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=5,
        gamma=0.5,
    )
    defaults.update(kwargs)
    return XGBClassifier(
        n_estimators=2000,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        **defaults,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def loso_cv(X, y, groups, feature_names, best_params=None):
    """
    Leave-One-Subject-Out CV using tuned hyperparameters.
    Identical LOSO structure to v2; only the model hyperparams change.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Leave-One-Subject-Out Cross-Validation")
    print("=" * 70)

    if best_params:
        print(f"Using Optuna-tuned hyperparameters.")
    else:
        print("Using default (v2) hyperparameters.")

    logo         = LeaveOneGroupOut()
    n_subjects   = len(np.unique(groups))
    subject_results  = []
    all_snippet_preds = np.zeros(len(y))
    all_snippet_probs = np.zeros(len(y))

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_sid    = groups[test_idx][0]
        true_label  = y[test_idx][0]

        X_train_fold      = X[train_idx]
        y_train_fold      = y[train_idx]
        groups_train_fold = groups[train_idx]

        # Internal early-stopping split (fresh per fold, never touches test subject)
        rng        = np.random.RandomState(RANDOM_STATE + fold_i)
        train_sids = np.unique(groups_train_fold)
        rng.shuffle(train_sids)
        n_val      = max(1, int(len(train_sids) * 0.15))
        val_sids   = set(train_sids[:n_val])
        inner_sids = set(train_sids[n_val:])

        it_mask = np.array([g in inner_sids for g in groups_train_fold])
        iv_mask = np.array([g in val_sids   for g in groups_train_fold])

        X_it, y_it = X_train_fold[it_mask], y_train_fold[it_mask]
        X_iv, y_iv = X_train_fold[iv_mask], y_train_fold[iv_mask]

        n_pos = (y_it == 1).sum()
        n_neg = (y_it == 0).sum()
        spw   = n_neg / n_pos if n_pos > 0 else 1.0

        model = _make_model(scale_pos_weight=spw, **(best_params or {}))
        model.set_params(early_stopping_rounds=50)
        model.fit(X_it, y_it, eval_set=[(X_iv, y_iv)], verbose=False)

        preds = model.predict(X[test_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        all_snippet_preds[test_idx] = preds
        all_snippet_probs[test_idx] = probs

        # Soft majority vote using mean probability
        avg_prob = float(np.mean(probs))
        vote     = int(avg_prob >= 0.5)
        ad_pct   = np.mean(preds) * 100
        correct  = vote == true_label

        true_str = "AD" if true_label == 1 else "CN"
        pred_str = "AD" if vote       == 1 else "CN"
        mark     = "OK" if correct else "WRONG"

        subject_results.append({
            "subject":       int(test_sid),
            "true":          true_str,
            "pred":          pred_str,
            "correct":       correct,
            "ad_snippet_pct": round(ad_pct, 1),
            "avg_prob":      round(avg_prob, 3),
            "n_snippets":    len(preds),
            "best_iteration": model.best_iteration,
        })

        print(f"  [{fold_i+1:2d}/{n_subjects}] Subject {int(test_sid):>3d}: "
              f"true={true_str} pred={pred_str} [{mark:>5s}]  "
              f"AD%={ad_pct:5.1f}%  prob={avg_prob:.3f}  "
              f"({len(preds)} snip, {model.best_iteration} trees)")

    # ── Summary ──
    n_correct   = sum(r["correct"] for r in subject_results)
    subj_acc    = n_correct / n_subjects
    snippet_acc = accuracy_score(y, all_snippet_preds)

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

    try:
        auc = roc_auc_score(y, all_snippet_probs)
        print(f"Snippet-level AUC-ROC: {auc:.4f}")
    except ValueError:
        auc = None
        print("AUC-ROC: could not compute")

    # Save results
    results_df = pd.DataFrame(subject_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "loso_subject_results_v3.csv"), index=False)
    print(f"\nSaved: loso_subject_results_v3.csv")

    # ── Plots ──
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
    plt.savefig(os.path.join(OUTPUT_DIR, "loso_confusion_matrix_v3.png"), dpi=150)
    plt.close()
    print("Saved: loso_confusion_matrix_v3.png")

    if auc is not None:
        fpr, tpr, _ = roc_curve(y, all_snippet_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"LOSO ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("LOSO-CV ROC Curve (Snippet-Level)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loso_roc_curve_v3.png"), dpi=150)
        plt.close()
        print("Saved: loso_roc_curve_v3.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    sids      = [r["subject"]  for r in subject_results]
    probs_list = [r["avg_prob"] for r in subject_results]
    colors = [
        ("steelblue" if r["true"] == "AD" else "seagreen") if r["correct"] else "red"
        for r in subject_results
    ]
    ax.bar(range(len(sids)), probs_list, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([str(s) for s in sids], rotation=45)
    ax.set_xlabel("Subject ID"); ax.set_ylabel("Average P(AD)")
    ax.set_title(f"LOSO-CV: Per-Subject AD Probability\n"
                 f"Blue=AD correct, Green=CN correct, Red=Wrong | Accuracy={subj_acc:.1%}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loso_subject_confidence_v3.png"), dpi=150)
    plt.close()
    print("Saved: loso_subject_confidence_v3.png")

    return subj_acc, subject_results, all_snippet_probs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — TRAIN FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(X, y, groups, feature_names, best_params=None):
    """Train the final model on ALL data using tuned hyperparameters."""
    print("\n" + "=" * 70)
    print("PHASE 5: Training final model on all data")
    print("=" * 70)

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw   = n_neg / n_pos if n_pos > 0 else 1.0

    # Determine optimal iteration count via GroupKFold
    gkf        = GroupKFold(n_splits=5)
    best_iters = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = _make_model(scale_pos_weight=spw, **(best_params or {}))
        model.set_params(early_stopping_rounds=50)
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        best_iters.append(model.best_iteration)

    avg_iter = int(np.mean(best_iters))
    print(f"GroupKFold best iterations: {best_iters} -> avg = {avg_iter}")

    final_model = _make_model(scale_pos_weight=spw, **(best_params or {}))
    final_model.set_params(n_estimators=avg_iter)
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importances = final_model.feature_importances_
    top_k   = 30
    top_idx = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), importances[top_idx][::-1], color="steelblue")
    plt.yticks(range(top_k), [feature_names[i] for i in top_idx][::-1])
    plt.xlabel("Feature Importance (Gain)")
    plt.title(f"Top {top_k} Features (Final Model v3)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_v3.png"), dpi=150)
    plt.close()
    print("Saved: feature_importance_v3.png")

    print(f"\nTop 15 features:")
    for rank, idx in enumerate(top_idx[:15]):
        print(f"  {rank+1:2d}. {feature_names[idx]:40s} importance={importances[idx]:.4f}")

    model_path = os.path.join(OUTPUT_DIR, "xgb_ad_cn_v3.json")
    final_model.save_model(model_path)
    print(f"\nSaved model: {model_path}")

    meta_path = os.path.join(OUTPUT_DIR, "feature_names_v3.txt")
    with open(meta_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")
    print(f"Saved feature names: {meta_path}")

    train_acc = accuracy_score(y, final_model.predict(X))
    print(f"\nFinal model train accuracy (sanity check only): {train_acc:.4f}")
    return final_model


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — OVERFITTING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def overfitting_analysis(X, y, groups, best_params=None):
    """5-Fold GroupKFold train-vs-val gap diagnostic."""
    print("\n" + "=" * 70)
    print("PHASE 6: Overfitting analysis (5-Fold GroupKFold)")
    print("=" * 70)

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw   = n_neg / n_pos if n_pos > 0 else 1.0

    gkf        = GroupKFold(n_splits=5)
    train_accs = []
    val_accs   = []

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = _make_model(scale_pos_weight=spw, **(best_params or {}))
        model.set_params(early_stopping_rounds=50)
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])], verbose=False)

        tr_acc = accuracy_score(y[tr_idx], model.predict(X[tr_idx]))
        va_acc = accuracy_score(y[va_idx], model.predict(X[va_idx]))
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        n_tr_subj = len(np.unique(groups[tr_idx]))
        n_va_subj = len(np.unique(groups[va_idx]))
        print(f"  Fold {fold_i+1}: train={tr_acc:.4f}  val={va_acc:.4f}  "
              f"gap={tr_acc - va_acc:.4f}  "
              f"(train={n_tr_subj} subj, val={n_va_subj} subj, {model.best_iteration} trees)")

    avg_train = np.mean(train_accs)
    avg_val   = np.mean(val_accs)
    avg_gap   = avg_train - avg_val
    print(f"\nAverage: train={avg_train:.4f}  val={avg_val:.4f}  gap={avg_gap:.4f}")

    if avg_gap > 0.20:
        print("VERDICT: SEVERE overfitting — train/val gap > 20%")
    elif avg_gap > 0.10:
        print("VERDICT: MODERATE overfitting — train/val gap 10-20%")
    else:
        print("VERDICT: LOW overfitting — train/val gap < 10%")

    fig, ax = plt.subplots(figsize=(8, 5))
    x, w = np.arange(1, 6), 0.35
    ax.bar(x - w/2, train_accs, w, label="Train",      color="steelblue")
    ax.bar(x + w/2, val_accs,   w, label="Validation", color="darkorange")
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
    ax.set_title(f"5-Fold GroupKFold: Train vs Val Accuracy\nAvg gap = {avg_gap:.4f}")
    ax.set_xticks(x); ax.legend(); ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overfitting_analysis_v3.png"), dpi=150)
    plt.close()
    print("Saved: overfitting_analysis_v3.png")
    return avg_gap


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("AD vs CN EEG Classification — XGBoost Pipeline v3")
    print(f"Sampling rate: {SFREQ} Hz | Window: {WINDOW_SEC}s | Step: {STEP_SEC}s")
    print(f"Features: 22×19 (v2) + 3×19 (nonlinear) + 171×5 (coherence) = 1330")
    print()

    # Phase 1: Load
    data_list, labels, subject_ids = load_data()

    # Phase 2: Segment
    segments, seg_labels, seg_subjects = segment_data(data_list, labels, subject_ids)

    # Phase 3: Features (1330 per segment)
    X, feature_names = extract_all_features(segments)

    # Phase 3.5: Optuna hyperparameter tuning
    best_params = tune_hyperparams(X, seg_labels, seg_subjects, n_trials=OPTUNA_TRIALS)

    # Phase 4: LOSO-CV with tuned params
    subj_acc, subject_results, _ = loso_cv(
        X, seg_labels, seg_subjects, feature_names, best_params=best_params
    )

    # Phase 5: Final model
    train_final_model(X, seg_labels, seg_subjects, feature_names, best_params=best_params)

    # Phase 6: Overfitting analysis
    avg_gap = overfitting_analysis(X, seg_labels, seg_subjects, best_params=best_params)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    n_correct = sum(r["correct"] for r in subject_results)
    n_total   = len(subject_results)
    wrong     = [r for r in subject_results if not r["correct"]]

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
