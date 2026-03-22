"""
XGBoost LOOCV pipeline for AD vs CN classification.

Runs Leave-One-Subject-Out cross-validation and writes
xgboost_predictions/loocv.csv for use by ensemble.py.

Usage:
    python scripts/train_xgboost.py

Prerequisites:
    python scripts/precompute_features.py   (features_cache must exist)
    pip install xgboost scikit-learn
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as cfg
from src.cache import load_manifest

OUTPUT_DIR = cfg.ROOT / "xgboost_predictions"

# Feature extraction constants — 22 features × 19 channels = 418 total
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
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]


# ── Feature extraction ────────────────────────────────────────────────────────

def _hjorth_params(signal):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0.0
    complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
                  if mobility > 0 and np.var(diff1) > 0 else 0.0)
    return activity, mobility, complexity


def _spectral_entropy(psd_norm):
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))


def extract_features(segment: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extract 418 features from a (19, EPOCH_LEN) EEG segment.
    22 features × 19 channels.
    """
    sfreq = cfg.SFREQ
    nperseg = min(segment.shape[1], sfreq * 2)
    features = []
    names = []

    for ch in range(segment.shape[0]):
        ch_name = CHANNEL_NAMES[ch]
        sig = segment[ch]
        freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
        total_power = np.sum(psd) or 1e-10

        band_powers = {}
        for bname, (flo, fhi) in BANDS.items():
            bp = np.sum(psd[np.logical_and(freqs >= flo, freqs <= fhi)])
            band_powers[bname] = bp
            features.append(bp)
            names.append(f"{ch_name}_abp_{bname}")

        for bname in BAND_NAMES:
            features.append(band_powers[bname] / total_power)
            names.append(f"{ch_name}_rbp_{bname}")

        eps = 1e-10
        for rname, rval in [
            ("theta_alpha", band_powers["theta"] / (band_powers["alpha"] + eps)),
            ("alpha_beta",  band_powers["alpha"] / (band_powers["beta"]  + eps)),
            ("theta_beta",  band_powers["theta"] / (band_powers["beta"]  + eps)),
            ("delta_alpha", band_powers["delta"] / (band_powers["alpha"] + eps)),
        ]:
            features.append(rval)
            names.append(f"{ch_name}_ratio_{rname}")

        features.extend([np.mean(sig), np.std(sig), skew(sig), kurtosis(sig)])
        names.extend([f"{ch_name}_mean", f"{ch_name}_std",
                      f"{ch_name}_skew", f"{ch_name}_kurt"])

        act, mob, comp = _hjorth_params(sig)
        features.extend([act, mob, comp])
        names.extend([f"{ch_name}_hjorth_act", f"{ch_name}_hjorth_mob",
                      f"{ch_name}_hjorth_comp"])

        features.append(_spectral_entropy(psd / total_power))
        names.append(f"{ch_name}_spec_entropy")

    return np.array(features, dtype=np.float64), names


def build_feature_matrix(manifest: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load raw recordings, segment with 50% overlap, extract features.

    Returns X (n_snippets, 418), y (n_snippets,), groups (n_snippets,), feature_names.
    """
    label_map = {"A": 1, "C": 0}
    X_rows, y_rows, g_rows = [], [], []
    feature_names = None

    for sid, meta in manifest.items():
        folder = cfg.CLASS_DIR[meta["label"]]
        path = cfg.DATA_ROOT / folder / f"{sid}.npy"
        eeg = np.load(path, allow_pickle=True).astype(np.float64)

        # Normalize (z-score per channel, matching cache.py)
        mu = eeg.mean(axis=1, keepdims=True)
        sigma = eeg.std(axis=1, keepdims=True) + 1e-8
        eeg = (eeg - mu) / sigma

        label = label_map[meta["label"]]
        step = cfg.SFREQ * (cfg.EPOCH_SEC // 2)   # 50% overlap

        for start in range(0, eeg.shape[1] - cfg.EPOCH_LEN + 1, step):
            seg = eeg[:, start: start + cfg.EPOCH_LEN]
            feats, names = extract_features(seg)
            X_rows.append(feats)
            y_rows.append(label)
            g_rows.append(sid)
            if feature_names is None:
                feature_names = names

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int32)
    groups = np.array(g_rows)
    return X, y, groups, feature_names


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
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
        random_state=cfg.SEED,
        verbosity=0,
    )


# ── LOOCV ─────────────────────────────────────────────────────────────────────

def run_loocv_xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[dict]:
    """
    Leave-One-Subject-Out cross-validation.

    Early stopping uses an internal ~15% subject holdout (never the test subject).
    Returns per-subject result dicts matching the loocv.csv schema.
    """
    logo = LeaveOneGroupOut()
    unique_sids = np.unique(groups)
    n_subjects = len(unique_sids)
    results = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_sid = groups[test_idx][0]
        true_label = y[test_idx][0]

        X_tr, y_tr = X[train_idx], y[train_idx]
        g_tr = groups[train_idx]

        # Internal early-stopping split: hold out ~15% of training subjects
        rng = np.random.RandomState(cfg.SEED + fold_i)
        train_sids = np.unique(g_tr)
        rng.shuffle(train_sids)
        n_val = max(1, int(len(train_sids) * 0.15))
        val_sids = set(train_sids[:n_val])
        inner_sids = set(train_sids[n_val:])

        it_mask = np.array([g in inner_sids for g in g_tr])
        iv_mask = np.array([g in val_sids   for g in g_tr])

        X_it, y_it = X_tr[it_mask], y_tr[it_mask]
        X_iv, y_iv = X_tr[iv_mask], y_tr[iv_mask]

        n_pos = np.sum(y_it == 1)
        n_neg = np.sum(y_it == 0)
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        model = _make_model(scale_pos_weight=spw)
        model.set_params(early_stopping_rounds=50)
        model.fit(X_it, y_it, eval_set=[(X_iv, y_iv)], verbose=False)

        preds = model.predict(X[test_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]

        vote = int(np.round(np.mean(preds)))
        ad_pct = float(np.mean(preds) * 100)
        avg_prob = float(np.mean(probs))
        correct = vote == true_label

        true_str = "AD" if true_label == 1 else "CN"
        pred_str = "AD" if vote == 1 else "CN"
        mark = "✓" if correct else "✗"

        print(f"  [{fold_i+1:2d}/{n_subjects}] Subject {test_sid:>3s}: "
              f"true={true_str} pred={pred_str} {mark}  "
              f"AD%={ad_pct:5.1f}%  prob={avg_prob:.3f}  "
              f"({len(preds)} snip, {model.best_iteration} trees)")

        results.append({
            "subject": int(test_sid),
            "true": true_str,
            "pred": pred_str,
            "correct": correct,
            "ad_snippet_pct": round(ad_pct, 1),
            "avg_prob": round(avg_prob, 3),
            "n_snippets": len(preds),
            "best_iteration": model.best_iteration,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"SFREQ: {cfg.SFREQ} Hz | Window: {cfg.EPOCH_SEC}s | Step: {cfg.EPOCH_SEC // 2}s\n")

    print("Loading manifest...")
    manifest = load_manifest()
    n_ad = sum(1 for v in manifest.values() if v["label"] == "A")
    n_cn = sum(1 for v in manifest.values() if v["label"] == "C")
    print(f"  {len(manifest)} subjects (AD={n_ad}, CN={n_cn})\n")

    print("Extracting features from raw recordings...")
    X, y, groups, feature_names = build_feature_matrix(manifest)
    print(f"  Feature matrix: {X.shape}  ({len(feature_names)} features)\n")

    assert not np.any(np.isnan(X)), "NaN in feature matrix"
    assert not np.any(np.isinf(X)), "Inf in feature matrix"

    print("=" * 60)
    print("LOOCV")
    print("=" * 60)
    results = run_loocv_xgb(X, y, groups)

    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)
    ad_results = [r for r in results if r["true"] == "AD"]
    cn_results = [r for r in results if r["true"] == "CN"]
    print(f"\nSubject accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}")
    print(f"  AD recall: {sum(r['correct'] for r in ad_results)}/{len(ad_results)}")
    print(f"  CN recall: {sum(r['correct'] for r in cn_results)}/{len(cn_results)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "loocv.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
