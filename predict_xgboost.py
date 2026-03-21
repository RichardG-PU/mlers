"""
EEG AD vs CN Classifier — Inference Script
==========================================
Classifies new EEG .npy files (19 channels, 128 Hz) as AD or CN using the trained XGBoost model.

Usage:
    python predict_xgboost.py new_data_folder/ [--model output/xgb_ad_cn.json] [--features output/feature_names.txt]

- Place all .npy files to classify in a folder (e.g., new_data/)
- Each .npy file should be shaped (19, N)
- Outputs a CSV with subject_id, predicted_label, AD_probability, n_windows
"""
import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# ---- Config ----
SFREQ = 128
WINDOW_SEC = 30
STEP_SEC = 15
WINDOW_SAMPLES = WINDOW_SEC * SFREQ
STEP_SAMPLES = STEP_SEC * SFREQ
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

# ---- Feature Extraction ----
def _hjorth_params(signal_1d):
    diff1 = np.diff(signal_1d)
    diff2 = np.diff(diff1)
    activity = np.var(signal_1d)
    mobility_num = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0.0
    mobility_den = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0.0
    complexity = mobility_den / mobility_num if mobility_num > 0 else 0.0
    return activity, mobility_num, complexity

def _spectral_entropy(psd_norm):
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))

def extract_features_xgb(segment, feature_names=None):
    n_channels = segment.shape[0]
    sfreq = SFREQ
    nperseg = min(segment.shape[1], sfreq * 2)
    features = []
    for ch in range(n_channels):
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
        for band_name in BAND_NAMES:
            rbp = band_powers[band_name] / total_power
            features.append(rbp)
        eps = 1e-10
        ratios = [
            band_powers["theta"] / (band_powers["alpha"] + eps),
            band_powers["alpha"] / (band_powers["beta"] + eps),
            band_powers["theta"] / (band_powers["beta"] + eps),
            band_powers["delta"] / (band_powers["alpha"] + eps),
        ]
        features.extend(ratios)
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(skew(signal))
        features.append(kurtosis(signal))
        act, mob, comp = _hjorth_params(signal)
        features.extend([act, mob, comp])
        psd_norm = psd / total_power
        se = _spectral_entropy(psd_norm)
        features.append(se)
    if feature_names is not None and len(features) != len(feature_names):
        raise ValueError(f"Feature count mismatch: got {len(features)}, expected {len(feature_names)}")
    return np.array(features, dtype=np.float64)

def extract_all_features(eeg, feature_names=None):
    n_points = eeg.shape[1]
    segments = []
    for start in range(0, n_points - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        segment = eeg[:, start:start + WINDOW_SAMPLES]
        segments.append(segment)
    feats = [extract_features_xgb(seg, feature_names) for seg in segments]
    return np.stack(feats) if feats else np.empty((0, len(feature_names) if feature_names else 418))

# ---- Main Inference ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="Folder with .npy EEG files to classify")
    parser.add_argument("--model", default="output/xgb_ad_cn.json", help="Path to trained model")
    parser.add_argument("--features", default="output/feature_names.txt", help="Path to feature names txt")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV file")
    args = parser.parse_args()

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    # Load feature names
    with open(args.features) as f:
        feature_names = [line.strip() for line in f]
    # Classify all .npy files
    results = []
    for fname in sorted(os.listdir(args.data_folder)):
        if not fname.endswith(".npy"): continue
        sid = os.path.splitext(fname)[0]
        eeg = np.load(os.path.join(args.data_folder, fname), allow_pickle=True)
        if eeg.shape[0] != 19:
            print(f"Skipping {fname}: expected 19 channels, got {eeg.shape[0]}")
            continue
        X = extract_all_features(eeg, feature_names)
        if X.shape[0] == 0:
            print(f"Skipping {fname}: too short for one window")
            continue
        probs = model.predict_proba(X)[:, 1]
        pred_snip = (probs > 0.5).astype(int)
        ad_pct = np.mean(pred_snip)
        maj_vote = int(ad_pct > 0.5)
        label = "AD" if maj_vote == 1 else "CN"
        results.append({
            "subject_id": sid,
            "predicted_label": label,
            "AD_probability": round(np.mean(probs), 3),
            "n_windows": X.shape[0],
        })
        print(f"{sid}: {label} (AD_prob={np.mean(probs):.3f}, n_windows={X.shape[0]})")
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()
