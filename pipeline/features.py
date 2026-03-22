"""
Feature extraction: Relative Band Power (RBP) and Spectral Coherence
Connectivity (SCC).

Both features are computed per-window and are parameter-free (no fitting),
so there is no risk of cross-subject leakage from feature extraction itself.

Output shapes per window:
    RBP: (30, 5, 19)  — [time_segments, bands, channels]
    SCC: (30, 5, 19)  — [time_segments, bands, channels]
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
from scipy.signal import welch

from . import config as cfg
from .preprocessing import preprocess_subject


# ──────────────────────────────────────────────
# Relative Band Power (RBP)
# ──────────────────────────────────────────────

def compute_rbp(
    window_data: np.ndarray,
    fs: int = cfg.FS,
    n_segments: int = cfg.N_SEGMENTS,
    bands: list = None,
) -> np.ndarray:
    """Compute Relative Band Power for a single 30-second window.

    Each 30s window is divided into 30 non-overlapping 1s segments.
    For each segment, Welch PSD is computed and the relative power in
    each frequency band is calculated.

    Args:
        window_data: (n_channels, window_samples), e.g. (19, 15000)
        fs: sampling rate
        n_segments: number of 1-second segments (30)
        bands: list of (low, high) tuples for frequency bands

    Returns:
        rbp: (n_segments, n_bands, n_channels) — values in [0, 1]
    """
    if bands is None:
        bands = cfg.BANDS

    n_channels, n_points = window_data.shape
    segment_len = n_points // n_segments  # 500 at 500 Hz
    n_bands = len(bands)

    rbp = np.zeros((n_segments, n_bands, n_channels), dtype=np.float32)

    for t in range(n_segments):
        start = t * segment_len
        end = start + segment_len
        segment = window_data[:, start:end]  # (19, 500)

        # Welch PSD: nperseg = segment_len for 1 Hz resolution
        freqs, psd = welch(segment, fs=fs, nperseg=segment_len, axis=1)

        total_power = np.sum(psd, axis=1, keepdims=True)  # (19, 1)
        total_power[total_power == 0] = 1e-10

        for b_idx, (fmin, fmax) in enumerate(bands):
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.sum(psd[:, freq_mask], axis=1)  # (19,)
            rbp[t, b_idx, :] = band_power / total_power.ravel()

    return rbp


# ──────────────────────────────────────────────
# Spectral Coherence Connectivity (SCC)
# ──────────────────────────────────────────────

def compute_scc(
    window_data: np.ndarray,
    fs: int = cfg.FS,
    n_segments: int = cfg.N_SEGMENTS,
    morlet_freqs: list = None,
    wavelet_name: str = cfg.WAVELET_NAME,
) -> np.ndarray:
    """Compute Spectral Coherence Connectivity for a single 30s window.

    Steps:
        1. CWT with complex Morlet wavelet on the full 30s window
           at 5 representative frequencies (one per band).
        2. For each 1s segment, compute pairwise coherence between
           all channel pairs at each band.
        3. Average coherence per channel → SCC value.

    Args:
        window_data: (n_channels, window_samples), e.g. (19, 15000)
        fs: sampling rate
        n_segments: number of 1-second segments
        morlet_freqs: representative frequencies per band
        wavelet_name: PyWavelets wavelet name

    Returns:
        scc: (n_segments, n_bands, n_channels) — values in [0, 1]
    """
    if morlet_freqs is None:
        morlet_freqs = cfg.MORLET_FREQS

    n_channels, n_points = window_data.shape
    segment_len = n_points // n_segments
    n_bands = len(morlet_freqs)

    # Compute wavelet scales from target frequencies
    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * fs) / np.array(morlet_freqs, dtype=float)

    # CWT for all channels — shape: (n_channels, n_bands, n_points)
    coeffs_all = np.zeros(
        (n_channels, n_bands, n_points), dtype=np.complex128
    )
    for ch in range(n_channels):
        cwt_out, _ = pywt.cwt(
            window_data[ch], scales, wavelet_name,
            sampling_period=1.0 / fs
        )
        coeffs_all[ch, :, :] = cwt_out  # (n_bands, n_points)

    # Compute coherence per segment
    scc = np.zeros((n_segments, n_bands, n_channels), dtype=np.float32)

    for t in range(n_segments):
        start = t * segment_len
        end = start + segment_len

        for b_idx in range(n_bands):
            # Coefficients for this band and time segment: (n_channels, segment_len)
            seg_coeffs = coeffs_all[:, b_idx, start:end]

            # Cross-spectral density matrix: (n_channels, n_channels)
            csd_matrix = seg_coeffs @ seg_coeffs.conj().T

            # Power spectral density: diagonal
            psd_vec = np.diag(csd_matrix).real  # (n_channels,)

            # Denominator: sqrt(PSD_i * PSD_j)
            denom = np.sqrt(np.outer(psd_vec, psd_vec))
            denom[denom == 0] = 1e-10

            # Coherence matrix: |CSD| / denom — values in [0, 1]
            coherence_matrix = np.abs(csd_matrix) / denom

            # Average coherence per channel (mean over other channels)
            scc[t, b_idx, :] = np.mean(coherence_matrix, axis=1)

    return scc


# ──────────────────────────────────────────────
# Combined extraction
# ──────────────────────────────────────────────

def extract_features(
    window_data: np.ndarray,
    fs: int = cfg.FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract both feature types for a single window.

    Returns:
        (rbp, scc) each of shape (30, 5, 19)
    """
    rbp = compute_rbp(window_data, fs)
    scc = compute_scc(window_data, fs)
    return rbp, scc


# ──────────────────────────────────────────────
# Pre-computation & caching (per subject)
# ──────────────────────────────────────────────

def precompute_subject_features(
    subject: dict,
    fs: int = cfg.FS,
    window_sec: int = cfg.WINDOW_SEC,
    overlap_sec: int = cfg.OVERLAP_SEC,
) -> Optional[dict]:
    """Preprocess one subject (incl. 500→128 Hz downsample), window, extract features.

    Returns:
        dict with 'rbp' (W,30,5,19), 'scc' (W,30,5,19),
        'label' int, 'sid' int, 'n_windows' int
        or None if subject produces 0 windows.
    """
    from .data_loading import create_windows

    # Preprocess raw signal: downsample 500→128, re-reference, bandpass
    preprocessed = preprocess_subject(subject["data"])  # uses cfg defaults

    # Create windows
    windows = create_windows(
        preprocessed, subject["sid"], subject["label"],
        fs, window_sec, overlap_sec,
    )
    if not windows:
        return None

    # Extract features for each window
    rbp_list, scc_list = [], []
    for w in windows:
        rbp, scc = extract_features(w["data"], fs)
        rbp_list.append(rbp)
        scc_list.append(scc)

    return {
        "rbp": np.stack(rbp_list),   # (W, 30, 5, 19)
        "scc": np.stack(scc_list),   # (W, 30, 5, 19)
        "label": subject["label"],
        "sid": subject["sid"],
        "n_windows": len(windows),
    }


def precompute_all_features(
    subjects: List[dict],
    cache_dir: str = cfg.CACHE_DIR,
    force: bool = False,
) -> None:
    """Extract features for all subjects and save to disk.

    Each subject is saved as cache_dir/{sid}.npz with keys:
        'rbp', 'scc', 'label', 'sid', 'n_windows'
    """
    os.makedirs(cache_dir, exist_ok=True)

    for i, subj in enumerate(subjects):
        out_path = os.path.join(cache_dir, f"{subj['sid']}.npz")
        if os.path.exists(out_path) and not force:
            print(f"  [{i+1}/{len(subjects)}] Subject {subj['sid']} — cached, skipping")
            continue

        print(f"  [{i+1}/{len(subjects)}] Subject {subj['sid']} — extracting features...")
        result = precompute_subject_features(subj)
        if result is None:
            print(f"    WARNING: Subject {subj['sid']} produced 0 windows, skipping.")
            continue

        np.savez_compressed(
            out_path,
            rbp=result["rbp"],
            scc=result["scc"],
            label=np.int64(result["label"]),
            sid=np.int64(result["sid"]),
            n_windows=np.int64(result["n_windows"]),
        )
        print(f"    Saved {result['n_windows']} windows → {out_path}")


def load_cached_features(
    subject_ids: List[int],
    cache_dir: str = cfg.CACHE_DIR,
) -> Dict[int, dict]:
    """Load pre-computed features from disk.

    Returns:
        dict {sid: {'rbp': (W,30,5,19), 'scc': (W,30,5,19), 'label': int}}
    """
    features = {}
    for sid in subject_ids:
        path = os.path.join(cache_dir, f"{sid}.npz")
        if not os.path.exists(path):
            print(f"WARNING: No cached features for subject {sid}")
            continue
        data = np.load(path)
        features[sid] = {
            "rbp": data["rbp"],
            "scc": data["scc"],
            "label": int(data["label"]),
            "sid": int(data["sid"]),
            "n_windows": int(data["n_windows"]),
        }
    return features


# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────

def sanity_check_features(rbp: np.ndarray, scc: np.ndarray, sid: int = -1) -> bool:
    """Validate feature arrays for a single window or stacked windows."""
    ok = True
    tag = f"Subject {sid}" if sid >= 0 else "Features"

    # Shape: last 3 dims should be (30, 5, 19)
    for name, arr in [("RBP", rbp), ("SCC", scc)]:
        if arr.shape[-3:] != (cfg.N_SEGMENTS, cfg.N_BANDS, cfg.N_CHANNELS):
            print(f"FAIL: {tag} {name} shape {arr.shape}, "
                  f"expected (..., {cfg.N_SEGMENTS}, {cfg.N_BANDS}, {cfg.N_CHANNELS})")
            ok = False
        if np.any(np.isnan(arr)):
            print(f"FAIL: {tag} {name} contains NaN")
            ok = False
        if np.any(np.isinf(arr)):
            print(f"FAIL: {tag} {name} contains Inf")
            ok = False

    # RBP should be in [0, 1] (relative power)
    if np.any(rbp < -1e-6) or np.any(rbp > 1.0 + 1e-6):
        print(f"WARNING: {tag} RBP values outside [0, 1]: "
              f"min={rbp.min():.4f}, max={rbp.max():.4f}")

    # SCC should be in [0, 1] (coherence)
    if np.any(scc < -1e-6) or np.any(scc > 1.0 + 1e-6):
        print(f"WARNING: {tag} SCC values outside [0, 1]: "
              f"min={scc.min():.4f}, max={scc.max():.4f}")

    return ok
