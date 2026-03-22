"""
EEG preprocessing: downsampling, bandpass filtering, re-referencing.

All operations are per-subject to avoid cross-subject leakage.
"""
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from . import config as cfg


def downsample(
    data: np.ndarray,
    fs_from: int = cfg.FS_ORIGINAL,
    fs_to: int = cfg.FS,
) -> np.ndarray:
    """Downsample EEG from fs_from to fs_to using polyphase resampling.

    scipy.signal.resample_poly applies an internal FIR anti-aliasing
    filter before decimation, so content above fs_to/2 is attenuated.

    Args:
        data: (n_channels, n_samples) at fs_from Hz
        fs_from: original sampling rate
        fs_to: target sampling rate

    Returns:
        resampled data at fs_to Hz
    """
    from math import gcd
    g = gcd(fs_to, fs_from)
    up = fs_to // g
    down = fs_from // g
    return resample_poly(data, up, down, axis=1).astype(data.dtype)


def bandpass_filter(
    data: np.ndarray,
    fs: int = cfg.FS,
    low: float = cfg.FILTER_LOW,
    high: float = cfg.FILTER_HIGH,
    order: int = cfg.FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        data: (n_channels, n_samples)
        fs: sampling rate in Hz
        low, high: cutoff frequencies
        order: filter order

    Returns:
        filtered data, same shape
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1).astype(data.dtype)


def average_reference(data: np.ndarray) -> np.ndarray:
    """Re-reference to average of all channels.

    Subtracts the mean across channels at each timepoint.
    """
    return data - data.mean(axis=0, keepdims=True)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization (per subject).

    Each channel is independently standardised to zero mean, unit variance
    using only that subject's data — no cross-subject leakage.
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero for flat channels
    return (data - mean) / std


def preprocess_subject(
    data: np.ndarray,
    fs_original: int = cfg.FS_ORIGINAL,
    fs_target: int = cfg.FS,
    do_reference: bool = True,
    do_zscore: bool = False,
) -> np.ndarray:
    """Apply the full preprocessing pipeline to one subject.

    Steps (in order):
        1. Downsample 500 → 128 Hz  (anti-aliased via resample_poly)
        2. Average re-reference      (recommended)
        3. Bandpass 0.5–45 Hz        (required)
        4. Z-score per channel        (optional)

    Args:
        data: raw EEG, shape (19, T) at fs_original Hz
        fs_original: original sampling rate (500)
        fs_target: target sampling rate (128)
        do_reference: apply average re-referencing
        do_zscore: apply per-channel z-score

    Returns:
        preprocessed data at fs_target Hz
    """
    # Step 1: Downsample
    if fs_original != fs_target:
        data = downsample(data, fs_original, fs_target)
    if do_reference:
        data = average_reference(data)
    data = bandpass_filter(data, fs_target)
    if do_zscore:
        data = zscore_normalize(data)
    return data
