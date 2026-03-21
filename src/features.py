"""
Pure signal-processing functions.  No I/O, no side effects.
Ported from neuro_participans.ipynb with minor adaptations for local use.
"""

import numpy as np
from scipy.signal import welch

import src.config as cfg


def segment_recording(eeg_norm: np.ndarray) -> list[np.ndarray]:
    """
    Split a full normalized recording into non-overlapping 30-second epochs.

    Args:
        eeg_norm: (19, T) float64 array, already z-scored per channel.

    Returns:
        List of (19, EPOCH_LEN) arrays.  Tail samples that don't fill a full
        epoch are silently dropped.
    """
    epoch_len = cfg.EPOCH_LEN
    n_epochs = eeg_norm.shape[1] // epoch_len
    return [eeg_norm[:, i * epoch_len : (i + 1) * epoch_len] for i in range(n_epochs)]


def extract_features(
    eeg_data: np.ndarray,
    sfreq: int = cfg.SFREQ,
    target_time_steps: int = cfg.N_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Relative Band Power (RBP) and Spectral Coherence Connectivity
    (SCC) from a single 30-second epoch.

    Args:
        eeg_data:          (19, EPOCH_LEN) — one 30-second epoch.
        sfreq:             Sampling frequency in Hz (default 500).
        target_time_steps: Number of 1-second sub-windows (default 30).

    Returns:
        rbp: (target_time_steps, N_BANDS, N_CHANNELS)  float64
        scc: (target_time_steps, N_BANDS, N_CHANNELS)  float64
    """
    import pywt  # optional dep; imported lazily so the rest of src works without it

    n_channels, n_points = eeg_data.shape
    segment_len = n_points // target_time_steps

    # ── 1. Relative Band Power (RBP) via Welch ────────────────────────────────
    bands = cfg.FREQ_BANDS
    rbp_features = np.zeros((target_time_steps, cfg.N_BANDS, n_channels))

    for t in range(target_time_steps):
        start = t * segment_len
        end   = start + segment_len
        segment = eeg_data[:, start:end]

        freqs, psd = welch(segment, fs=sfreq, nperseg=segment_len, axis=1)

        total_power = np.sum(psd, axis=1, keepdims=True)
        total_power[total_power == 0] = 1e-10

        for b_idx, (fmin, fmax) in enumerate(bands):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.sum(psd[:, idx], axis=1)
            rbp_features[t, b_idx, :] = band_power / total_power.flatten()

    # ── 2. Spectral Coherence Connectivity (SCC) via PyWavelets CWT ───────────
    morlet_freqs = np.array(cfg.MORLET_FREQS)
    wavelet_name = cfg.WAVELET_NAME

    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * sfreq) / morlet_freqs

    # CWT coefficients: (n_channels, n_bands, n_points)
    coeffs_all = np.zeros((n_channels, len(morlet_freqs), n_points), dtype=np.complex128)
    for ch in range(n_channels):
        cwt_out, _ = pywt.cwt(eeg_data[ch], scales, wavelet_name,
                               sampling_period=1.0 / sfreq)
        coeffs_all[ch] = cwt_out   # (n_bands, n_points)

    # Aggregate per sub-window: mean absolute magnitude → (time_steps, bands, channels)
    scc_features = np.zeros((target_time_steps, len(morlet_freqs), n_channels))
    for t in range(target_time_steps):
        start = t * segment_len
        end   = start + segment_len
        window_coeffs = coeffs_all[:, :, start:end]   # (ch, bands, seg_len)
        scc_features[t] = np.mean(np.abs(window_coeffs), axis=2).T  # (bands, ch)

    return rbp_features, scc_features
