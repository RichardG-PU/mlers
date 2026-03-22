"""
Data loading, label mapping, windowing, and integrity checks.

IMPORTANT: Only subjects present in the CSV label mapping are used.
Folder membership is NOT treated as an authoritative label source.
"""
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as cfg


# ──────────────────────────────────────────────
# Label map from CSV
# ──────────────────────────────────────────────

def build_label_map(
    csv_path: str = cfg.LABEL_CSV,
    task: str = "ad_cn",
) -> Dict[int, int]:
    """Build {subject_id: int_label} from the authoritative CSV.

    Only subjects present in the CSV are included.

    Args:
        csv_path: path to train_label_mapping.csv
        task: 'ad_cn' keeps A and C only.

    Returns:
        dict mapping subject_id (int) -> integer label
    """
    keep_labels = {"A", "C"}
    label_map: Dict[int, int] = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            csv_lbl = row["label"].strip()
            if csv_lbl not in keep_labels:
                continue
            sid = int(row["anonymized_id"])
            label_map[sid] = cfg.CSV_LABEL_TO_INT[csv_lbl]

    return label_map


# ──────────────────────────────────────────────
# Load a single subject
# ──────────────────────────────────────────────

def _find_subject_file(data_root: str, sid: int) -> Optional[str]:
    """Locate the .npy file for a subject across class folders."""
    for folder in ("AD", "CN"):
        path = os.path.join(data_root, folder, f"{sid}.npy")
        if os.path.isfile(path):
            return path
    return None


def load_subject(
    sid: int,
    label: int,
    data_root: str = cfg.DATA_ROOT,
) -> Optional[dict]:
    """Load one subject's EEG data.

    Returns:
        dict with keys 'data' (ndarray (19, T)), 'sid', 'label',
        or None if file not found.
    """
    path = _find_subject_file(data_root, sid)
    if path is None:
        print(f"WARNING: .npy file not found for subject {sid}")
        return None

    data = np.load(path)
    if data.shape[0] != cfg.N_CHANNELS:
        print(f"WARNING: Subject {sid} has {data.shape[0]} channels, "
              f"expected {cfg.N_CHANNELS}. Skipping.")
        return None

    return {"data": data, "sid": sid, "label": label}


def load_all_subjects(
    task: str = "ad_cn",
    data_root: str = cfg.DATA_ROOT,
    csv_path: str = cfg.LABEL_CSV,
) -> List[dict]:
    """Load all subjects for a given task.

    Returns:
        list of dicts, each with 'data', 'sid', 'label'.
    """
    label_map = build_label_map(csv_path, task=task)
    subjects = []
    for sid, label in sorted(label_map.items()):
        subj = load_subject(sid, label, data_root)
        if subj is not None:
            subjects.append(subj)
    return subjects


# ──────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────

def create_windows(
    eeg_data: np.ndarray,
    sid: int,
    label: int,
    fs: int = cfg.FS,
    window_sec: int = cfg.WINDOW_SEC,
    overlap_sec: int = cfg.OVERLAP_SEC,
) -> List[dict]:
    """Slice a subject's recording into overlapping windows.

    Partial final windows are discarded (no padding).

    Returns:
        list of dicts with 'data' (19, window_samples), 'sid', 'label'
    """
    window_samples = window_sec * fs
    step_samples = (window_sec - overlap_sec) * fs
    _, total_samples = eeg_data.shape

    if total_samples < window_samples:
        print(f"WARNING: Subject {sid} has only {total_samples / fs:.1f}s "
              f"(need {window_sec}s). Skipping all windows.")
        return []

    windows: List[dict] = []
    start = 0
    while start + window_samples <= total_samples:
        win = eeg_data[:, start : start + window_samples]
        windows.append({"data": win, "sid": sid, "label": label})
        start += step_samples

    return windows


def create_all_windows(subjects: List[dict]) -> List[dict]:
    """Create windows for every subject."""
    all_windows = []
    for subj in subjects:
        wins = create_windows(subj["data"], subj["sid"], subj["label"])
        all_windows.extend(wins)
    return all_windows


# ──────────────────────────────────────────────
# Integrity checks
# ──────────────────────────────────────────────

def run_integrity_checks(
    subjects: List[dict],
    label_map: Dict[int, int],
) -> bool:
    """Run pre-training sanity checks. Returns True if all pass."""
    ok = True

    # 1. Subject count
    n_ad = sum(1 for s in subjects if s["label"] == 1)
    n_cn = sum(1 for s in subjects if s["label"] == 0)
    print(f"Subjects loaded: AD={n_ad}, CN={n_cn}, total={len(subjects)}")

    # 2. All have 19 channels
    bad_ch = [s["sid"] for s in subjects if s["data"].shape[0] != cfg.N_CHANNELS]
    if bad_ch:
        print(f"FAIL: Wrong channel count for subjects: {bad_ch}")
        ok = False

    # 3. All have enough samples for at least 1 window
    min_samples = cfg.WINDOW_SEC * cfg.FS
    short = [s["sid"] for s in subjects if s["data"].shape[1] < min_samples]
    if short:
        print(f"WARNING: Subjects too short for a single window: {short}")

    # 4. No subject ID appears twice
    sids = [s["sid"] for s in subjects]
    if len(sids) != len(set(sids)):
        print("FAIL: Duplicate subject IDs detected!")
        ok = False

    # 5. Labels match CSV
    for s in subjects:
        expected = label_map.get(s["sid"])
        if expected is not None and s["label"] != expected:
            print(f"FAIL: Subject {s['sid']} label mismatch: "
                  f"got {s['label']}, CSV says {expected}")
            ok = False

    # 6. No duplicate recordings (very unlikely, but check)
    shapes = {}
    for s in subjects:
        key = s["data"].shape
        shapes.setdefault(key, []).append(s["sid"])
    dupes = {k: v for k, v in shapes.items() if len(v) > 1}
    if dupes:
        print(f"NOTE: Subjects sharing exact shape (may be coincidence): {dupes}")

    if ok:
        print("All integrity checks PASSED.")
    return ok
