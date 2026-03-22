"""
PyTorch Dataset for windowed EEG features.

Each sample is one window, returning (rbp_tensor, scc_tensor, label, subject_id).
Subject identity is always preserved for split-safe validation.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as cfg


class EEGWindowDataset(Dataset):
    """Dataset that serves pre-computed RBP and SCC feature windows.

    Attributes:
        rbp: (N_total_windows, 30, 5, 19) float32
        scc: (N_total_windows, 30, 5, 19) float32
        labels: (N_total_windows,) int64
        subject_ids: (N_total_windows,) int64
    """

    def __init__(
        self,
        rbp: np.ndarray,
        scc: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
    ):
        assert len(rbp) == len(scc) == len(labels) == len(subject_ids)
        self.rbp = torch.tensor(rbp, dtype=torch.float32)
        self.scc = torch.tensor(scc, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        return (
            self.rbp[idx],          # (30, 5, 19)
            self.scc[idx],          # (30, 5, 19)
            self.labels[idx],       # scalar
            self.subject_ids[idx],  # scalar
        )


def build_fold_datasets(
    features: Dict[int, dict],
    train_sids: List[int],
    val_sids: List[int],
) -> Tuple[EEGWindowDataset, EEGWindowDataset]:
    """Build train and validation datasets for one CV fold.

    All windows from a subject go entirely into train OR val — never split.

    Args:
        features: dict {sid: {'rbp': (W,30,5,19), 'scc': ..., 'label': int}}
        train_sids: subject IDs for training
        val_sids: subject IDs for validation

    Returns:
        (train_dataset, val_dataset)
    """
    # Verify no overlap
    overlap = set(train_sids) & set(val_sids)
    assert not overlap, f"LEAKAGE: subjects in both train & val: {overlap}"

    def _collect(sids):
        rbp_list, scc_list, lbl_list, sid_list = [], [], [], []
        for sid in sids:
            if sid not in features:
                continue
            f = features[sid]
            n = f["n_windows"]
            rbp_list.append(f["rbp"])
            scc_list.append(f["scc"])
            lbl_list.append(np.full(n, f["label"], dtype=np.int64))
            sid_list.append(np.full(n, f["sid"], dtype=np.int64))

        if not rbp_list:
            # Return empty dataset
            empty = np.zeros((0, cfg.N_SEGMENTS, cfg.N_BANDS, cfg.N_CHANNELS),
                             dtype=np.float32)
            return EEGWindowDataset(empty, empty,
                                    np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64))

        return EEGWindowDataset(
            np.concatenate(rbp_list),
            np.concatenate(scc_list),
            np.concatenate(lbl_list),
            np.concatenate(sid_list),
        )

    return _collect(train_sids), _collect(val_sids)
