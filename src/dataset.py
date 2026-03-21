"""
PyTorch Dataset over precomputed feature cache.

Each item is one epoch from one subject:
    rbp: FloatTensor (N_WINDOWS, N_BANDS, N_CHANNELS)  = (30, 5, 19)
    scc: FloatTensor (N_WINDOWS, N_BANDS, N_CHANNELS)
    label: int  (1 = AD, 0 = CN)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

import src.config as cfg


class EEGDataset(Dataset):
    """
    Args:
        subject_ids: List of subject ID strings to include.
        manifest:    Dict loaded from manifest.json (key = subject_id str).
        augment:     If True, apply training-time augmentation.
    """

    LABEL_MAP = {"A": 1, "C": 0}

    def __init__(self, subject_ids: list[str], manifest: dict, augment: bool = False):
        self.augment = augment
        self.items: list[tuple[np.ndarray, np.ndarray, int]] = []

        for sid in subject_ids:
            info  = manifest[sid]
            label = self.LABEL_MAP[info["label"]]

            rbp_all = np.load(cfg.CACHE_DIR / f"{sid}_rbp.npy", mmap_mode="r")
            scc_all = np.load(cfg.CACHE_DIR / f"{sid}_scc.npy", mmap_mode="r")

            for ep_idx in range(info["n_epochs"]):
                # Copy slice out of mmap to avoid holding file handles per-item
                self.items.append((
                    np.array(rbp_all[ep_idx], dtype=np.float32),
                    np.array(scc_all[ep_idx], dtype=np.float32),
                    label,
                ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rbp, scc, label = self.items[idx]

        rbp = torch.from_numpy(rbp)   # (30, 5, 19)
        scc = torch.from_numpy(scc)

        if self.augment:
            rbp, scc = self._augment(rbp, scc)

        return rbp, scc, label

    # ── Augmentation ──────────────────────────────────────────────────────────

    @staticmethod
    def _augment(rbp: torch.Tensor, scc: torch.Tensor):
        # 1. Gaussian noise
        rbp = rbp + torch.randn_like(rbp) * 0.01
        scc = scc + torch.randn_like(scc) * 0.01

        # 2. Amplitude scaling per channel: Uniform(0.8, 1.2)
        scale = 0.8 + 0.4 * torch.rand(1, 1, rbp.shape[2])  # (1, 1, 19)
        rbp = rbp * scale
        scc = scc * scale

        # 3. Random time-window dropout: zero 1-3 sub-windows with prob 0.4
        if torch.rand(1).item() < 0.4:
            n_drop = torch.randint(1, 4, (1,)).item()
            drop_idx = torch.randperm(rbp.shape[0])[:n_drop]
            rbp[drop_idx] = 0.0
            scc[drop_idx] = 0.0

        # 4. Channel dropout (p=0.2): zero 1-3 random channels
        if torch.rand(1).item() < 0.2:
            n_ch_drop = torch.randint(1, 4, (1,)).item()
            ch_idx = torch.randperm(rbp.shape[2])[:n_ch_drop]
            rbp[:, :, ch_idx] = 0.0
            scc[:, :, ch_idx] = 0.0

        # 5. Frequency band masking (p=0.2): zero 1 random band
        if torch.rand(1).item() < 0.2:
            band_idx = torch.randint(0, rbp.shape[1], (1,)).item()
            rbp[:, band_idx, :] = 0.0
            scc[:, band_idx, :] = 0.0

        return rbp, scc
