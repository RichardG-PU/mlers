"""
DICE-net: Dual-Input Convolutional Encoder with Transformer.

Input per epoch:
    rbp: (B, N_WINDOWS, N_BANDS, N_CHANNELS)  = (B, 30, 5, 19)
    scc: (B, N_WINDOWS, N_BANDS, N_CHANNELS)

Architecture:
    CNNBranch × 2  (separate weights for RBP and SCC)
    → concat → Linear projection → positional embedding
    → TransformerEncoder
    → global avg pool → classification head
    → raw logit  (use BCEWithLogitsLoss, no Sigmoid here)
"""

import torch
import torch.nn as nn

import src.config as cfg


class CNNBranch(nn.Module):
    """
    3-D CNN that maps (B, 30, 5, 19) → (B, 30, BRANCH_DIM).

    We treat the input as a single-channel 3-D volume:
        dim0 = time windows (30)
        dim1 = frequency bands (5)
        dim2 = EEG channels (19)
    """

    def __init__(self):
        super().__init__()
        f = cfg.CNN_FILTERS   # [32, 64, 128]

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, f[0], kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(f[0]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(f[0], f[1], kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(f[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)),  # ch: 19→9
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(f[1], f[2], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(f[2]),
            nn.ReLU(inplace=True),
        )
        # Pool band dim 5→2 and channel dim 9→4 with fixed kernel (MPS-compatible).
        # After conv2 MaxPool: (B, 64, 30, 5, 9); after conv3: (B, 128, 30, 5, 9)
        # AvgPool3d(k=(1,2,3), stride=(1,2,2)): time=30→30, bands=5→2, ch=9→4
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 3), stride=(1, 2, 2))
        # → (B, 128, 30, 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, 5, 19)
        x = x.unsqueeze(1)          # (B, 1, 30, 5, 19)
        x = self.conv1(x)           # (B, 32, 30, 5, 19)
        x = self.conv2(x)           # (B, 64, 30, 5, 9)
        x = self.conv3(x)           # (B, 128, 30, 5, 9)
        x = self.pool(x)            # (B, 128, 30, 2, 4)
        B = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B, 30, 128, 2, 4)
        x = x.view(B, cfg.N_WINDOWS, -1)             # (B, 30, 1024)
        return x


class DICENet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_rbp = CNNBranch()
        self.cnn_scc = CNNBranch()

        self.projection = nn.Sequential(
            nn.Linear(cfg.FUSED_DIM, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL),
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.N_WINDOWS, cfg.D_MODEL) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.D_FF,
            dropout=cfg.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.N_LAYERS)

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cfg.D_MODEL, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, rbp: torch.Tensor, scc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rbp, scc: (B, 30, 5, 19)
        Returns:
            logits: (B, 1) — raw, un-sigmoided
        """
        feat_rbp = self.cnn_rbp(rbp)                          # (B, 30, 1024)
        feat_scc = self.cnn_scc(scc)                          # (B, 30, 1024)
        x = torch.cat([feat_rbp, feat_scc], dim=-1)           # (B, 30, 2048)
        x = self.projection(x)                                 # (B, 30, 128)
        x = x + self.pos_embed                                 # broadcast over batch
        x = self.transformer(x)                               # (B, 30, 128)
        x = x.mean(dim=1)                                     # (B, 128)
        return self.head(x)                                   # (B, 1)
