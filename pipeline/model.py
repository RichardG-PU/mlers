"""
DICE-Net V2 -- Dual-Input Classification Encoder (micro-transformer).

Single architecture: two parallel 1-layer transformers (RBP + SCC),
additive fusion, mean-pooling, ~14.8K params.

Input per branch:  (B, 30, 5, 19) -> flatten -> (B, 30, 95)
Each branch:       LayerNorm -> Linear(95->d) -> SinPE -> Transformer(1L) -> mean pool -> (B, d)
Fusion:            h_rbp + h_scc -> (B, d)
Classifier:        LayerNorm(d) -> Dropout -> Linear(d->2)
"""
import math

import torch
import torch.nn as nn

from . import config as cfg


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DICENetV2Branch(nn.Module):
    """One branch of the V2 micro-transformer.

    InputLayerNorm -> Linear projection -> SinPE -> 1-layer Transformer -> mean pool.
    No CLS token; mean-pool over the full sequence.
    """

    def __init__(
        self,
        d_input: int = cfg.D_INPUT,
        d_model: int = 24,
        nhead: int = 2,
        dim_ff: int = 48,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(d_input)
        self.proj = nn.Linear(d_input, d_model)

        self.pos_enc = SinusoidalPE(d_model, max_len=64, dropout=dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 30, 5, 19) -> (B, d_model)"""
        B = x.size(0)
        x = x.view(B, cfg.N_SEGMENTS, -1)   # (B, 30, 95)
        x = self.input_norm(x)
        x = self.proj(x)                     # (B, 30, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                  # (B, 30, d_model)
        return x.mean(dim=1)                  # mean pool -> (B, d_model)


class DICENetV2(nn.Module):
    """Dual-branch micro-transformer with additive fusion.

    Two parallel DICENetV2Branch (RBP + SCC), outputs summed,
    followed by LayerNorm -> Dropout -> Linear classifier.
    ~14.8K params with d_model=24.
    """

    def __init__(
        self,
        d_input: int = cfg.D_INPUT,
        d_model: int = 24,
        nhead: int = 2,
        dim_ff: int = 48,
        n_classes: int = cfg.N_CLASSES,
        dropout: float = 0.3,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model

        branch_kwargs = dict(
            d_input=d_input, d_model=d_model, nhead=nhead,
            dim_ff=dim_ff, dropout=dropout,
        )
        self.branch_rbp = DICENetV2Branch(**branch_kwargs)
        self.branch_scc = DICENetV2Branch(**branch_kwargs)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rbp: torch.Tensor, scc: torch.Tensor) -> torch.Tensor:
        h_rbp = self.branch_rbp(rbp)   # (B, d_model)
        h_scc = self.branch_scc(scc)   # (B, d_model)
        h = h_rbp + h_scc              # additive fusion
        return self.classifier(h)
