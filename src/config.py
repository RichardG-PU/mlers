from pathlib import Path
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR   = Path(__file__).parent
ROOT      = SRC_DIR.parent
DATA_ROOT = ROOT / "training"
CACHE_DIR = ROOT / "features_cache"
CKPT_DIR  = ROOT / "checkpoints"

TEST_DIR       = ROOT / "testing"
TEST_CACHE_DIR = ROOT / "test_features_cache"

LABEL_CSV = DATA_ROOT / "train_label_mapping.csv"

# Class directories inside training/
CLASS_DIR = {"A": "AD", "C": "CN", "F": "FTD"}

# ── Signal ────────────────────────────────────────────────────────────────────
SFREQ      = 500          # Hz
EPOCH_SEC       = 30           # seconds per epoch
EPOCH_LEN       = SFREQ * EPOCH_SEC  # 15 000 samples
EPOCH_STRIDE_SEC = 15          # 50% overlap between epochs

N_CHANNELS  = 19
N_BANDS     = 5
N_WINDOWS   = 30          # sub-windows inside one 30-s epoch

FREQ_BANDS      = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
MORLET_FREQS    = [2, 6, 10, 18, 35]
WAVELET_NAME    = "cmor1.5-1.0"

# ── Model ─────────────────────────────────────────────────────────────────────
CNN_FILTERS = [32, 64, 128]   # Conv3d out-channels per layer
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 2
D_FF        = 256
DROPOUT     = 0.1

# After AvgPool3d the spatial dims are (N_WINDOWS, pool_bands, pool_ch)
POOL_BANDS  = 2
POOL_CH     = 4
# Single-branch flattened dim: CNN_FILTERS[-1] * POOL_BANDS * POOL_CH
BRANCH_DIM  = CNN_FILTERS[-1] * POOL_BANDS * POOL_CH   # 128*2*4 = 1024
FUSED_DIM   = BRANCH_DIM * 2                            # 2048

# ── Training ──────────────────────────────────────────────────────────────────
LR           = 5e-4
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 100
PATIENCE     = 20
BATCH_SIZE   = 16
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.05

SEED = 42

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
