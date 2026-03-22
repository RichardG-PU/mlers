"""
Central configuration for the DICE-Net EEG classification pipeline.
All constants and hyperparameters in one place.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "training")
LABEL_CSV = os.path.join(DATA_ROOT, "train_label_mapping.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cached_features_128hz")

# ──────────────────────────────────────────────
# EEG Recording Parameters
# ──────────────────────────────────────────────
FS_ORIGINAL = 500  # Raw recording sampling rate (Hz)
FS = 128           # Target sampling rate after downsampling
N_CHANNELS = 19
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# ──────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────
WINDOW_SEC = 30       # seconds per window
OVERLAP_SEC = 15      # overlap between consecutive windows
N_SEGMENTS = 30       # 1-second segments per 30s window

# Derived
WINDOW_SAMPLES = WINDOW_SEC * FS       # 15 000
STEP_SAMPLES = (WINDOW_SEC - OVERLAP_SEC) * FS  # 7 500

# ──────────────────────────────────────────────
# Frequency Bands & Wavelet
# ──────────────────────────────────────────────
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BANDS)

MORLET_FREQS = [2, 6, 10, 18, 35]     # representative freq per band
WAVELET_NAME = "cmor1.5-1.0"          # Complex Morlet (bandwidth 1.5, center 1.0)

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
FILTER_LOW = 0.5      # Hz — bandpass lower bound
FILTER_HIGH = 45.0    # Hz — bandpass upper bound
FILTER_ORDER = 4      # Butterworth order

# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
D_INPUT = N_BANDS * N_CHANNELS   # 5 * 19 = 95
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.1
CLASSIFIER_DROPOUT = 0.3
N_CLASSES = 2

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15          # early-stopping patience (epochs)
GRAD_CLIP = 1.0        # max gradient norm
SEED = 42

# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
N_SPLITS = 5           # for StratifiedGroupKFold during development

# ──────────────────────────────────────────────
# Label Encoding
# ──────────────────────────────────────────────
CSV_LABEL_TO_INT = {"C": 0, "A": 1}   # CN=0, AD=1
INT_TO_LABEL = {0: "CN", 1: "AD"}
