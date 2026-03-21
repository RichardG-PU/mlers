"""
One-time feature precomputation script.

Usage:
    python scripts/precompute_features.py

Creates features_cache/ with one *_rbp.npy and *_scc.npy per subject, plus
manifest.json.  Takes ~5-15 minutes depending on hardware.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import precompute_all

if __name__ == "__main__":
    print("Precomputing RBP + SCC features for AD vs CN subjects...\n")
    precompute_all(label_filter=("A", "C"))
