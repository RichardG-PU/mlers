"""
Train and evaluate DICE-net for AD vs CN classification via LOOCV.

Usage:
    python scripts/train_ad_cn.py

Prerequisites:
    python scripts/precompute_features.py   (run once first)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import load_manifest
from src.train import run_loocv
from src.evaluate import print_results_table
import src.config as cfg

if __name__ == "__main__":
    print(f"Device: {cfg.DEVICE}")
    print("Loading manifest...")
    manifest = load_manifest()
    print(f"  {len(manifest)} subjects  "
          f"(AD={sum(1 for v in manifest.values() if v['label']=='A')}, "
          f"CN={sum(1 for v in manifest.values() if v['label']=='C')})\n")

    fold_results = run_loocv(manifest)
    print_results_table(fold_results)
