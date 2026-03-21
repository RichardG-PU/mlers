"""
Label-shuffle sanity check.

Randomly reassigns labels across subjects, then runs the same LOOCV pipeline.
If the pipeline is leak-free, accuracy should drop to ~65% (majority class rate).
If accuracy stays high, there's a bug.

Usage:
    python scripts/sanity_check.py
"""

import sys
import random
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

    # Collect all labels, shuffle them, reassign
    sids = list(manifest.keys())
    labels = [manifest[s]["label"] for s in sids]

    random.seed(123)
    random.shuffle(labels)

    shuffled_manifest = {}
    for sid, new_label in zip(sids, labels):
        shuffled_manifest[sid] = dict(manifest[sid])
        shuffled_manifest[sid]["label"] = new_label

    n_a = sum(1 for v in shuffled_manifest.values() if v["label"] == "A")
    n_c = sum(1 for v in shuffled_manifest.values() if v["label"] == "C")
    print(f"  {len(shuffled_manifest)} subjects after shuffle  (AD={n_a}, CN={n_c})")
    print("  (Labels have been randomly reassigned — expect ~65% accuracy)\n")

    fold_results = run_loocv(shuffled_manifest)
    metrics = print_results_table(fold_results)

    if metrics["acc"] > 0.80:
        print("\n*** WARNING: Accuracy is suspiciously high with shuffled labels! ***")
        print("*** This suggests data leakage in the pipeline. ***")
    else:
        print(f"\n  Shuffled accuracy = {metrics['acc']:.1%} (expected ~65%)")
        print("  Pipeline appears leak-free.")
