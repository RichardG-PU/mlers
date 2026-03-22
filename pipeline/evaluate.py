"""
Evaluation: subject-level aggregation, metrics, and reporting.

Window-level predictions are aggregated to subject-level via
average probability (soft voting), then standard classification
metrics are computed.
"""
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config as cfg


def aggregate_subject_predictions(
    window_probs: np.ndarray,
    window_sids: np.ndarray,
    window_labels: np.ndarray,
) -> Tuple[List[int], List[int], List[int], List[np.ndarray]]:
    """Aggregate window-level softmax probabilities to subject-level.

    Method: average probability (soft voting).

    Args:
        window_probs: (N_windows, n_classes) softmax probabilities
        window_sids: (N_windows,) subject IDs
        window_labels: (N_windows,) ground-truth labels

    Returns:
        (subject_ids, subject_preds, subject_labels, subject_probs)
    """
    unique_sids = np.unique(window_sids)
    subject_ids = []
    subject_preds = []
    subject_labels = []
    subject_probs = []

    for sid in unique_sids:
        mask = window_sids == sid
        avg_prob = window_probs[mask].mean(axis=0)  # (n_classes,)
        pred = int(np.argmax(avg_prob))
        label = int(window_labels[mask][0])  # all windows share label

        subject_ids.append(int(sid))
        subject_preds.append(pred)
        subject_labels.append(label)
        subject_probs.append(avg_prob)

    return subject_ids, subject_preds, subject_labels, subject_probs


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray = None,
) -> Dict:
    """Compute subject-level classification metrics.

    Args:
        preds: (N_subjects,) predicted class
        labels: (N_subjects,) true class
        probs: (N_subjects, n_classes) optional — for ROC-AUC

    Returns:
        dict of metric name → value
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, preds)
    metrics["macro_f1"] = f1_score(labels, preds, average="macro")
    metrics["sensitivity"] = f1_score(labels, preds, average="binary",
                                       pos_label=1)  # recall for AD
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    metrics["confusion_matrix"] = cm

    # Specificity = TN / (TN + FP) = recall for CN class
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["sensitivity_raw"] = tp / max(tp + fn, 1)  # true AD recall
    else:
        metrics["specificity"] = float("nan")
        metrics["sensitivity_raw"] = float("nan")

    # ROC-AUC (requires probability scores)
    if probs is not None and len(np.unique(labels)) == 2:
        # Use probability of class 1 (AD) for binary AUC
        if isinstance(probs, list):
            probs = np.array(probs)
        if probs.ndim == 2:
            auc_probs = probs[:, 1]
        else:
            auc_probs = probs
        try:
            metrics["roc_auc"] = roc_auc_score(labels, auc_probs)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    metrics["n_subjects"] = len(labels)
    metrics["n_correct"] = int(np.sum(preds == labels))

    return metrics


def print_results(metrics: Dict) -> None:
    """Pretty-print subject-level results."""
    n = metrics["n_subjects"]
    nc = metrics["n_correct"]
    print(f"  Accuracy:         {nc}/{n} = {metrics['accuracy']:.1%}")
    print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.1%}")
    print(f"  Macro F1:         {metrics['macro_f1']:.3f}")
    print(f"  Sensitivity (AD): {metrics['sensitivity_raw']:.1%}")
    print(f"  Specificity (CN): {metrics['specificity']:.1%}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"  ROC-AUC:          {metrics['roc_auc']:.3f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Pred CN   Pred AD")
    print(f"    Actual CN      {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"    Actual AD      {cm[1,0]:3d}       {cm[1,1]:3d}")
