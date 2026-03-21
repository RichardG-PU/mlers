"""
Metrics computation in pure NumPy (no scikit-learn dependency).
"""

import numpy as np


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
) -> dict:
    """
    Compute classification metrics from subject-level predictions.

    Args:
        y_true: Ground-truth binary labels (1=AD, 0=CN).
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (AD).

    Returns:
        Dict with keys: acc, sensitivity, specificity, f1, auc.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc         = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn + 1e-8)   # recall / TPR
    specificity = tn / (tn + fp + 1e-8)   # TNR
    f1          = 2 * tp / (2 * tp + fp + fn + 1e-8)
    auc         = _manual_auc(y_true, y_prob)

    return dict(
        acc=acc,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1,
        auc=auc,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def _manual_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trapezoidal AUC without scikit-learn."""
    # Sort by descending probability
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr_list, fpr_list = [0.0], [0.0]
    cum_tp = cum_fp = 0

    for label in y_true_sorted:
        if label == 1:
            cum_tp += 1
        else:
            cum_fp += 1
        tpr_list.append(cum_tp / n_pos)
        fpr_list.append(cum_fp / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    return float(np.trapz(tpr_list, fpr_list))


def print_results_table(fold_results: list[tuple]) -> None:
    """
    Print per-fold results and aggregate metrics.

    Args:
        fold_results: List of (subject_id, true_label, pred_label, mean_prob).
    """
    print("\n── Per-fold results ──────────────────────────────────────────")
    print(f"{'Subject':>8}  {'True':>5}  {'Pred':>5}  {'P(AD)':>7}  {'Correct':>7}")
    print("─" * 45)

    y_true, y_pred, y_prob = [], [], []
    for sid, true_lbl, pred_lbl, prob in fold_results:
        correct = "✓" if true_lbl == pred_lbl else "✗"
        print(f"{sid:>8}  {true_lbl:>5}  {pred_lbl:>5}  {prob:>7.3f}  {correct:>7}")
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)
        y_prob.append(prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n── Aggregate metrics ─────────────────────────────────────────")
    print(f"  Accuracy:    {metrics['acc']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}  (AD recall)")
    print(f"  Specificity: {metrics['specificity']:.3f}  (CN recall)")
    print(f"  F1 score:    {metrics['f1']:.3f}")
    print(f"  AUC:         {metrics['auc']:.3f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print("─" * 45)
    return metrics
