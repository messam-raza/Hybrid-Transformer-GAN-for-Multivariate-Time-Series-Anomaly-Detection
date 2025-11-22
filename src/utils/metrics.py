from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)


def evaluate_window_metrics(
    scores: np.ndarray, labels: np.ndarray, num_thresholds: int = 200
) -> dict:
    """
    Compute ROC-AUC, PR-AUC, best F1 based on threshold sweep.
    """
    assert scores.shape == labels.shape

    roc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    best_f1 = -1.0
    best_thr = thresholds[0]
    best_p = 0.0
    best_r = 0.0

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_p = p
            best_r = r

    return {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_precision": best_p,
        "best_recall": best_r,
        "best_threshold": best_thr,
    }
