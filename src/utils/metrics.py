from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    """
    Compute ROC-AUC given binary labels and anomaly scores.
    """
    labels_arr = np.asarray(labels).astype(int)
    scores_arr = np.asarray(scores).astype(float)

    if len(np.unique(labels_arr)) < 2:
        return float("nan")

    return float(roc_auc_score(labels_arr, scores_arr))
