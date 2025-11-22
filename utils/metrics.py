"""
Evaluation Metrics Module

Implements comprehensive metrics for anomaly detection evaluation:
- Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Point-adjusted metrics
- Best F1 threshold selection
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)


def adjust_predictions(labels, predictions, delay=7):
    """
    Adjust predictions using point-adjusted evaluation.
    
    Point-adjusted evaluation credits early detection of anomalies
    within a tolerance window. If an anomaly is detected within 'delay'
    steps before or after the actual anomaly, it's considered a true positive.
    
    This is particularly important for time series anomaly detection where
    early warnings are valuable.
    
    Args:
        labels (np.ndarray): True binary labels
        predictions (np.ndarray): Predicted binary labels
        delay (int): Maximum delay tolerance for early detection
        
    Returns:
        tuple: (adjusted_predictions, adjusted_labels)
    """
    adjusted_predictions = predictions.copy()
    adjusted_labels = labels.copy()
    
    # Find anomaly segments
    anomaly_state = False
    anomaly_start = 0
    
    for i in range(len(labels)):
        if labels[i] == 1 and not anomaly_state:
            # Start of anomaly segment
            anomaly_state = True
            anomaly_start = i
            
            # Check if any prediction in the tolerance window
            window_start = max(0, i - delay)
            if np.any(predictions[window_start:i+1] == 1):
                # Early detection - mark entire segment as correct
                adjusted_predictions[anomaly_start:i+1] = 1
        
        elif labels[i] == 0 and anomaly_state:
            # End of anomaly segment
            anomaly_state = False
    
    return adjusted_predictions, adjusted_labels


def find_best_threshold(labels, scores):
    """
    Find the threshold that maximizes F1-score.
    
    Args:
        labels (np.ndarray): True binary labels
        scores (np.ndarray): Anomaly scores (higher = more anomalous)
        
    Returns:
        tuple: (best_threshold, best_f1)
    """
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    
    # Compute F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    
    # Get corresponding threshold
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]
    
    return best_threshold, best_f1


def compute_metrics(labels, scores, threshold=None, use_best_f1=True):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        labels (np.ndarray): True binary labels (0: normal, 1: anomaly)
        scores (np.ndarray): Anomaly scores (higher = more anomalous)
        threshold (float, optional): Threshold for binary predictions
        use_best_f1 (bool): If True, use threshold that maximizes F1
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Handle edge cases
    if len(np.unique(labels)) == 1:
        print("Warning: Labels contain only one class")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_roc': 0.5,
            'auc_pr': 0.0,
            'threshold': 0.0
        }
    
    # Find best threshold if not provided
    if threshold is None or use_best_f1:
        threshold, _ = find_best_threshold(labels, scores)
    
    # Convert scores to binary predictions
    predictions = (scores >= threshold).astype(int)
    
    # Compute basic metrics
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Compute AUC metrics
    try:
        auc_roc = roc_auc_score(labels, scores)
    except:
        auc_roc = 0.0
    
    try:
        auc_pr = average_precision_score(labels, scores)
    except:
        auc_pr = 0.0
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # Compute additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(threshold),
        'specificity': float(specificity),
        'npv': float(npv),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics


def compute_point_adjusted_metrics(labels, scores, delay=7):
    """
    Compute point-adjusted evaluation metrics.
    
    Point adjustment gives credit for early anomaly detection,
    which is important in practice.
    
    Args:
        labels (np.ndarray): True binary labels
        scores (np.ndarray): Anomaly scores
        delay (int): Tolerance window for early detection
        
    Returns:
        dict: Dictionary containing point-adjusted metrics
    """
    # Find best threshold
    threshold, _ = find_best_threshold(labels, scores)
    
    # Convert to binary predictions
    predictions = (scores >= threshold).astype(int)
    
    # Apply point adjustment
    adj_predictions, adj_labels = adjust_predictions(labels, predictions, delay)
    
    # Compute metrics on adjusted predictions
    precision = precision_score(adj_labels, adj_predictions, zero_division=0)
    recall = recall_score(adj_labels, adj_predictions, zero_division=0)
    f1 = f1_score(adj_labels, adj_predictions, zero_division=0)
    
    return {
        'pa_precision': float(precision),
        'pa_recall': float(recall),
        'pa_f1': float(f1),
        'threshold': float(threshold)
    }


def print_metrics(metrics, point_adjusted_metrics=None):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (dict): Standard metrics
        point_adjusted_metrics (dict, optional): Point-adjusted metrics
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    print(f"\nStandard Metrics (Threshold: {metrics['threshold']:.4f}):")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1-Score:     {metrics['f1']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    
    print(f"\nAUC Metrics:")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:       {metrics['auc_pr']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:   {metrics['tp']}")
    print(f"  False Positives:  {metrics['fp']}")
    print(f"  True Negatives:   {metrics['tn']}")
    print(f"  False Negatives:  {metrics['fn']}")
    
    if point_adjusted_metrics is not None:
        print(f"\nPoint-Adjusted Metrics:")
        print(f"  PA-Precision: {point_adjusted_metrics['pa_precision']:.4f}")
        print(f"  PA-Recall:    {point_adjusted_metrics['pa_recall']:.4f}")
        print(f"  PA-F1-Score:  {point_adjusted_metrics['pa_f1']:.4f}")
    
    print("="*50 + "\n")


def compute_anomaly_segments(labels):
    """
    Identify continuous anomaly segments in labels.
    
    Args:
        labels (np.ndarray): Binary labels
        
    Returns:
        list: List of tuples (start_idx, end_idx) for each anomaly segment
    """
    segments = []
    in_anomaly = False
    start_idx = 0
    
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            # Start of anomaly
            in_anomaly = True
            start_idx = i
        elif labels[i] == 0 and in_anomaly:
            # End of anomaly
            in_anomaly = False
            segments.append((start_idx, i))
    
    # Handle case where anomaly extends to end
    if in_anomaly:
        segments.append((start_idx, len(labels)))
    
    return segments


def segment_based_metrics(labels, predictions):
    """
    Compute segment-based evaluation metrics.
    
    An anomaly segment is considered detected if at least one point
    within the segment is correctly predicted as anomalous.
    
    Args:
        labels (np.ndarray): True binary labels
        predictions (np.ndarray): Predicted binary labels
        
    Returns:
        dict: Segment-based metrics
    """
    segments = compute_anomaly_segments(labels)
    
    if len(segments) == 0:
        return {
            'detected_segments': 0,
            'total_segments': 0,
            'segment_detection_rate': 0.0
        }
    
    detected = 0
    for start, end in segments:
        if np.any(predictions[start:end] == 1):
            detected += 1
    
    detection_rate = detected / len(segments)
    
    return {
        'detected_segments': detected,
        'total_segments': len(segments),
        'segment_detection_rate': float(detection_rate)
    }


def compute_all_metrics(labels, scores, delay=7):
    """
    Compute all evaluation metrics (standard, point-adjusted, and segment-based).
    
    Args:
        labels (np.ndarray): True binary labels
        scores (np.ndarray): Anomaly scores
        delay (int): Delay tolerance for point adjustment
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Standard metrics
    standard_metrics = compute_metrics(labels, scores)
    
    # Point-adjusted metrics
    pa_metrics = compute_point_adjusted_metrics(labels, scores, delay)
    
    # Segment-based metrics
    threshold = standard_metrics['threshold']
    predictions = (scores >= threshold).astype(int)
    segment_metrics = segment_based_metrics(labels, predictions)
    
    # Combine all metrics
    all_metrics = {**standard_metrics, **pa_metrics, **segment_metrics}
    
    return all_metrics