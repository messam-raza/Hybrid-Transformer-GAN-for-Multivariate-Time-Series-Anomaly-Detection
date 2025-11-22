"""
Visualization Utilities

Provides functions for visualizing training progress, anomaly detection results,
and model performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(history, save_dir, machine_id):
    """
    Plot training loss curves.
    
    Args:
        history (dict): Training history containing losses
        save_dir (str): Directory to save plots
        machine_id (str): Machine identifier for filename
    """
    train_losses = history['train_losses']
    
    if len(train_losses) == 0:
        print("No training history to plot")
        return
    
    # Extract losses
    epochs = list(range(1, len(train_losses) + 1))
    g_losses = [loss['g_loss'] for loss in train_losses]
    d_losses = [loss['d_loss'] for loss in train_losses]
    rec_losses = [loss['rec_loss'] for loss in train_losses]
    cont_losses = [loss['cont_loss'] for loss in train_losses]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator vs Discriminator Loss
    axes[0, 0].plot(epochs, g_losses, label='Generator', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, d_losses, label='Discriminator', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator vs Discriminator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(epochs, rec_losses, color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contrastive Loss
    axes[1, 0].plot(epochs, cont_losses, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Contrastive Loss')
    axes[1, 0].set_title('Contrastive Loss Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Real vs Fake Scores
    real_scores = [loss['real_score'] for loss in train_losses]
    fake_scores = [loss['fake_score'] for loss in train_losses]
    axes[1, 1].plot(epochs, real_scores, label='Real Score', color='blue', linewidth=2)
    axes[1, 1].plot(epochs, fake_scores, label='Fake Score', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Discriminator Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'{machine_id}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_validation_metrics(history, save_dir, machine_id):
    """
    Plot validation metrics over epochs.
    
    Args:
        history (dict): Training history containing validation metrics
        save_dir (str): Directory to save plots
        machine_id (str): Machine identifier
    """
    val_metrics = history.get('val_metrics', [])
    
    if len(val_metrics) == 0:
        print("No validation metrics to plot")
        return
    
    # Extract metrics
    epochs = [i * 5 for i in range(1, len(val_metrics) + 1)]  # Validated every 5 epochs
    precisions = [m['precision'] for m in val_metrics]
    recalls = [m['recall'] for m in val_metrics]
    f1_scores = [m['f1'] for m in val_metrics]
    auc_rocs = [m['auc_roc'] for m in val_metrics]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision, Recall, F1
    axes[0, 0].plot(epochs, precisions, label='Precision', marker='o', linewidth=2)
    axes[0, 0].plot(epochs, recalls, label='Recall', marker='s', linewidth=2)
    axes[0, 0].plot(epochs, f1_scores, label='F1-Score', marker='^', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Classification Metrics')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # F1-Score trend
    axes[0, 1].plot(epochs, f1_scores, color='red', marker='o', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score Trend')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # AUC-ROC trend
    axes[1, 0].plot(epochs, auc_rocs, color='green', marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('AUC-ROC Trend')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Combined view
    axes[1, 1].plot(epochs, f1_scores, label='F1-Score', marker='o', linewidth=2)
    axes[1, 1].plot(epochs, auc_rocs, label='AUC-ROC', marker='s', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Overall Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f'{machine_id}_validation_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved validation metrics to {save_path}")
    plt.close()


def plot_anomaly_detection(scores, labels, save_dir, machine_id):
    """
    Plot anomaly detection results.
    
    Args:
        scores (np.ndarray): Anomaly scores
        labels (np.ndarray): True labels
        save_dir (str): Directory to save plots
        machine_id (str): Machine identifier
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Anomaly scores with true labels
    time_steps = np.arange(len(scores))
    axes[0].plot(time_steps, scores, color='blue', alpha=0.6, linewidth=1)
    
    # Highlight anomalous regions
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) > 0:
        axes[0].scatter(anomaly_indices, scores[anomaly_indices], 
                       color='red', s=10, alpha=0.5, label='True Anomalies')
    
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title('Anomaly Scores Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Score distribution by class
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    axes[1].hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    axes[1].hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Score Distribution by Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    ap_score = average_precision_score(labels, scores)
    
    axes[2].plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.3f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f'{machine_id}_anomaly_detection.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved anomaly detection plot to {save_path}")
    plt.close()


def plot_roc_curve(labels, scores, save_dir, machine_id):
    """
    Plot ROC curve.
    
    Args:
        labels (np.ndarray): True labels
        scores (np.ndarray): Anomaly scores
        save_dir (str): Directory to save plot
        machine_id (str): Machine identifier
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'{machine_id}_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {save_path}")
    plt.close()


def plot_confusion_matrix(labels, predictions, save_dir, machine_id):
    """
    Plot confusion matrix.
    
    Args:
        labels (np.ndarray): True labels
        predictions (np.ndarray): Predicted labels
        save_dir (str): Directory to save plot
        machine_id (str): Machine identifier
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(save_dir, f'{machine_id}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_sample_reconstruction(original, reconstructed, save_dir, machine_id, n_samples=3):
    """
    Plot original vs reconstructed time series samples.
    
    Args:
        original (np.ndarray): Original data (n_samples, window_size, n_features)
        reconstructed (np.ndarray): Reconstructed data
        save_dir (str): Directory to save plot
        machine_id (str): Machine identifier
        n_samples (int): Number of samples to plot
    """
    n_samples = min(n_samples, original.shape[0])
    n_features = min(3, original.shape[2])  # Plot up to 3 features
    
    fig, axes = plt.subplots(n_samples, n_features, figsize=(15, 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        for j in range(n_features):
            axes[i, j].plot(original[i, :, j], label='Original', linewidth=2)
            axes[i, j].plot(reconstructed[i, :, j], label='Reconstructed', 
                          linewidth=2, linestyle='--')
            axes[i, j].set_title(f'Sample {i+1}, Feature {j+1}')
            axes[i, j].legend()
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{machine_id}_reconstruction_samples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction samples to {save_path}")
    plt.close()


def plot_attention_weights(attention_weights, save_dir, machine_id):
    """
    Plot attention weight heatmap.
    
    Args:
        attention_weights (np.ndarray): Attention weights (n_heads, seq_len, seq_len)
        save_dir (str): Directory to save plot
        machine_id (str): Machine identifier
    """
    n_heads = min(4, attention_weights.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_heads):
        sns.heatmap(attention_weights[i], cmap='viridis', ax=axes[i], 
                   cbar=True, square=True)
        axes[i].set_title(f'Attention Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{machine_id}_attention_weights.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention weights to {save_path}")
    plt.close()


def create_summary_report(metrics, save_dir, machine_id):
    """
    Create a visual summary report of all metrics.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        save_dir (str): Directory to save report
        machine_id (str): Machine identifier
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Create text summary
    summary_text = f"""
    Anomaly Detection Performance Report
    =====================================
    Machine: {machine_id}
    
    Standard Metrics:
    -----------------
    Precision:     {metrics['precision']:.4f}
    Recall:        {metrics['recall']:.4f}
    F1-Score:      {metrics['f1']:.4f}
    Specificity:   {metrics.get('specificity', 0):.4f}
    
    AUC Metrics:
    ------------
    AUC-ROC:       {metrics['auc_roc']:.4f}
    AUC-PR:        {metrics['auc_pr']:.4f}
    
    Confusion Matrix:
    -----------------
    True Positives:  {metrics.get('tp', 0)}
    False Positives: {metrics.get('fp', 0)}
    True Negatives:  {metrics.get('tn', 0)}
    False Negatives: {metrics.get('fn', 0)}
    
    Threshold: {metrics['threshold']:.4f}
    """
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')
    plt.axis('off')
    
    save_path = os.path.join(save_dir, f'{machine_id}_summary_report.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary report to {save_path}")
    plt.close()