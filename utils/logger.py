"""
Logging Utilities

Provides comprehensive logging functionality for training and evaluation.
"""

import os
import logging
import sys
from datetime import datetime


def setup_logger(log_dir, log_file=None, level=logging.INFO):
    """
    Setup logger with both file and console handlers.
    
    Args:
        log_dir (str): Directory to save log files
        log_file (str, optional): Log filename. If None, uses timestamp
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'training_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger('TransformerGAN')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_path}")
    
    return logger


class MetricLogger:
    """
    Logger for tracking and saving metrics during training.
    """
    
    def __init__(self, log_dir, machine_id):
        """
        Initialize metric logger.
        
        Args:
            log_dir (str): Directory to save metric logs
            machine_id (str): Machine identifier
        """
        self.log_dir = log_dir
        self.machine_id = machine_id
        self.metrics_file = os.path.join(log_dir, f'{machine_id}_metrics.csv')
        
        # Initialize CSV file with headers
        with open(self.metrics_file, 'w') as f:
            f.write('epoch,g_loss,d_loss,rec_loss,adv_loss,cont_loss,')
            f.write('real_score,fake_score,')
            f.write('val_precision,val_recall,val_f1,val_auc_roc,val_auc_pr\n')
    
    def log_epoch(self, epoch, train_losses, val_metrics=None):
        """
        Log metrics for one epoch.
        
        Args:
            epoch (int): Epoch number
            train_losses (dict): Training losses
            val_metrics (dict, optional): Validation metrics
        """
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},")
            f.write(f"{train_losses['g_loss']:.6f},")
            f.write(f"{train_losses['d_loss']:.6f},")
            f.write(f"{train_losses['rec_loss']:.6f},")
            f.write(f"{train_losses['adv_loss']:.6f},")
            f.write(f"{train_losses['cont_loss']:.6f},")
            f.write(f"{train_losses['real_score']:.6f},")
            f.write(f"{train_losses['fake_score']:.6f},")
            
            if val_metrics is not None:
                f.write(f"{val_metrics['precision']:.6f},")
                f.write(f"{val_metrics['recall']:.6f},")
                f.write(f"{val_metrics['f1']:.6f},")
                f.write(f"{val_metrics['auc_roc']:.6f},")
                f.write(f"{val_metrics['auc_pr']:.6f}")
            else:
                f.write(",,,,")
            
            f.write("\n")


class TensorBoardLogger:
    """
    TensorBoard logger wrapper.
    """
    
    def __init__(self, log_dir, machine_id):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir (str): Directory for TensorBoard logs
            machine_id (str): Machine identifier
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard', machine_id))
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(tag, value_dict, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, img, step):
        """Log image."""
        if self.enabled:
            self.writer.add_image(tag, img, step)
    
    def close(self):
        """Close writer."""
        if self.enabled:
            self.writer.close()