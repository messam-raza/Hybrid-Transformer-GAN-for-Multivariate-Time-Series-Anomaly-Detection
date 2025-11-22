"""
Transformer-GAN with Contrastive Learning for Time Series Anomaly Detection
Main Training Script

This script implements a comprehensive framework combining:
- Geometric masking for data augmentation
- Transformer-based autoencoder for reconstruction
- GAN framework for handling training data contamination
- Contrastive learning for distinguishing normal and anomalous patterns

Author: [Your Name]
Date: November 2024
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import json
from datetime import datetime

from models.transformer_gan import TransformerGAN
from data.data_loader import SMDDataset, create_data_loaders
from utils.metrics import compute_metrics, adjust_predictions
from utils.visualization import plot_training_curves, plot_anomaly_detection
from utils.logger import setup_logger

def parse_arguments():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training parameters
    """
    parser = argparse.ArgumentParser(description='Train Transformer-GAN for Anomaly Detection')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/SMD',
                        help='Path to SMD dataset')
    parser.add_argument('--machine_id', type=str, default='machine-1-1',
                        help='Machine ID to use from SMD dataset')
    parser.add_argument('--window_size', type=int, default=100,
                        help='Sliding window size for time series segmentation')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    
    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimension of transformer model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam optimizer')
    
    # Loss weights
    parser.add_argument('--lambda_rec', type=float, default=50.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='Weight for adversarial loss')
    parser.add_argument('--lambda_cont', type=float, default=0.5,
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss')
    
    # Masking parameters
    parser.add_argument('--mask_prob', type=float, default=0.15,
                        help='Probability of masking a time point')
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help='Ratio of masked regions to apply')
    
    # Training strategy
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for saving logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log training stats every N batches')
    
    return parser.parse_args()

def set_seed(seed):
    """
    Set random seed for reproducibility across multiple libraries.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, optimizer_g, optimizer_d, args, epoch, logger):
    """
    Train the model for one epoch.
    
    Args:
        model: TransformerGAN model
        train_loader: DataLoader for training data
        optimizer_g: Optimizer for generator
        optimizer_d: Optimizer for discriminator
        args: Training arguments
        epoch: Current epoch number
        logger: Logger instance
        
    Returns:
        dict: Dictionary containing average losses for the epoch
    """
    model.train()
    
    # Initialize loss accumulators
    epoch_losses = {
        'g_loss': 0.0,
        'd_loss': 0.0,
        'rec_loss': 0.0,
        'adv_loss': 0.0,
        'cont_loss': 0.0,
        'real_score': 0.0,
        'fake_score': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(args.device)
        batch_size = data.size(0)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        for _ in range(args.n_critic):
            optimizer_d.zero_grad()
            
            # Forward pass through discriminator
            d_loss, real_score, fake_score = model.discriminator_loss(data)
            
            # Backward pass
            d_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), args.grad_clip)
            
            optimizer_d.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        optimizer_g.zero_grad()
        
        # Forward pass through generator
        g_loss, rec_loss, adv_loss, cont_loss = model.generator_loss(
            data,
            lambda_rec=args.lambda_rec,
            lambda_adv=args.lambda_adv,
            lambda_cont=args.lambda_cont,
            temperature=args.temperature
        )
        
        # Backward pass
        g_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), args.grad_clip)
        
        optimizer_g.step()
        
        # Accumulate losses
        epoch_losses['g_loss'] += g_loss.item()
        epoch_losses['d_loss'] += d_loss.item()
        epoch_losses['rec_loss'] += rec_loss.item()
        epoch_losses['adv_loss'] += adv_loss.item()
        epoch_losses['cont_loss'] += cont_loss.item()
        epoch_losses['real_score'] += real_score.item()
        epoch_losses['fake_score'] += fake_score.item()
        
        # Update progress bar
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'Rec': f'{rec_loss.item():.4f}',
                'Cont': f'{cont_loss.item():.4f}'
            })
    
    # Calculate average losses
    num_batches = len(train_loader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses

def validate(model, val_loader, args):
    """
    Validate the model on validation set.
    
    Args:
        model: TransformerGAN model
        val_loader: DataLoader for validation data
        args: Training arguments
        
    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc='Validating'):
            data = data.to(args.device)
            
            # Get anomaly scores
            scores = model.get_anomaly_scores(data)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_scores)
    
    return metrics, all_scores, all_labels

def save_checkpoint(model, optimizer_g, optimizer_d, epoch, args, metrics, filename):
    """
    Save model checkpoint.
    
    Args:
        model: TransformerGAN model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        epoch: Current epoch
        args: Training arguments
        metrics: Validation metrics
        filename: Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }
    
    torch.save(checkpoint, filename)

def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    logger.info(f"Training configuration: {json.dumps(vars(args), indent=2)}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, n_features = create_data_loaders(
        data_path=args.data_path,
        machine_id=args.machine_id,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_prob=args.mask_prob,
        mask_ratio=args.mask_ratio
    )
    logger.info(f"Data loaded. Number of features: {n_features}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerGAN(
        n_features=n_features,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizers
    optimizer_g = torch.optim.Adam(
        model.generator.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2)
    )
    
    # Training history
    history = {
        'train_losses': [],
        'val_metrics': []
    }
    
    best_f1 = 0.0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        epoch_losses = train_epoch(
            model, train_loader, optimizer_g, optimizer_d, args, epoch, logger
        )
        
        # Log training losses
        logger.info(f"\nEpoch {epoch}/{args.epochs} Summary:")
        logger.info(f"  Generator Loss: {epoch_losses['g_loss']:.4f}")
        logger.info(f"  Discriminator Loss: {epoch_losses['d_loss']:.4f}")
        logger.info(f"  Reconstruction Loss: {epoch_losses['rec_loss']:.4f}")
        logger.info(f"  Contrastive Loss: {epoch_losses['cont_loss']:.4f}")
        logger.info(f"  Real Score: {epoch_losses['real_score']:.4f}")
        logger.info(f"  Fake Score: {epoch_losses['fake_score']:.4f}")
        
        history['train_losses'].append(epoch_losses)
        
        # Validate
        if epoch % 5 == 0:
            logger.info("Validating...")
            val_metrics, val_scores, val_labels = validate(model, val_loader, args)
            
            logger.info(f"  Validation Metrics:")
            logger.info(f"    Precision: {val_metrics['precision']:.4f}")
            logger.info(f"    Recall: {val_metrics['recall']:.4f}")
            logger.info(f"    F1-Score: {val_metrics['f1']:.4f}")
            logger.info(f"    AUC-ROC: {val_metrics['auc_roc']:.4f}")
            logger.info(f"    AUC-PR: {val_metrics['auc_pr']:.4f}")
            
            history['val_metrics'].append(val_metrics)
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_checkpoint = os.path.join(
                    args.checkpoint_dir,
                    f"{args.machine_id}_best.pth"
                )
                save_checkpoint(
                    model, optimizer_g, optimizer_d, epoch, args, val_metrics, best_checkpoint
                )
                logger.info(f"  Saved best model with F1: {best_f1:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"{args.machine_id}_epoch_{epoch}.pth"
            )
            save_checkpoint(
                model, optimizer_g, optimizer_d, epoch, args, {}, checkpoint_path
            )
            logger.info(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_metrics, test_scores, test_labels = validate(model, test_loader, args)
    
    logger.info("Test Set Results:")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR: {test_metrics['auc_pr']:.4f}")
    
    # Save training history
    history_path = os.path.join(args.log_dir, f"{args.machine_id}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    logger.info("Generating plots...")
    plot_training_curves(history, args.log_dir, args.machine_id)
    plot_anomaly_detection(
        test_scores, test_labels, args.log_dir, args.machine_id
    )
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()