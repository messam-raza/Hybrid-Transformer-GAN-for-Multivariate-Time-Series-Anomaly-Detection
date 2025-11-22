"""
Data Loading and Preprocessing Module

This module handles:
- Loading SMD (Server Machine Dataset) data
- Applying geometric masking for data augmentation
- Creating sliding windows
- Train/val/test split
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class GeometricMasking:
    """
    Geometric masking for time series data augmentation.
    
    Applies masks sampled from a geometric distribution to create
    diverse training samples and improve model robustness.
    """
    
    def __init__(self, mask_prob=0.15, mask_ratio=0.3):
        """
        Initialize geometric masking.
        
        Args:
            mask_prob (float): Probability parameter for geometric distribution
            mask_ratio (float): Ratio of sequence to potentially mask
        """
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
    
    def geometric_mask(self, length):
        """
        Generate mask indices using geometric distribution.
        
        The geometric distribution naturally creates variable-length
        consecutive masks, simulating real-world missing patterns.
        
        Args:
            length (int): Sequence length
            
        Returns:
            np.ndarray: Boolean mask array
        """
        mask = np.zeros(length, dtype=bool)
        
        # Calculate number of regions to mask
        n_masks = int(length * self.mask_ratio)
        
        if n_masks == 0:
            return mask
        
        # Sample start positions
        start_positions = np.random.choice(length, size=n_masks, replace=False)
        
        for start in start_positions:
            # Sample mask length from geometric distribution
            mask_length = np.random.geometric(self.mask_prob)
            mask_length = min(mask_length, length - start)
            
            # Apply mask
            mask[start:start + mask_length] = True
        
        return mask
    
    def apply_mask(self, data, mask=None):
        """
        Apply mask to data.
        
        Masked values are replaced with zeros (or could use mean/noise).
        
        Args:
            data (np.ndarray): Time series data of shape (seq_len, n_features)
            mask (np.ndarray, optional): Pre-computed mask
            
        Returns:
            tuple: (masked_data, mask)
        """
        if mask is None:
            mask = self.geometric_mask(data.shape[0])
        
        masked_data = data.copy()
        masked_data[mask] = 0  # Zero masking
        
        return masked_data, mask
    
    def __call__(self, data):
        """
        Apply geometric masking to data.
        
        Args:
            data (np.ndarray): Input time series
            
        Returns:
            tuple: (masked_data, mask)
        """
        return self.apply_mask(data)


class SMDDataset(Dataset):
    """
    Server Machine Dataset (SMD) Dataset class.
    
    Handles loading, preprocessing, and windowing of SMD data with
    optional geometric masking for data augmentation.
    """
    
    def __init__(self, data, labels, window_size=100, stride=1, 
                 apply_masking=False, mask_prob=0.15, mask_ratio=0.3):
        """
        Initialize SMD Dataset.
        
        Args:
            data (np.ndarray): Time series data of shape (n_samples, n_features)
            labels (np.ndarray): Binary labels (0: normal, 1: anomaly)
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            apply_masking (bool): Whether to apply geometric masking
            mask_prob (float): Probability for geometric masking
            mask_ratio (float): Ratio for geometric masking
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.apply_masking = apply_masking
        
        # Initialize masking
        if apply_masking:
            self.masking = GeometricMasking(mask_prob, mask_ratio)
        else:
            self.masking = None
        
        # Create windows
        self.windows, self.window_labels = self._create_windows()
        
    def _create_windows(self):
        """
        Create sliding windows from time series data.
        
        Returns:
            tuple: (windows, labels) where windows is array of shape
                   (n_windows, window_size, n_features)
        """
        n_samples, n_features = self.data.shape
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        windows = []
        labels = []
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            window = self.data[start_idx:end_idx]
            # Label window as anomalous if any point in window is anomalous
            label = np.max(self.labels[start_idx:end_idx])
            
            windows.append(window)
            labels.append(label)
        
        windows = np.array(windows, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        return windows, labels
    
    def __len__(self):
        """Return number of windows."""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a single windowed sample.
        
        Args:
            idx (int): Index of sample
            
        Returns:
            tuple: (window, label) where window is tensor of shape
                   (window_size, n_features)
        """
        window = self.windows[idx].copy()
        label = self.window_labels[idx]
        
        # Apply masking during training
        if self.apply_masking and self.masking is not None:
            window, _ = self.masking(window)
        
        # Convert to tensors
        window = torch.from_numpy(window).float()
        label = torch.tensor(label).float()
        
        return window, label


def load_smd_data(data_path, machine_id):
    """
    Load SMD (Server Machine Dataset) data.
    
    Expected directory structure:
        data_path/
            train/
                machine-1-1.txt
                ...
            test/
                machine-1-1.txt
                ...
            test_label/
                machine-1-1.txt
                ...
    
    Args:
        data_path (str): Path to SMD dataset directory
        machine_id (str): Machine identifier (e.g., 'machine-1-1')
        
    Returns:
        tuple: (train_data, test_data, test_labels)
               Each numpy array of shape (n_samples, n_features)
    """
    # Load training data (normal data only)
    train_file = os.path.join(data_path, 'train', f'{machine_id}.txt')
    train_data = np.loadtxt(train_file, delimiter=',', dtype=np.float32)
    
    # Load test data
    test_file = os.path.join(data_path, 'test', f'{machine_id}.txt')
    test_data = np.loadtxt(test_file, delimiter=',', dtype=np.float32)
    
    # Load test labels
    label_file = os.path.join(data_path, 'test_label', f'{machine_id}.txt')
    test_labels = np.loadtxt(label_file, delimiter=',', dtype=np.float32)
    
    # If test_labels is 1D, reshape to (n_samples,)
    if len(test_labels.shape) == 1:
        test_labels = test_labels
    else:
        # If multi-column, take max across columns (any anomaly = anomaly)
        test_labels = np.max(test_labels, axis=1)
    
    return train_data, test_data, test_labels


def normalize_data(train_data, test_data):
    """
    Normalize data using statistics from training set.
    
    Uses z-score normalization (standardization) to center data
    around zero with unit variance.
    
    Args:
        train_data (np.ndarray): Training data
        test_data (np.ndarray): Test data
        
    Returns:
        tuple: (normalized_train, normalized_test, mean, std)
    """
    # Compute statistics from training data
    mean = np.mean(train_data, axis=0, keepdims=True)
    std = np.std(train_data, axis=0, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    # Normalize
    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std
    
    return train_normalized, test_normalized, mean, std


def split_train_val(data, labels, val_ratio=0.2, shuffle=False, seed=42):
    """
    Split data into training and validation sets.
    
    Args:
        data (np.ndarray): Data array
        labels (np.ndarray): Label array
        val_ratio (float): Ratio of validation data
        shuffle (bool): Whether to shuffle before split
        seed (int): Random seed
        
    Returns:
        tuple: (train_data, train_labels, val_data, val_labels)
    """
    n_samples = len(data)
    n_val = int(n_samples * val_ratio)
    
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        data = data[indices]
        labels = labels[indices]
    
    # Split
    val_data = data[:n_val]
    val_labels = labels[:n_val]
    train_data = data[n_val:]
    train_labels = labels[n_val:]
    
    return train_data, train_labels, val_data, val_labels


def create_data_loaders(data_path, machine_id, window_size=100, stride=1,
                       batch_size=128, num_workers=4, mask_prob=0.15,
                       mask_ratio=0.3, val_ratio=0.2):
    """
    Create data loaders for training, validation, and testing.
    
    This function handles the complete data pipeline:
    1. Load raw SMD data
    2. Normalize using training statistics
    3. Split into train/val/test
    4. Create datasets with windowing and masking
    5. Create data loaders
    
    Args:
        data_path (str): Path to SMD dataset
        machine_id (str): Machine identifier
        window_size (int): Sliding window size
        stride (int): Stride for sliding windows
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        mask_prob (float): Probability for geometric masking
        mask_ratio (float): Ratio for geometric masking
        val_ratio (float): Validation set ratio
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, n_features)
    """
    # Load data
    print(f"Loading data for {machine_id}...")
    train_data, test_data, test_labels = load_smd_data(data_path, machine_id)
    
    # Get number of features
    n_features = train_data.shape[1]
    print(f"Number of features: {n_features}")
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Normalize data
    print("Normalizing data...")
    train_data, test_data, mean, std = normalize_data(train_data, test_data)
    
    # Create training labels (all normal for unsupervised learning)
    train_labels = np.zeros(len(train_data), dtype=np.float32)
    
    # Split training data into train and validation
    print("Splitting train/validation...")
    train_data, train_labels, val_data, val_labels = split_train_val(
        train_data, train_labels, val_ratio=val_ratio
    )
    
    print(f"Training samples after split: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SMDDataset(
        train_data, train_labels, window_size, stride,
        apply_masking=True, mask_prob=mask_prob, mask_ratio=mask_ratio
    )
    
    val_dataset = SMDDataset(
        val_data, val_labels, window_size, stride,
        apply_masking=False
    )
    
    test_dataset = SMDDataset(
        test_data, test_labels, window_size, stride,
        apply_masking=False
    )
    
    print(f"Training windows: {len(train_dataset)}")
    print(f"Validation windows: {len(val_dataset)}")
    print(f"Test windows: {len(test_dataset)}")
    
    # Calculate anomaly ratio in test set
    anomaly_ratio = np.mean(test_dataset.window_labels)
    print(f"Test set anomaly ratio: {anomaly_ratio:.2%}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, n_features


def save_preprocessing_params(mean, std, save_path):
    """
    Save preprocessing parameters for inference.
    
    Args:
        mean (np.ndarray): Mean values
        std (np.ndarray): Standard deviation values
        save_path (str): Path to save parameters
    """
    params = {
        'mean': mean,
        'std': std
    }
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_preprocessing_params(load_path):
    """
    Load preprocessing parameters.
    
    Args:
        load_path (str): Path to load parameters from
        
    Returns:
        dict: Dictionary containing mean and std
    """
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return params