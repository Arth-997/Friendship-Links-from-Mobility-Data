"""
Utilities for SRINet Implementation

This module contains utility classes and functions for configuration management,
random seed setting, early stopping, and other helper functionality.
"""

import torch
import numpy as np
import random
import json
import os
from typing import Dict, Any, Optional


class SRINetConfig:
    """Configuration class for SRINet hyperparameters and settings"""
    
    def __init__(self):
        # Model architecture parameters
        self.embedding_dim = 512
        self.hidden_dim = 256  
        self.num_layers = 2
        self.dropout = 0.01
        self.num_categories = 10  # Will be updated based on data
        self.layer_type = 'gcn'  # 'gcn' or 'gat'
        self.fusion_type = 'mean'  # 'mean' or 'attention'
        
        # Binary concrete parameters
        self.temperature_init = 1.0
        self.temperature_final = 0.1
        self.gamma = -0.1  # Lower stretch parameter
        self.eta = 1.1     # Upper stretch parameter
        
        # Training parameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.omega = 0.003  # Sparsity loss weight
        self.batch_size = 1024
        self.num_epochs = 100
        self.patience = 10
        
        # Data processing parameters
        self.time_window_hours = 2  # τ parameter for meeting detection
        self.min_checkins_per_user = 5
        self.min_meetings_per_category = 100
        
        # Evaluation parameters
        self.test_ratio = 0.2
        self.val_ratio = 0.1
        
        # Advanced options
        self.gradient_clip_norm = 1.0
        self.scheduler_factor = 0.5
        self.scheduler_patience = 5
        self.min_lr = 1e-6
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter '{key}'")
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod 
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        config = cls()
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            config.update(data)
        else:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        return config
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.omega >= 0, "omega must be non-negative"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.time_window_hours > 0, "time_window_hours must be positive"
        assert 0 < self.test_ratio < 1, "test_ratio must be in (0, 1)"
        assert 0 < self.val_ratio < 1, "val_ratio must be in (0, 1)"
        assert self.test_ratio + self.val_ratio < 1, "test_ratio + val_ratio must be < 1"
        print("✓ Configuration validation passed")
    
    def __repr__(self):
        return f"SRINetConfig(embedding_dim={self.embedding_dim}, layers={self.num_layers}, omega={self.omega})"


class EarlyStopping:
    """Early stopping utility to halt training when validation performance stops improving"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for metrics like loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should be stopped
        
        Args:
            val_score: Current validation score
            
        Returns:
            bool: True if training should be stopped
        """
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if the score is an improvement"""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class MetricsTracker:
    """Track and manage training metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value for a metric"""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_best(self, key: str, mode: str = 'max') -> Optional[float]:
        """Get the best value for a metric"""
        if key not in self.metrics or not self.metrics[key]:
            return None
        
        if mode == 'max':
            return max(self.metrics[key])
        else:
            return min(self.metrics[key])
    
    def to_dict(self) -> Dict[str, list]:
        """Convert metrics to dictionary"""
        return self.metrics.copy()


class ModelSaver:
    """Utility for saving and loading model checkpoints"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       metrics: Dict[str, float], filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'save_time': torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'
        }
        
        torch.save(checkpoint, filepath)
        return filepath
    
    def load_checkpoint(self, filepath: str, model, optimizer=None, scheduler=None):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seeds set to {seed}")


def count_parameters(model) -> Dict[str, int]:
    """Count the number of parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_device_info() -> Dict[str, Any]:
    """Get information about available computing devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
    
    return info


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_memory(bytes_val: int) -> str:
    """Format memory in bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}PB"


def log_system_info():
    """Log system information"""
    device_info = get_device_info()
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"Current Device: {device_info['current_device']}")
    
    if device_info['cuda_available']:
        print(f"GPU: {device_info['cuda_device_name']}")
        print(f"Total Memory: {format_memory(device_info['cuda_memory_total'])}")
        print(f"Allocated Memory: {format_memory(device_info['cuda_memory_allocated'])}")
    
    print("="*50 + "\n")


def create_run_id() -> str:
    """Create a unique run ID for experiment tracking"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"srinet_run_{timestamp}"


class ProgressLogger:
    """Utility for logging training progress"""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.start_time = None
        
    def start(self):
        """Start timing"""
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch progress"""
        if epoch % self.log_interval == 0:
            log_str = f"Epoch {epoch:3d}: "
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_str += f"{key}={value:.4f} "
                else:
                    log_str += f"{key}={value} "
            print(log_str)
    
    def log_final(self, total_epochs: int, best_metrics: Dict[str, float]):
        """Log final results"""
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total Epochs: {total_epochs}")
        print("Best Metrics:")
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"{'='*60}\n")


# Validation utilities
def validate_data_format(checkin_df, required_columns=None):
    """Validate check-in data format"""
    if required_columns is None:
        required_columns = ['user_id', 'poi_id', 'timestamp', 'category']
    
    missing_cols = set(required_columns) - set(checkin_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values
    null_counts = checkin_df[required_columns].isnull().sum()
    if null_counts.any():
        print("Warning: Found null values:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} null values")
    
    print("✓ Data format validation passed")


def validate_graph_data(adjacency_matrices):
    """Validate graph data structure"""
    required_keys = ['edge_index', 'edge_weights', 'num_edges', 'num_nodes']
    
    for category, data in adjacency_matrices.items():
        missing_keys = set(required_keys) - set(data.keys())
        if missing_keys:
            raise ValueError(f"Category '{category}' missing keys: {missing_keys}")
        
        # Validate tensor shapes
        edge_index = data['edge_index']
        edge_weights = data['edge_weights']
        
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index should have shape [2, E], got {edge_index.shape}")
        
        if edge_index.shape[1] != edge_weights.shape[0]:
            raise ValueError(f"edge_index and edge_weights shape mismatch")
    
    print("✓ Graph data validation passed")


# Export main classes and functions
__all__ = [
    'SRINetConfig',
    'EarlyStopping', 
    'MetricsTracker',
    'ModelSaver',
    'ProgressLogger',
    'set_random_seeds',
    'count_parameters',
    'get_device_info',
    'log_system_info',
    'create_run_id',
    'validate_data_format',
    'validate_graph_data',
    'format_time',
    'format_memory'
]