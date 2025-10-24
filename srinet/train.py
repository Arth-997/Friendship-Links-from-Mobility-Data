"""
Training Script for SRINet

This script handles the complete training pipeline for SRINet including
data loading, model initialization, training loop, and evaluation.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import json
import pickle
from datetime import datetime
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.srinet import SRINet
from src.data_processing import DataProcessor
from src.graph_builder import GraphBuilder
from src.utils import SRINetConfig, set_random_seeds, EarlyStopping


class FriendshipDataset:
    """Dataset for generating positive/negative user pairs for training"""
    
    def __init__(self, adjacency_matrices, num_users, test_ratio=0.2, val_ratio=0.1):
        self.num_users = num_users
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        
        # Collect all edges (friendships) from all categories
        all_edges = set()
        for category, data in adjacency_matrices.items():
            edge_index = data['edge_index'].numpy()
            # Only keep one direction for undirected edges
            for i in range(edge_index.shape[1]):
                u, v = edge_index[:, i]
                if u < v:  # Canonical order to avoid duplicates
                    all_edges.add((u, v))
        
        self.positive_pairs = list(all_edges)
        print(f"Found {len(self.positive_pairs)} positive pairs from graphs")
        
        # Split into train/val/test
        train_pos, test_pos = train_test_split(
            self.positive_pairs, test_size=test_ratio, random_state=42
        )
        
        self.train_pos, self.val_pos = train_test_split(
            train_pos, test_size=val_ratio/(1-test_ratio), random_state=42
        )
        
        self.test_pos = test_pos
        
        print(f"Split: {len(self.train_pos)} train, {len(self.val_pos)} val, {len(self.test_pos)} test positive pairs")
    
    def sample_negative_pairs(self, num_negative, exclude_positive=None, subset='train'):
        """Sample negative pairs (non-connected users)"""
        if exclude_positive is None:
            exclude_positive = set(self.positive_pairs)
        
        # Add current subset positives to exclusion
        if subset == 'train':
            exclude_positive.update(self.train_pos)
        elif subset == 'val':
            exclude_positive.update(self.train_pos + self.val_pos)
        elif subset == 'test':
            exclude_positive.update(self.positive_pairs)
        
        negative_pairs = []
        max_attempts = num_negative * 10
        
        for _ in range(max_attempts):
            if len(negative_pairs) >= num_negative:
                break
                
            u = np.random.randint(0, self.num_users)
            v = np.random.randint(0, self.num_users)
            
            if u != v:
                pair = (min(u, v), max(u, v))
                if pair not in exclude_positive:
                    negative_pairs.append(pair)
        
        return negative_pairs[:num_negative]


class SRINetTrainer:
    """Training pipeline for SRINet"""
    
    def __init__(self, model, dataset, config, device, save_dir):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.patience, min_delta=0.001)
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_semi_loss': [],
            'train_sparsity_loss': [],
            'val_loss': [],
            'val_roc_auc': [],
            'val_pr_auc': [],
            'test_roc_auc': [],
            'test_pr_auc': [],
            'temperature': [],
            'learning_rate': [],
            'mean_mask_values': []
        }
        
        # Best model tracking
        self.best_val_score = 0.0
        self.best_model_state = None
        self.best_epoch = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_semi_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        # Create batches of positive/negative pairs
        batch_size = self.config.batch_size
        train_pos = self.dataset.train_pos
        
        # Shuffle training data
        indices = np.random.permutation(len(train_pos))
        
        for i in range(0, len(train_pos), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_pos = [train_pos[idx] for idx in batch_indices]
            
            # Sample equal number of negative pairs
            batch_neg = self.dataset.sample_negative_pairs(
                len(batch_pos), subset='train'
            )
            
            if len(batch_neg) < len(batch_pos):
                continue
                
            # Convert to tensors
            pos_pairs = torch.tensor(batch_pos, dtype=torch.long).to(self.device)
            neg_pairs = torch.tensor(batch_neg, dtype=torch.long).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            result = self.model(pos_pairs, neg_pairs)
            
            loss = result['total_loss']
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_semi_loss += result['semi_supervised_loss'].item()
            total_sparsity_loss += result['sparsity_loss'].item()
            num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'semi_loss': 0.0, 'sparsity_loss': 0.0}
        
        return {
            'loss': total_loss / num_batches,
            'semi_loss': total_semi_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches
        }
    
    def evaluate(self, subset='val'):
        """Evaluate model on validation or test set"""
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = self.model.get_embeddings().cpu().numpy()
            
            # Get pairs for evaluation
            if subset == 'val':
                pos_pairs = self.dataset.val_pos
            elif subset == 'test':
                pos_pairs = self.dataset.test_pos
            else:
                raise ValueError("subset must be 'val' or 'test'")
            
            # Sample negative pairs
            neg_pairs = self.dataset.sample_negative_pairs(
                len(pos_pairs), subset=subset
            )
            
            # Compute scores
            pos_scores = []
            for u, v in pos_pairs:
                score = np.dot(embeddings[u], embeddings[v])
                pos_scores.append(score)
            
            neg_scores = []
            for u, v in neg_pairs:
                score = np.dot(embeddings[u], embeddings[v])
                neg_scores.append(score)
            
            # Combine labels and scores
            y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
            y_scores = pos_scores + neg_scores
            
            # Compute metrics
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)
            except:
                roc_auc = 0.0
                pr_auc = 0.0
            
            # Compute validation loss
            val_loss = 0.0
            if subset == 'val' and len(pos_pairs) > 0:
                pos_tensor = torch.tensor(pos_pairs[:self.config.batch_size], dtype=torch.long).to(self.device)
                neg_tensor = torch.tensor(neg_pairs[:self.config.batch_size], dtype=torch.long).to(self.device)
                result = self.model(pos_tensor, neg_tensor)
                val_loss = result['total_loss'].item()
            
            return roc_auc, pr_auc, val_loss
    
    def get_mask_statistics(self):
        """Get current mask statistics"""
        self.model.eval()
        
        with torch.no_grad():
            mask_summary = self.model.get_mask_summary()
            return mask_summary['overall_mean_mask']
    
    def train(self):
        """Full training loop"""
        print("Starting SRINet training...")
        print(f"Training on {len(self.dataset.train_pos)} positive pairs")
        print(f"Validation on {len(self.dataset.val_pos)} positive pairs") 
        print(f"Test on {len(self.dataset.test_pos)} positive pairs")
        print(f"Model: {self.model}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            # Update temperature
            self.model.update_temperature(epoch, self.config.num_epochs)
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate on validation and test sets
            val_roc_auc, val_pr_auc, val_loss = self.evaluate('val')
            test_roc_auc, test_pr_auc, _ = self.evaluate('test')
            
            # Get mask statistics
            mean_mask = self.get_mask_statistics()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_semi_loss'].append(train_metrics['semi_loss'])
            self.history['train_sparsity_loss'].append(train_metrics['sparsity_loss'])
            self.history['val_loss'].append(val_loss)
            self.history['val_roc_auc'].append(val_roc_auc)
            self.history['val_pr_auc'].append(val_pr_auc)
            self.history['test_roc_auc'].append(test_roc_auc)
            self.history['test_pr_auc'].append(test_pr_auc)
            self.history['temperature'].append(self.model.temperature.item())
            self.history['learning_rate'].append(current_lr)
            self.history['mean_mask_values'].append(mean_mask)
            
            # Scheduler step
            self.scheduler.step(val_pr_auc)
            
            # Early stopping check
            if val_pr_auc > self.best_val_score:
                self.best_val_score = val_pr_auc
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pt')
            
            # Early stopping
            if self.early_stopping(val_pr_auc):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 5 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: "
                      f"Loss={train_metrics['loss']:.4f} "
                      f"Semi={train_metrics['semi_loss']:.4f} "
                      f"Sparse={train_metrics['sparsity_loss']:.4f} "
                      f"Val_ROC={val_roc_auc:.4f} "
                      f"Val_PR={val_pr_auc:.4f} "
                      f"Test_ROC={test_roc_auc:.4f} "
                      f"Test_PR={test_pr_auc:.4f} "
                      f"T={self.model.temperature.item():.3f} "
                      f"Mask={mean_mask:.3f} "
                      f"LR={current_lr:.2e}")
            
            # Save periodic checkpoint
            if epoch % 20 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Loaded best model from epoch {self.best_epoch} with Val PR-AUC: {self.best_val_score:.4f}")
        
        # Final evaluation
        final_val_roc, final_val_pr, _ = self.evaluate('val')
        final_test_roc, final_test_pr, _ = self.evaluate('test')
        
        print(f"\nFinal Results:")
        print(f"  Validation: ROC-AUC={final_val_roc:.4f}, PR-AUC={final_val_pr:.4f}")
        print(f"  Test:       ROC-AUC={final_test_roc:.4f}, PR-AUC={final_test_pr:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': len(self.history['epoch']) - 1 if self.history['epoch'] else 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch
        }
        
        torch.save(checkpoint, filepath)
        
    def save_final_results(self):
        """Save final training results"""
        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        
        # Save final metrics
        final_metrics = {
            'best_val_pr_auc': self.best_val_score,
            'best_epoch': self.best_epoch,
            'final_val_roc_auc': self.history['val_roc_auc'][-1],
            'final_val_pr_auc': self.history['val_pr_auc'][-1],
            'final_test_roc_auc': self.history['test_roc_auc'][-1],
            'final_test_pr_auc': self.history['test_pr_auc'][-1],
            'final_temperature': self.history['temperature'][-1],
            'final_mean_mask': self.history['mean_mask_values'][-1],
            'total_epochs': len(self.history['epoch']),
            'training_completed_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.save_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"✓ Saved training results to {self.save_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train SRINet model')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='srinet/data',
                       help='Directory containing processed data')
    parser.add_argument('--save_dir', type=str, default='srinet/experiments',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load or create configuration
    if os.path.exists(args.config):
        config = SRINetConfig.load(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = SRINetConfig()
        config.save(args.config)
        print(f"Created default configuration at {args.config}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if processed data exists
    if not os.path.exists(f"{args.data_dir}/adjacency_matrices.pt"):
        print("Processed data not found. Please run data preprocessing first.")
        print("Example: python src/data_processing.py")
        return
    
    # Load processed data
    print("Loading processed data...")
    adjacency_matrices = torch.load(f"{args.data_dir}/adjacency_matrices.pt")
    
    with open(f"{args.data_dir}/user_mapping.pkl", 'rb') as f:
        user_to_idx = pickle.load(f)
    
    num_users = len(user_to_idx)
    config.num_categories = len(adjacency_matrices)
    
    print(f"Loaded data: {num_users} users, {config.num_categories} categories")
    
    # Create dataset
    dataset = FriendshipDataset(adjacency_matrices, num_users, config.test_ratio, config.val_ratio)
    
    # Initialize model
    model = SRINet(config, num_users, adjacency_matrices).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    trainer = SRINetTrainer(model, dataset, config, device, args.save_dir)
    
    # Train model
    history = trainer.train()
    
    # Save final results
    trainer.save_final_results()
    
    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main()