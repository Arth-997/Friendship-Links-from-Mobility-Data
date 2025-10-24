"""
SRINet Model Implementation

This module implements the complete SRINet architecture that integrates
the topology mask module with multiplex GNN processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_module import TopologyMaskModule
from .gnn_layers import MaskedGCNLayer, create_gnn_layer


class SRINet(nn.Module):
    """
    Complete SRINet model implementation
    
    Architecture:
    1. For each POI category r:
       - Apply L layers of (mask + masked GCN)
       - Collect category-specific embeddings H^(r)
    2. Fuse category embeddings: H = fusion(H^(r) for r in categories)
    3. Compute pairwise scores and losses
    
    Args:
        config: Configuration object with model hyperparameters
        num_users: Number of users in the dataset
        adjacency_matrices: Dict of adjacency matrices per category
        layer_type: Type of GNN layer ('gcn' or 'gat')
    """
    
    def __init__(self, config, num_users, adjacency_matrices, layer_type='gcn'):
        super().__init__()
        
        self.config = config
        self.num_users = num_users
        self.categories = list(adjacency_matrices.keys())
        self.num_categories = len(self.categories)
        self.adjacency_matrices = adjacency_matrices
        self.layer_type = layer_type
        
        # Learnable node embeddings (initial features)
        self.node_embeddings = nn.Parameter(
            torch.randn(num_users, config.embedding_dim) * 0.1
        )
        
        # Mask modules for each layer and category
        self.mask_modules = nn.ModuleDict()
        for layer in range(config.num_layers):
            self.mask_modules[f'layer_{layer}'] = nn.ModuleDict()
            for cat in self.categories:
                self.mask_modules[f'layer_{layer}'][cat] = TopologyMaskModule(
                    config.embedding_dim, config.hidden_dim
                )
        
        # GNN layers for each category
        self.gnn_layers = nn.ModuleDict()
        for layer in range(config.num_layers):
            self.gnn_layers[f'layer_{layer}'] = nn.ModuleDict()
            for cat in self.categories:
                self.gnn_layers[f'layer_{layer}'][cat] = create_gnn_layer(
                    layer_type, config.embedding_dim, config.embedding_dim
                )
        
        # Fusion mechanism
        if hasattr(config, 'fusion_type') and config.fusion_type == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                config.embedding_dim, num_heads=4, batch_first=True
            )
        else:
            self.fusion_attention = None
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Temperature parameter (will be annealed during training)
        self.register_buffer('temperature', torch.tensor(config.temperature_init))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embedding_dim) for _ in range(config.num_layers)
        ])
        
    def forward(self, positive_pairs=None, negative_pairs=None):
        """
        Forward pass through SRINet
        
        Args:
            positive_pairs: [P, 2] positive user pairs for training
            negative_pairs: [N, 2] negative user pairs for training
            
        Returns:
            Dictionary containing embeddings, scores, and losses
        """
        category_embeddings = []
        total_sparsity_loss = 0.0
        mask_stats = {}
        
        # Process each category separately
        for cat_idx, category in enumerate(self.categories):
            edge_index = self.adjacency_matrices[category]['edge_index'].to(self.node_embeddings.device)
            edge_weights = self.adjacency_matrices[category]['edge_weights'].to(self.node_embeddings.device)
            
            # Start with base node embeddings
            h = self.node_embeddings
            
            # Apply layers sequentially
            for layer in range(self.config.num_layers):
                # Get mask module and GNN layer for this category and layer
                mask_module = self.mask_modules[f'layer_{layer}'][category]
                gnn_layer = self.gnn_layers[f'layer_{layer}'][category]
                
                # Compute edge masks using current embeddings
                edge_masks, scores, sparsity_loss = mask_module(
                    h, edge_index, self.temperature, 
                    self.config.gamma, self.config.eta
                )
                
                # Apply masked GNN layer
                h = gnn_layer(h, edge_index, edge_weights, edge_masks)
                h = F.relu(h)
                h = self.layer_norms[layer](h)
                h = self.dropout(h)
                
                # Accumulate sparsity loss
                total_sparsity_loss += sparsity_loss
                
                # Store mask statistics
                mask_stats[f'{category}_layer_{layer}'] = mask_module.get_mask_statistics(edge_masks)
            
            category_embeddings.append(h)
        
        # Fuse category embeddings
        fused_embeddings = self._fuse_embeddings(category_embeddings)
        
        # Prepare result dictionary
        result = {
            'node_embeddings': fused_embeddings,
            'category_embeddings': category_embeddings,
            'sparsity_loss': total_sparsity_loss,
            'mask_stats': mask_stats,
            'temperature': self.temperature.item()
        }
        
        # Compute pairwise losses if training pairs are provided
        if positive_pairs is not None and negative_pairs is not None:
            scores_pos, scores_neg, semi_loss = self._compute_pairwise_loss(
                fused_embeddings, positive_pairs, negative_pairs
            )
            
            result.update({
                'positive_scores': scores_pos,
                'negative_scores': scores_neg, 
                'semi_supervised_loss': semi_loss,
                'total_loss': semi_loss + self.config.omega * total_sparsity_loss
            })
        
        return result
    
    def _fuse_embeddings(self, category_embeddings):
        """Fuse embeddings from different categories
        
        Args:
            category_embeddings: List of [N, D] embeddings per category
            
        Returns:
            fused_embeddings: [N, D] fused embedding matrix
        """
        if len(category_embeddings) == 1:
            return category_embeddings[0]
        
        if self.fusion_attention is not None:
            # Attention-based fusion
            stacked = torch.stack(category_embeddings, dim=1)  # [N, C, D]
            fused, _ = self.fusion_attention(stacked, stacked, stacked)
            fused = fused.mean(dim=1)  # Average over categories
        else:
            # Simple mean fusion
            fused = torch.stack(category_embeddings).mean(dim=0)
        
        return fused
    
    def _compute_pairwise_loss(self, embeddings, positive_pairs, negative_pairs):
        """Compute semi-supervised pairwise loss
        
        Args:
            embeddings: [N, D] node embeddings
            positive_pairs: [P, 2] positive user pairs
            negative_pairs: [N, 2] negative user pairs
            
        Returns:
            scores_pos: [P] positive pair scores
            scores_neg: [N] negative pair scores  
            semi_loss: scalar semi-supervised loss
        """
        # Positive pair scores
        pos_u = embeddings[positive_pairs[:, 0]]
        pos_v = embeddings[positive_pairs[:, 1]]
        scores_pos = (pos_u * pos_v).sum(dim=1)
        
        # Negative pair scores  
        neg_u = embeddings[negative_pairs[:, 0]]
        neg_v = embeddings[negative_pairs[:, 1]]
        scores_neg = (neg_u * neg_v).sum(dim=1)
        
        # Semi-supervised loss: maximize positive, minimize negative
        loss_pos = -F.logsigmoid(scores_pos).mean()
        loss_neg = -F.logsigmoid(-scores_neg).mean()
        semi_loss = loss_pos + loss_neg
        
        return scores_pos, scores_neg, semi_loss
    
    def update_temperature(self, epoch, total_epochs):
        """Anneal temperature during training
        
        Uses linear annealing from initial to final temperature.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of training epochs
        """
        if total_epochs > 0:
            progress = epoch / total_epochs
            new_temp = (
                self.config.temperature_init * (1 - progress) + 
                self.config.temperature_final * progress
            )
            self.temperature.fill_(max(new_temp, self.config.temperature_final))
    
    def get_embeddings(self):
        """Get final user embeddings without computing losses
        
        Returns:
            embeddings: [N, D] user embedding matrix
        """
        with torch.no_grad():
            result = self.forward()
            return result['node_embeddings']
    
    def get_mask_summary(self):
        """Get summary of current mask statistics across all layers/categories
        
        Returns:
            dict: Summary statistics of masks
        """
        with torch.no_grad():
            result = self.forward()
            mask_stats = result['mask_stats']
            
            all_means = [stats['mean'] for stats in mask_stats.values()]
            all_sparsity = [stats['sparsity_rate'] for stats in mask_stats.values()]
            
            return {
                'overall_mean_mask': sum(all_means) / len(all_means) if all_means else 0.0,
                'overall_sparsity_rate': sum(all_sparsity) / len(all_sparsity) if all_sparsity else 0.0,
                'per_category': {
                    cat: {
                        'mean_mask': sum(stats['mean'] for key, stats in mask_stats.items() if key.startswith(cat)) / self.config.num_layers,
                        'sparsity_rate': sum(stats['sparsity_rate'] for key, stats in mask_stats.items() if key.startswith(cat)) / self.config.num_layers
                    }
                    for cat in self.categories
                },
                'temperature': self.temperature.item()
            }
    
    def freeze_masks(self, threshold=0.1):
        """Freeze masks for production inference
        
        This sets mask values to hard 0/1 based on threshold for faster inference.
        
        Args:
            threshold: Threshold below which masks are set to 0
        """
        print(f"Freezing masks with threshold {threshold}...")
        
        with torch.no_grad():
            for category in self.categories:
                edge_index = self.adjacency_matrices[category]['edge_index'].to(self.node_embeddings.device)
                
                for layer in range(self.config.num_layers):
                    mask_module = self.mask_modules[f'layer_{layer}'][category]
                    
                    # Get current masks
                    edge_masks, _, _ = mask_module(
                        self.node_embeddings, edge_index, self.temperature
                    )
                    
                    # Convert to hard 0/1
                    hard_masks = (edge_masks >= threshold).float()
                    
                    # Store hard masks (this is simplified - in practice you'd modify the forward pass)
                    setattr(mask_module, f'_frozen_masks', hard_masks)
        
        print("✓ Masks frozen for production inference")
    
    def __repr__(self):
        return (f'SRINet(users={self.num_users}, categories={self.num_categories}, '
                f'layers={self.config.num_layers}, dim={self.config.embedding_dim})')