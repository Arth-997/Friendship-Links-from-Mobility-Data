"""
SRINet Topology Mask Module

This module implements the binary concrete topology filtering mechanism
from the SRINet paper. It includes:
- MLP scorer network for edge scoring
- Binary concrete sampling with temperature annealing
- Differentiable sparsity loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyMaskModule(nn.Module):
    """Binary concrete topology filtering module
    
    This module implements the core mask learning mechanism that decides
    which edges to keep/prune in the multiplex user meeting graphs.
    
    Args:
        input_dim (int): Dimension of node embeddings
        hidden_dim (int): Hidden dimension for MLP scorer
    """
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # MLP scorer network f_θ(h_i, h_j) -> a_ij
        self.scorer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Concatenated node features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Output scalar score a_ij
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, node_embeddings, edge_index, temperature, gamma=-0.1, eta=1.1):
        """
        Forward pass of mask module
        
        Args:
            node_embeddings: [N, D] node feature matrix
            edge_index: [2, E] edge connectivity in COO format  
            temperature: current temperature T for binary concrete
            gamma: lower stretch parameter (should be ≤ 0)
            eta: upper stretch parameter (should be ≥ 1)
            
        Returns:
            edge_masks: [E] binary concrete masks in [0,1]
            scores: [E] raw scores a_ij from MLP scorer
            sparsity_loss: scalar L_s sparsity regularization term
        """
        # Get source and target node features
        src_nodes = edge_index[0]  # [E]
        tgt_nodes = edge_index[1]  # [E]
        
        src_features = node_embeddings[src_nodes]  # [E, D]
        tgt_features = node_embeddings[tgt_nodes]  # [E, D] 
        
        # Concatenate node features [h_i || h_j]
        edge_features = torch.cat([src_features, tgt_features], dim=1)  # [E, 2*D]
        
        # Compute edge scores a_ij
        scores = self.scorer(edge_features).squeeze(-1)  # [E]
        
        # Binary concrete sampling
        edge_masks = self._binary_concrete_sample(scores, temperature, gamma, eta)
        
        # Compute analytic sparsity loss
        sparsity_loss = self._compute_sparsity_loss(scores, temperature, gamma, eta)
        
        return edge_masks, scores, sparsity_loss
    
    def _binary_concrete_sample(self, scores, temperature, gamma, eta):
        """
        Binary concrete relaxation sampling (Equation 3 from paper)
        
        Implementation:
        1. Sample ε ~ Uniform(0,1)
        2. Compute s = sigmoid((log ε - log(1-ε) + a) / T)
        3. Stretch: s̄ = s*(η-γ) + γ  
        4. Clip: M = clip(s̄, 0, 1)
        
        Args:
            scores: [E] edge scores a_ij
            temperature: temperature T
            gamma: lower bound for stretching
            eta: upper bound for stretching
            
        Returns:
            masks: [E] binary concrete masks in [0,1]
        """
        # Sample uniform noise with numerical stability
        eps = torch.rand_like(scores)
        eps = torch.clamp(eps, 1e-7, 1 - 1e-7)  # Avoid log(0)
        
        # Gumbel noise: log ε - log(1-ε)
        gumbel_noise = torch.log(eps) - torch.log(1 - eps)
        
        # Binary concrete with temperature
        logits = (gumbel_noise + scores) / temperature
        s = torch.sigmoid(logits)
        
        # Stretch to [γ, η] then clip to [0, 1]
        s_stretched = s * (eta - gamma) + gamma
        masks = torch.clamp(s_stretched, 0.0, 1.0)
        
        return masks
    
    def _compute_sparsity_loss(self, scores, temperature, gamma, eta):
        """
        Compute analytic expectation of L_s sparsity loss
        
        For binary concrete distribution, the expectation is:
        E[M_ij] = clip(sigmoid(a_ij/T) * (η-γ) + γ, 0, 1)
        
        Args:
            scores: [E] edge scores a_ij
            temperature: temperature T
            gamma: lower stretch parameter
            eta: upper stretch parameter
            
        Returns:
            sparsity_loss: scalar expectation of sum of masks
        """
        # Analytic expectation of stretched sigmoid
        sigmoid_scores = torch.sigmoid(scores / temperature)
        expected_masks = sigmoid_scores * (eta - gamma) + gamma
        
        # Clamp to [0,1] and sum for L_s
        expected_masks = torch.clamp(expected_masks, 0.0, 1.0)
        sparsity_loss = expected_masks.sum()
        
        return sparsity_loss
    
    def get_mask_statistics(self, edge_masks):
        """Get statistics about current mask values
        
        Args:
            edge_masks: [E] mask values
            
        Returns:
            dict: Statistics including mean, std, sparsity rate
        """
        with torch.no_grad():
            return {
                'mean': edge_masks.mean().item(),
                'std': edge_masks.std().item(),
                'min': edge_masks.min().item(),
                'max': edge_masks.max().item(),
                'sparsity_rate': (edge_masks < 0.1).float().mean().item(),
                'num_edges': len(edge_masks)
            }