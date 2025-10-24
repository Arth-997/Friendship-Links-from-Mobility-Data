"""
SRINet GNN Layers

This module implements the graph neural network layers used in SRINet,
including masked GCN layers that can apply edge masks during message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class MaskedGCNLayer(MessagePassing):
    """GCN layer with edge masking support
    
    This layer extends standard GCN to support edge masks that can
    dynamically filter edges during message passing.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        bias (bool): Whether to use bias
        normalize (bool): Whether to apply degree normalization
    """
    
    def __init__(self, in_channels, out_channels, bias=True, normalize=True):
        super().__init__(aggr='add')  # Use 'add' aggregation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        # Linear transformation weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None, edge_mask=None):
        """
        Forward pass with optional edge masking
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity in COO format
            edge_weight: [E] edge weights (optional)
            edge_mask: [E] edge masks in [0,1] (optional)
            
        Returns:
            out: [N, out_channels] updated node features
        """
        # Apply linear transformation first
        x = torch.matmul(x, self.weight)
        
        # Combine edge weights and masks
        if edge_mask is not None:
            if edge_weight is not None:
                edge_weight = edge_weight * edge_mask
            else:
                edge_weight = edge_mask
        
        # Apply normalization if requested
        if self.normalize and edge_weight is not None:
            edge_weight = self._normalize_edge_weights(edge_index, edge_weight, x.size(0))
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Add bias
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def _normalize_edge_weights(self, edge_index, edge_weight, num_nodes):
        """Apply symmetric normalization like standard GCN
        
        Computes: D^(-1/2) * A * D^(-1/2) where A is adjacency matrix
        """
        row, col = edge_index[0], edge_index[1]
        
        # Compute degree
        deg = torch.zeros(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
        deg.scatter_add_(0, row, edge_weight)
        
        # Compute D^(-1/2)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Apply normalization
        edge_weight_norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        return edge_weight_norm
    
    def message(self, x_j, edge_weight):
        """Message function: how to compute messages"""
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class MaskedGATLayer(MessagePassing):
    """Graph Attention Network layer with edge masking support
    
    Extends GAT to support edge masks by multiplying attention weights
    with mask values.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension  
        heads (int): Number of attention heads
        concat (bool): Whether to concatenate multi-head outputs
        dropout (float): Dropout rate for attention weights
        bias (bool): Whether to use bias
    """
    
    def __init__(self, in_channels, out_channels, heads=1, concat=True, 
                 dropout=0.0, bias=True):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformations for each head
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        
        # Attention mechanism weights
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_mask=None):
        """
        Forward pass with optional edge masking
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity
            edge_mask: [E] edge masks in [0,1] (optional)
            
        Returns:
            out: [N, heads * out_channels] or [N, out_channels] node features
        """
        # Linear transformation
        x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)
        
        # Message passing with attention
        out = self.propagate(edge_index, x=x, edge_mask=edge_mask)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out += self.bias
        
        return out
    
    def message(self, x_i, x_j, edge_index_i, edge_mask, index, ptr, size_i):
        """Compute attention-weighted messages"""
        # Compute attention scores
        alpha = (x_i * self.att_src).sum(dim=-1) + (x_j * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=1)
        
        # Apply edge masks to attention weights
        if edge_mask is not None:
            alpha = alpha * edge_mask.view(-1, 1)
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'


class ResidualBlock(nn.Module):
    """Residual block for deeper GNN architectures
    
    Implements residual connections to help with training deeper networks.
    """
    
    def __init__(self, layer, input_dim, output_dim):
        super().__init__()
        self.layer = layer
        
        # Projection for residual connection if dimensions don't match
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, *args, **kwargs):
        """Forward pass with residual connection"""
        identity = self.projection(x)
        out = self.layer(x, *args, **kwargs)
        out = out + identity
        out = self.layer_norm(out)
        return out


def create_gnn_layer(layer_type, in_channels, out_channels, **kwargs):
    """Factory function to create different types of GNN layers
    
    Args:
        layer_type (str): Type of layer ('gcn', 'gat')
        in_channels (int): Input dimension
        out_channels (int): Output dimension
        **kwargs: Additional layer-specific arguments
        
    Returns:
        nn.Module: The requested GNN layer
    """
    if layer_type.lower() == 'gcn':
        return MaskedGCNLayer(in_channels, out_channels, **kwargs)
    elif layer_type.lower() == 'gat':
        return MaskedGATLayer(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}. Supported: 'gcn', 'gat'")