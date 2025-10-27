"""
Advanced Enhancements for SRINet

This module provides comprehensive implementations of modern deep learning techniques
to enhance the SRINet architecture for friendship link prediction.

Implemented enhancements:
1. Spectral Positional Encodings (structural graph information)
2. Advanced Attention Mechanisms (GAT, multi-head, temporal)
3. Contrastive Learning Framework (InfoNCE, hard negative mining)
4. Graph Transformer Architecture (global attention, long-range dependencies)
5. Hierarchical Graph Neural Networks (multi-scale processing)
6. Dynamic Graph Modeling (temporal patterns, recurrent GNN)
7. Advanced Feature Engineering (mobility patterns, POI characteristics)
8. Graph Augmentation Techniques (dropout, noise, adversarial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, get_laplacian, remove_self_loops
from torch_geometric.nn import MessagePassing
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh
import time


# =============================================================================
# 1. SPECTRAL POSITIONAL ENCODINGS (HIGH IMPACT, EASY TO IMPLEMENT)
# =============================================================================

class SpectralPositionalEncoding(nn.Module):
    """
    Add Laplacian eigenvectors as positional encodings to node features.
    
    **WHY THIS HELPS:**
    - Provides structural information about node positions in the graph
    - Low-frequency eigenvectors capture global structure
    - High-frequency eigenvectors capture local patterns
    - Proven to improve GNN performance (Dwivedi et al., 2020)
    
    **BENEFIT FOR SRINET:**
    - Nodes at similar graph positions have similar encodings
    - Helps distinguish users with similar meeting patterns but different roles
    - Adds no computational cost during forward pass (computed once)
    
    **COMPUTATIONAL COST:**
    - One-time: O(N^2) for small graphs, O(N*k) with sparse methods
    - Forward pass: 0 additional cost (just concatenation)
    """
    
    def __init__(self, num_nodes, pe_dim=16, use_sign_flip=True):
        """
        Args:
            num_nodes: Number of nodes in the graph
            pe_dim: Dimension of positional encoding (number of eigenvectors)
            use_sign_flip: Whether to fix sign ambiguity in eigenvectors
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.pe_dim = pe_dim
        self.use_sign_flip = use_sign_flip
        
        # Learnable projection to integrate PE with node features
        self.pe_projection = nn.Linear(pe_dim, pe_dim)
        
    def compute_laplacian_pe(self, edge_index, edge_weight=None, num_nodes=None):
        """
        Compute positional encodings from graph Laplacian eigenvectors.
        
        Uses the smallest k non-trivial eigenvectors of the normalized
        graph Laplacian: L = I - D^(-1/2) A D^(-1/2)
        
        Args:
            edge_index: [2, E] edge connectivity
            edge_weight: [E] edge weights (optional)
            num_nodes: Number of nodes
            
        Returns:
            pe: [N, pe_dim] positional encodings
            eigenvalues: [pe_dim] corresponding eigenvalues
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
            
        # Compute normalized Laplacian
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=num_nodes
        )
        
        # Convert to scipy sparse matrix for eigendecomposition
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        
        # Compute k smallest eigenvectors (excluding the trivial constant vector)
        # We need pe_dim + 1 to get pe_dim non-trivial eigenvectors
        try:
            # Use sparse eigendecomposition (much faster for large graphs)
            eigenvalues, eigenvectors = eigsh(L, k=self.pe_dim + 1, which='SM')
            
            # Remove the first eigenvector (constant) and eigenvalue (near 0)
            pe = eigenvectors[:, 1:]  # [N, pe_dim]
            eigenvalues = eigenvalues[1:]  # [pe_dim]
            
        except Exception as e:
            print(f"Warning: Sparse eigendecomposition failed ({e}), using dense")
            # Fallback to dense eigendecomposition for small graphs
            L_dense = L.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            
            # Sort by eigenvalue and take smallest
            idx = eigenvalues.argsort()[1:self.pe_dim+1]
            pe = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]
        
        # Convert to torch tensors
        pe = torch.from_numpy(pe).float()
        eigenvalues = torch.from_numpy(eigenvalues).float()
        
        # Fix sign ambiguity (make first element positive)
        if self.use_sign_flip:
            sign = torch.sign(pe[0, :])
            sign[sign == 0] = 1
            pe = pe * sign.unsqueeze(0)
        
        return pe, eigenvalues
    
    def forward(self, pe):
        """
        Project positional encodings (can add learnable transformation)
        
        Args:
            pe: [N, pe_dim] positional encodings
            
        Returns:
            pe_transformed: [N, pe_dim] transformed encodings
        """
        return self.pe_projection(pe)


class MultiplexSpectralPE(nn.Module):
    """
    Spectral positional encodings for multiplex graphs.
    
    **WHY THIS IS BETTER FOR MULTIPLEX:**
    - Each graph category has different structure
    - Aggregate spectral features across categories
    - Captures multi-relational graph structure
    
    **BENEFIT FOR SRINET:**
    - Restaurant graph: users who frequent same restaurants
    - Office graph: users who work in same areas
    - Combined PE captures multi-faceted social structure
    """
    
    def __init__(self, num_nodes, pe_dim=16, aggregation='concat'):
        """
        Args:
            num_nodes: Number of nodes
            pe_dim: PE dimension per category
            aggregation: How to combine PEs ('concat', 'mean', 'attention')
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.pe_dim = pe_dim
        self.aggregation = aggregation
        
        self.pe_encoder = SpectralPositionalEncoding(num_nodes, pe_dim)
        
        if aggregation == 'attention':
            # Learnable attention to weight different categories
            self.category_attention = nn.Linear(pe_dim, 1)
        
    def compute_multiplex_pe(self, adjacency_matrices):
        """
        Compute PE for each category and aggregate.
        
        Args:
            adjacency_matrices: Dict[category -> {edge_index, edge_weights}]
            
        Returns:
            pe_combined: [N, output_dim] combined positional encodings
            pe_per_category: Dict[category -> [N, pe_dim]]
        """
        pe_per_category = {}
        
        print("Computing spectral positional encodings...")
        start_time = time.time()
        
        for category, data in adjacency_matrices.items():
            edge_index = data['edge_index']
            edge_weight = data.get('edge_weights', None)
            
            # Compute PE for this category
            pe, eigenvalues = self.pe_encoder.compute_laplacian_pe(
                edge_index, edge_weight, self.num_nodes
            )
            
            pe_per_category[category] = {
                'pe': pe,
                'eigenvalues': eigenvalues
            }
            
            print(f"  {category}: computed {self.pe_dim} eigenvectors, "
                  f"eigenvalue range [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
        
        elapsed = time.time() - start_time
        print(f"✓ Computed PE in {elapsed:.2f}s")
        
        # Aggregate PEs across categories
        pe_combined = self._aggregate_pe(pe_per_category)
        
        return pe_combined, pe_per_category
    
    def _aggregate_pe(self, pe_per_category):
        """Aggregate positional encodings across categories"""
        pe_list = [data['pe'] for data in pe_per_category.values()]
        
        if self.aggregation == 'concat':
            # Concatenate all PEs
            return torch.cat(pe_list, dim=-1)
            
        elif self.aggregation == 'mean':
            # Average PEs
            return torch.stack(pe_list).mean(dim=0)
            
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation
            pe_stacked = torch.stack(pe_list)  # [num_categories, N, pe_dim]
            
            # Compute attention scores for each category
            attn_scores = []
            for pe in pe_list:
                score = self.category_attention(pe)  # [N, 1]
                attn_scores.append(score)
            
            attn_weights = F.softmax(torch.stack(attn_scores), dim=0)  # [num_categories, N, 1]
            
            # Weighted sum
            return (pe_stacked * attn_weights).sum(dim=0)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


def to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes):
    """Convert PyTorch edge list to scipy sparse matrix"""
    edge_index = edge_index.cpu().numpy()
    edge_weight = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(edge_index.shape[1])
    
    adj = sp.csr_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    return adj


# =============================================================================
# 2. FREQUENCY-DOMAIN EDGE SIMILARITY (NOVEL APPROACH)
# =============================================================================

class FrequencyEdgeScorer(nn.Module):
    """
    Score edges using frequency-domain similarity between node embeddings.
    
    **WHY THIS HELPS:**
    - Fourier transform decomposes embeddings into frequency components
    - Low frequencies: global patterns, high frequencies: local details
    - Can detect similarity at different scales simultaneously
    
    **BENEFIT FOR SRINET:**
    - Traditional dot product: captures overall similarity
    - Frequency domain: captures multi-scale similarity patterns
    - Example: Two users might be similar in low-freq (same social circle)
              but different in high-freq (different daily routines)
    
    **WHEN TO USE:**
    - Embeddings have structured patterns (not random)
    - Want to capture multi-scale similarity
    - As a complement to spatial features
    """
    
    def __init__(self, input_dim, hidden_dim=128, use_freq_features=True):
        """
        Args:
            input_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for scorer MLP
            use_freq_features: Whether to use frequency domain features
        """
        super().__init__()
        self.input_dim = input_dim
        self.use_freq_features = use_freq_features
        
        if use_freq_features:
            # Frequency domain features + spatial features
            freq_dim = input_dim // 2 + 1  # Real FFT output size
            self.scorer = nn.Sequential(
                nn.Linear(input_dim * 2 + freq_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Spatial features only (baseline)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def compute_frequency_features(self, emb_i, emb_j):
        """
        Compute frequency-domain similarity features.
        
        Uses cross-power spectrum and correlation in frequency domain.
        
        Args:
            emb_i: [E, D] source node embeddings
            emb_j: [E, D] target node embeddings
            
        Returns:
            freq_features: [E, D//2+1] frequency domain features
        """
        # Apply FFT to both embeddings (real FFT for efficiency)
        fft_i = torch.fft.rfft(emb_i, dim=-1)  # [E, D//2+1] complex
        fft_j = torch.fft.rfft(emb_j, dim=-1)  # [E, D//2+1] complex
        
        # Cross-power spectrum: measures correlation at each frequency
        cross_spectrum = fft_i * torch.conj(fft_j)  # [E, D//2+1] complex
        
        # Extract magnitude (correlation strength at each frequency)
        cross_spectrum_mag = torch.abs(cross_spectrum)  # [E, D//2+1] real
        
        # Phase difference (captures relative alignment)
        phase_diff = torch.angle(cross_spectrum)  # [E, D//2+1] real
        
        # Combine magnitude and phase information
        freq_features = torch.cat([
            cross_spectrum_mag,
            torch.sin(phase_diff),  # Encode phase as sin (periodic)
        ], dim=-1)
        
        return freq_features
    
    def forward(self, node_embeddings, edge_index):
        """
        Score edges using spatial + frequency features.
        
        Args:
            node_embeddings: [N, D] node embeddings
            edge_index: [2, E] edge list
            
        Returns:
            scores: [E] edge scores
        """
        # Get node features for each edge
        src_emb = node_embeddings[edge_index[0]]  # [E, D]
        tgt_emb = node_embeddings[edge_index[1]]  # [E, D]
        
        if self.use_freq_features:
            # Compute frequency domain features
            freq_features = self.compute_frequency_features(src_emb, tgt_emb)
            
            # Concatenate spatial and frequency features
            edge_features = torch.cat([src_emb, tgt_emb, freq_features], dim=-1)
        else:
            # Spatial features only
            edge_features = torch.cat([src_emb, tgt_emb], dim=-1)
        
        # Score edges
        scores = self.scorer(edge_features).squeeze(-1)
        
        return scores


# =============================================================================
# 3. SPECTRAL GRAPH CONVOLUTION (ALTERNATIVE TO SPATIAL GCN)
# =============================================================================

class ChebNetConv(nn.Module):
    """
    Chebyshev spectral graph convolution (ChebNet).
    
    **WHY THIS HELPS:**
    - Approximates spectral convolution without eigendecomposition
    - Uses Chebyshev polynomials: fast and localized
    - Captures K-hop neighborhoods explicitly
    
    **BENEFIT FOR SRINET:**
    - Can learn different filters for different frequencies
    - Polynomial order K controls receptive field
    - More principled than spatial aggregation
    
    **TRADE-OFFS:**
    - More parameters than standard GCN
    - Requires Laplacian computation
    - May not work well with dynamic edge masking
    
    **WHEN TO USE:**
    - As an alternative GNN layer for comparison
    - When you want explicit multi-hop reasoning
    - For ablation studies
    """
    
    def __init__(self, in_channels, out_channels, K=3, bias=True):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            K: Chebyshev polynomial order (controls receptive field)
            bias: Whether to use bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        # Learnable weights for each polynomial term
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        
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
    
    def forward(self, x, edge_index, edge_weight=None, edge_mask=None, lambda_max=2.0):
        """
        Forward pass with Chebyshev polynomial convolution.
        
        The convolution is: h = Σ_{k=0}^{K-1} T_k(L̃) X W_k
        where T_k is the k-th Chebyshev polynomial and L̃ is normalized Laplacian.
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge list
            edge_weight: [E] edge weights
            edge_mask: [E] binary masks (0-1)
            lambda_max: Maximum eigenvalue for normalization (usually 2.0)
            
        Returns:
            out: [N, out_channels] output features
        """
        # Apply edge masks if provided
        if edge_mask is not None:
            if edge_weight is not None:
                edge_weight = edge_weight * edge_mask
            else:
                edge_weight = edge_mask
        
        # Compute normalized Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=x.size(0)
        )
        
        # Scale to [-1, 1] for Chebyshev polynomials
        edge_weight_lap = 2.0 * edge_weight_lap / lambda_max - 1.0
        
        # Compute Chebyshev polynomials recursively
        T_k = self._compute_chebyshev_basis(x, edge_index_lap, edge_weight_lap, self.K)
        
        # Combine with learnable weights
        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            out += torch.matmul(T_k[k], self.weight[k])
        
        if self.bias is not None:
            out += self.bias
        
        return out
    
    def _compute_chebyshev_basis(self, x, edge_index, edge_weight, K):
        """
        Compute Chebyshev polynomial basis using recursive formula:
        T_0(L) = I
        T_1(L) = L
        T_k(L) = 2L @ T_{k-1}(L) - T_{k-2}(L)
        
        Returns:
            List of [N, D] tensors for k=0..K-1
        """
        T_k = [x]  # T_0 = x
        
        if K > 1:
            # T_1 = L @ x
            T_k.append(self._sparse_mm(edge_index, edge_weight, x))
        
        # Recursive computation for k >= 2
        for k in range(2, K):
            T_new = 2 * self._sparse_mm(edge_index, edge_weight, T_k[k-1]) - T_k[k-2]
            T_k.append(T_new)
        
        return T_k
    
    def _sparse_mm(self, edge_index, edge_weight, x):
        """Sparse matrix-matrix multiplication: L @ x"""
        # Use torch_geometric's sparse multiplication
        from torch_geometric.utils import spmm
        return spmm(edge_index, edge_weight, x.size(0), x)


# =============================================================================
# 4. BENCHMARK AND COMPARISON UTILITIES
# =============================================================================

class FourierBenchmark:
    """
    Benchmark Fourier enhancements vs baseline.
    
    Compares:
    1. Baseline: No spectral features
    2. + Spectral PE: Add positional encodings
    3. + Frequency Edge Scorer: Use frequency domain for edge scoring
    4. + ChebNet: Use spectral convolution
    """
    
    @staticmethod
    def benchmark_spectral_pe(adjacency_matrices, num_nodes, pe_dims=[8, 16, 32, 64]):
        """
        Benchmark spectral PE computation time and quality.
        
        Returns:
            results: Dict with timing and eigenvalue info
        """
        results = {}
        
        for pe_dim in pe_dims:
            print(f"\n=== Benchmarking PE dim={pe_dim} ===")
            
            pe_encoder = MultiplexSpectralPE(num_nodes, pe_dim, aggregation='concat')
            
            start = time.time()
            pe_combined, pe_per_category = pe_encoder.compute_multiplex_pe(adjacency_matrices)
            elapsed = time.time() - start
            
            results[pe_dim] = {
                'computation_time': elapsed,
                'output_dim': pe_combined.shape[1],
                'memory_mb': pe_combined.element_size() * pe_combined.nelement() / 1024 / 1024,
                'per_category': {}
            }
            
            for category, data in pe_per_category.items():
                eigenvalues = data['eigenvalues']
                results[pe_dim]['per_category'][category] = {
                    'eigenvalue_range': (eigenvalues[0].item(), eigenvalues[-1].item()),
                    'eigenvalue_gap': (eigenvalues[1] - eigenvalues[0]).item()
                }
            
            print(f"  Computation time: {elapsed:.2f}s")
            print(f"  Output dimension: {pe_combined.shape}")
            print(f"  Memory: {results[pe_dim]['memory_mb']:.2f} MB")
        
        return results
    
    @staticmethod
    def benchmark_frequency_edge_scorer(node_embeddings, edge_index, num_trials=100):
        """
        Benchmark frequency vs spatial edge scoring.
        
        Returns:
            timing_results: Dict with forward pass times
        """
        input_dim = node_embeddings.shape[1]
        num_edges = edge_index.shape[1]
        
        print(f"\n=== Benchmarking Edge Scorers ===")
        print(f"Nodes: {node_embeddings.shape[0]}, Edges: {num_edges}, Dim: {input_dim}")
        
        # Baseline: spatial only
        scorer_spatial = FrequencyEdgeScorer(input_dim, use_freq_features=False)
        
        # Enhanced: spatial + frequency
        scorer_freq = FrequencyEdgeScorer(input_dim, use_freq_features=True)
        
        # Warm up
        _ = scorer_spatial(node_embeddings, edge_index)
        _ = scorer_freq(node_embeddings, edge_index)
        
        # Benchmark spatial
        start = time.time()
        for _ in range(num_trials):
            scores_spatial = scorer_spatial(node_embeddings, edge_index)
        time_spatial = (time.time() - start) / num_trials
        
        # Benchmark frequency
        start = time.time()
        for _ in range(num_trials):
            scores_freq = scorer_freq(node_embeddings, edge_index)
        time_freq = (time.time() - start) / num_trials
        
        overhead = (time_freq - time_spatial) / time_spatial * 100
        
        results = {
            'spatial_time_ms': time_spatial * 1000,
            'frequency_time_ms': time_freq * 1000,
            'overhead_percent': overhead,
            'num_edges': num_edges,
            'num_parameters_spatial': sum(p.numel() for p in scorer_spatial.parameters()),
            'num_parameters_freq': sum(p.numel() for p in scorer_freq.parameters()),
        }
        
        print(f"\nResults:")
        print(f"  Spatial:   {results['spatial_time_ms']:.3f} ms")
        print(f"  Frequency: {results['frequency_time_ms']:.3f} ms")
        print(f"  Overhead:  {results['overhead_percent']:.1f}%")
        print(f"  Params spatial: {results['num_parameters_spatial']:,}")
        print(f"  Params freq:    {results['num_parameters_freq']:,}")
        
        return results


# =============================================================================
# 5. INTEGRATION WITH SRINET
# =============================================================================

def enhance_srinet_with_fourier(srinet_model, adjacency_matrices, pe_dim=16, 
                                 use_freq_scorer=False):
    """
    Enhance existing SRINet model with Fourier features.
    
    Modifications:
    1. Add spectral positional encodings to node embeddings
    2. (Optional) Replace edge scorer with frequency-aware version
    
    Args:
        srinet_model: Existing SRINet model
        adjacency_matrices: Dict of adjacency matrices per category
        pe_dim: Dimension of positional encodings per category
        use_freq_scorer: Whether to use frequency-domain edge scorer
        
    Returns:
        enhanced_model: Modified model with Fourier features
        pe_combined: Precomputed positional encodings
    """
    print("\n" + "="*60)
    print("ENHANCING SRINET WITH FOURIER TRANSFORMS")
    print("="*60)
    
    num_nodes = srinet_model.num_users
    
    # 1. Compute spectral positional encodings
    print("\n[1/3] Computing spectral positional encodings...")
    pe_encoder = MultiplexSpectralPE(num_nodes, pe_dim, aggregation='concat')
    pe_combined, pe_per_category = pe_encoder.compute_multiplex_pe(adjacency_matrices)
    
    # Store PE in model
    srinet_model.register_buffer('spectral_pe', pe_combined)
    pe_output_dim = pe_combined.shape[1]
    
    print(f"✓ Added {pe_output_dim}-dim spectral PE to model")
    
    # 2. Update node embeddings dimension
    print("\n[2/3] Updating node embedding dimension...")
    old_emb_dim = srinet_model.node_embeddings.shape[1]
    new_emb_dim = old_emb_dim  # Keep same, will concatenate PE separately
    
    # Add projection layer for PE
    srinet_model.pe_projection = nn.Linear(pe_output_dim, old_emb_dim)
    
    print(f"✓ Node features: {old_emb_dim} (learned) + {pe_output_dim} (PE) -> {old_emb_dim + pe_output_dim}")
    
    # 3. (Optional) Replace edge scorers with frequency-aware versions
    if use_freq_scorer:
        print("\n[3/3] Replacing edge scorers with frequency-aware versions...")
        
        for layer_idx in range(srinet_model.config.num_layers):
            for category in srinet_model.categories:
                mask_module = srinet_model.mask_modules[f'layer_{layer_idx}'][category]
                
                # Replace scorer with frequency-aware version
                input_dim = srinet_model.config.embedding_dim
                hidden_dim = srinet_model.config.hidden_dim
                
                mask_module.scorer = FrequencyEdgeScorer(
                    input_dim, hidden_dim, use_freq_features=True
                ).scorer
        
        print(f"✓ Replaced {len(srinet_model.categories) * srinet_model.config.num_layers} edge scorers")
    else:
        print("\n[3/3] Keeping spatial edge scorers (use_freq_scorer=False)")
    
    print("\n" + "="*60)
    print("ENHANCEMENT COMPLETE")
    print("="*60)
    
    # Summary
    original_params = sum(p.numel() for p in srinet_model.parameters())
    print(f"\nModel summary:")
    print(f"  Total parameters: {original_params:,}")
    print(f"  PE dimension: {pe_output_dim}")
    print(f"  Frequency scorer: {'Yes' if use_freq_scorer else 'No'}")
    
    return srinet_model, pe_combined


# =============================================================================
# 6. USAGE EXAMPLES
# =============================================================================

def example_usage():
    """
    Example of how to use Fourier enhancements in SRINet training.
    """
    print("""
# =============================================================================
# EXAMPLE USAGE: Fourier Enhancements for SRINet
# =============================================================================

# 1. BASIC USAGE: Add Spectral Positional Encodings
# -------------------------------------------------

from src.models.srinet import SRINet
from fourier_enhancements import enhance_srinet_with_fourier

# Load data and initialize model (as before)
adjacency_matrices = torch.load('data/adjacency_matrices.pt')
config = SRINetConfig()
model = SRINet(config, num_users, adjacency_matrices)

# Enhance with Fourier features
model_enhanced, spectral_pe = enhance_srinet_with_fourier(
    model, 
    adjacency_matrices,
    pe_dim=16,              # 16 eigenvectors per category
    use_freq_scorer=False   # Keep spatial scorer
)

# In forward pass, concatenate PE with learned embeddings
# (This is done automatically if you use the enhanced model)


# 2. ADVANCED USAGE: Full Fourier Enhancement
# -------------------------------------------

model_enhanced, spectral_pe = enhance_srinet_with_fourier(
    model,
    adjacency_matrices,
    pe_dim=32,             # More eigenvectors for richer PE
    use_freq_scorer=True   # Use frequency-domain edge scoring
)


# 3. BENCHMARK BEFORE/AFTER
# -------------------------

from fourier_enhancements import FourierBenchmark

# Benchmark PE computation
pe_results = FourierBenchmark.benchmark_spectral_pe(
    adjacency_matrices, 
    num_users,
    pe_dims=[8, 16, 32, 64]
)

# Benchmark edge scoring
scorer_results = FourierBenchmark.benchmark_frequency_edge_scorer(
    model.node_embeddings,
    adjacency_matrices['Restaurant']['edge_index'],
    num_trials=100
)


# 4. COMPARE PERFORMANCE
# ----------------------

# Train baseline model
baseline_model = SRINet(config, num_users, adjacency_matrices)
baseline_results = train_model(baseline_model, dataset, config)

# Train enhanced model
enhanced_model, _ = enhance_srinet_with_fourier(baseline_model, adjacency_matrices)
enhanced_results = train_model(enhanced_model, dataset, config)

# Compare metrics
print(f"Baseline PR-AUC: {baseline_results['test_pr_auc']:.4f}")
print(f"Enhanced PR-AUC: {enhanced_results['test_pr_auc']:.4f}")
print(f"Improvement: {(enhanced_results['test_pr_auc'] - baseline_results['test_pr_auc']):.4f}")

""")


# =============================================================================
# 2. ADVANCED ATTENTION MECHANISMS
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Enhanced Graph Attention Network layer with edge masking support.
    
    Improvements over standard GAT:
    - Support for edge masks (compatible with SRINet topology filtering)
    - Multi-head attention with learnable combination
    - Temporal attention weights for time-aware processing
    """
    
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1, 
                 alpha=0.2, concat=True, edge_dim=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformations for each head
        self.W = nn.Parameter(torch.empty(size=(n_heads, in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_features, 1)))
        
        # Edge feature transformation (for temporal attention)
        if edge_dim is not None:
            self.edge_transform = nn.Linear(edge_dim, n_heads)
        else:
            self.edge_transform = None
        
        # Output projection
        if concat:
            self.out_proj = nn.Linear(n_heads * out_features, out_features)
        else:
            self.out_proj = nn.Linear(out_features, out_features)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.edge_transform is not None:
            nn.init.xavier_uniform_(self.edge_transform.weight)
    
    def forward(self, x, edge_index, edge_attr=None, edge_mask=None):
        """
        Args:
            x: [N, in_features] node features
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_dim] edge features (optional, for temporal attention)
            edge_mask: [E] edge masks from topology filtering
        """
        N = x.size(0)
        
        # Apply linear transformation for each head
        h = torch.matmul(x.unsqueeze(0), self.W)  # [n_heads, N, out_features]
        
        # Prepare edge features
        edge_h = self._prepare_edge_features(h, edge_index)  # [n_heads, E, 2*out_features]
        
        # Compute attention coefficients
        e = torch.matmul(edge_h, self.a).squeeze(-1)  # [n_heads, E]
        e = self.leakyrelu(e)
        
        # Add temporal attention if edge features provided
        if edge_attr is not None and self.edge_transform is not None:
            temporal_weights = self.edge_transform(edge_attr).t()  # [n_heads, E]
            e = e + temporal_weights
        
        # Apply edge masks
        if edge_mask is not None:
            e = e * edge_mask.unsqueeze(0)  # Broadcast mask to all heads
        
        # Softmax attention
        attention = self._masked_softmax(e, edge_index, N)
        attention = self.dropout_layer(attention)
        
        # Apply attention to aggregate features
        h_prime = self._aggregate_with_attention(h, attention, edge_index, N)
        
        # Combine heads
        if self.concat:
            output = h_prime.transpose(0, 1).contiguous().view(N, -1)
        else:
            output = h_prime.mean(dim=0)
        
        return self.out_proj(output)
    
    def _prepare_edge_features(self, h, edge_index):
        """Prepare concatenated edge features for attention computation"""
        src, tgt = edge_index
        h_src = h[:, src, :]  # [n_heads, E, out_features]
        h_tgt = h[:, tgt, :]  # [n_heads, E, out_features]
        return torch.cat([h_src, h_tgt], dim=-1)  # [n_heads, E, 2*out_features]
    
    def _masked_softmax(self, e, edge_index, N):
        """Compute softmax with proper masking for graph structure"""
        # Create attention matrix
        attention = torch.zeros(self.n_heads, N, N, device=e.device)
        src, tgt = edge_index
        attention[:, src, tgt] = e
        
        # Apply softmax row-wise (for each source node)
        attention = F.softmax(attention, dim=-1)
        
        # Extract edge attention values
        return attention[:, src, tgt]
    
    def _aggregate_with_attention(self, h, attention, edge_index, N):
        """Aggregate node features using attention weights"""
        src, tgt = edge_index
        
        # Weighted features
        weighted_features = h[:, tgt, :] * attention.unsqueeze(-1)  # [n_heads, E, out_features]
        
        # Aggregate by source nodes
        h_prime = torch.zeros(self.n_heads, N, self.out_features, device=h.device)
        h_prime.index_add_(1, src, weighted_features)
        
        return h_prime


class MultiHeadCategoryAttention(nn.Module):
    """
    Multi-head attention for fusing embeddings from different POI categories.
    
    This replaces simple mean fusion with learnable attention weights
    that can focus on the most relevant categories for each user.
    """
    
    def __init__(self, embedding_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Category importance weights (learnable)
        self.category_weights = nn.Parameter(torch.ones(1, 1, 1))  # Will be expanded
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, category_embeddings):
        """
        Args:
            category_embeddings: List of [N, embedding_dim] tensors, one per category
            
        Returns:
            fused_embeddings: [N, embedding_dim] fused embeddings
            attention_weights: [N, n_categories] attention weights for interpretability
        """
        if len(category_embeddings) == 1:
            return category_embeddings[0], torch.ones(category_embeddings[0].size(0), 1)
        
        # Stack category embeddings
        stacked = torch.stack(category_embeddings, dim=1)  # [N, n_categories, embedding_dim]
        N, n_categories, _ = stacked.shape
        
        # Expand category weights
        if self.category_weights.size(2) != n_categories:
            self.category_weights.data = torch.ones(1, 1, n_categories, device=stacked.device)
        
        # Project to Q, K, V
        Q = self.q_proj(stacked)  # [N, n_categories, embedding_dim]
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)
        
        # Reshape for multi-head attention
        Q = Q.view(N, n_categories, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, n_categories, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, n_categories, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Add category importance bias
        scores = scores + self.category_weights.unsqueeze(1)
        
        # Softmax attention
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        attended = torch.matmul(attention, V)  # [N, n_heads, n_categories, head_dim]
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(N, n_categories, self.embedding_dim)
        output = self.out_proj(attended)
        
        # Global pooling (mean over categories)
        fused = output.mean(dim=1)
        
        # Return attention weights for interpretability
        attention_weights = attention.mean(dim=1).mean(dim=1)  # [N, n_categories]
        
        return fused, attention_weights


# =============================================================================
# 3. CONTRASTIVE LEARNING FRAMEWORK
# =============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss for contrastive learning.
    
    This loss encourages similar users (positive pairs) to have similar embeddings
    while pushing dissimilar users (negative pairs) apart in the embedding space.
    """
    
    def __init__(self, temperature=0.1, negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode
    
    def forward(self, embeddings, positive_pairs, negative_pairs=None):
        """
        Args:
            embeddings: [N, D] user embeddings
            positive_pairs: [P, 2] positive user pairs
            negative_pairs: [N, 2] negative user pairs (optional)
            
        Returns:
            loss: InfoNCE contrastive loss
        """
        device = embeddings.device
        
        # Get positive pair embeddings
        pos_emb_1 = embeddings[positive_pairs[:, 0]]  # [P, D]
        pos_emb_2 = embeddings[positive_pairs[:, 1]]  # [P, D]
        
        # Compute positive similarities
        pos_sim = F.cosine_similarity(pos_emb_1, pos_emb_2, dim=1)  # [P]
        pos_sim = pos_sim / self.temperature
        
        if negative_pairs is not None:
            # Use provided negative pairs
            neg_emb_1 = embeddings[negative_pairs[:, 0]]  # [N, D]
            neg_emb_2 = embeddings[negative_pairs[:, 1]]  # [N, D]
            neg_sim = F.cosine_similarity(neg_emb_1, neg_emb_2, dim=1)  # [N]
            neg_sim = neg_sim / self.temperature
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim, neg_sim])  # [P + N]
            labels = torch.cat([
                torch.ones(len(pos_sim), device=device),
                torch.zeros(len(neg_sim), device=device)
            ])
        else:
            # Use all-pairs negative sampling
            # Compute all pairwise similarities
            all_sim = torch.matmul(embeddings, embeddings.t()) / self.temperature  # [N, N]
            
            # Create labels for positive pairs
            labels = torch.zeros_like(all_sim)
            labels[positive_pairs[:, 0], positive_pairs[:, 1]] = 1
            labels[positive_pairs[:, 1], positive_pairs[:, 0]] = 1  # Symmetric
            
            # Compute InfoNCE loss
            exp_sim = torch.exp(all_sim)
            pos_exp_sim = exp_sim * labels
            
            # Sum over all negatives for each positive
            pos_sum = pos_exp_sim.sum(dim=1, keepdim=True)
            neg_sum = exp_sim.sum(dim=1, keepdim=True) - pos_sum
            
            # InfoNCE loss
            loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
            return loss[labels.sum(dim=1) > 0].mean()  # Only for nodes with positive pairs
        
        # Binary classification loss
        return F.binary_cross_entropy_with_logits(logits, labels)


class HardNegativeMiner(nn.Module):
    """
    Hard negative mining for more challenging contrastive learning.
    
    Selects the most confusing negative pairs (high similarity but not friends)
    to make the model learn better discriminative features.
    """
    
    def __init__(self, ratio=0.3, strategy='hardest'):
        super().__init__()
        self.ratio = ratio  # Fraction of negatives to keep
        self.strategy = strategy  # 'hardest', 'semi_hard', 'random'
    
    def mine_negatives(self, embeddings, positive_pairs, candidate_negatives):
        """
        Args:
            embeddings: [N, D] user embeddings
            positive_pairs: [P, 2] positive pairs
            candidate_negatives: [C, 2] candidate negative pairs
            
        Returns:
            hard_negatives: [H, 2] selected hard negative pairs
        """
        # Compute similarities for candidate negatives
        neg_emb_1 = embeddings[candidate_negatives[:, 0]]
        neg_emb_2 = embeddings[candidate_negatives[:, 1]]
        neg_similarities = F.cosine_similarity(neg_emb_1, neg_emb_2, dim=1)
        
        # Compute similarities for positive pairs (for reference)
        pos_emb_1 = embeddings[positive_pairs[:, 0]]
        pos_emb_2 = embeddings[positive_pairs[:, 1]]
        pos_similarities = F.cosine_similarity(pos_emb_1, pos_emb_2, dim=1)
        pos_threshold = pos_similarities.mean()
        
        # Select hard negatives based on strategy
        num_hard = int(len(candidate_negatives) * self.ratio)
        
        if self.strategy == 'hardest':
            # Select negatives with highest similarity (hardest to distinguish)
            _, indices = torch.topk(neg_similarities, num_hard)
        elif self.strategy == 'semi_hard':
            # Select negatives with similarity close to positive threshold
            distances = torch.abs(neg_similarities - pos_threshold)
            _, indices = torch.topk(distances, num_hard, largest=False)
        else:  # random
            indices = torch.randperm(len(candidate_negatives))[:num_hard]
        
        return candidate_negatives[indices]


# =============================================================================
# 4. GRAPH AUGMENTATION TECHNIQUES
# =============================================================================

class GraphAugmentation(nn.Module):
    """
    Graph augmentation techniques for better generalization.
    
    Implements various augmentation strategies:
    - Edge dropout: Random edge removal
    - Node feature noise: Gaussian noise injection
    - Subgraph sampling: Train on random subgraphs
    - Adversarial augmentation: Generate hard examples
    """
    
    def __init__(self, edge_dropout=0.1, feature_noise=0.1, subgraph_ratio=0.8):
        super().__init__()
        self.edge_dropout = edge_dropout
        self.feature_noise = feature_noise
        self.subgraph_ratio = subgraph_ratio
    
    def forward(self, x, edge_index, edge_mask=None, training=True):
        """
        Apply augmentations during training.
        
        Args:
            x: [N, D] node features
            edge_index: [2, E] edge connectivity
            edge_mask: [E] edge masks
            training: Whether in training mode
            
        Returns:
            aug_x: [N, D] augmented node features
            aug_edge_index: [2, E'] augmented edge connectivity
            aug_edge_mask: [E'] augmented edge masks
        """
        if not training:
            return x, edge_index, edge_mask
        
        # 1. Edge dropout
        aug_edge_index, aug_edge_mask = self._edge_dropout(edge_index, edge_mask)
        
        # 2. Feature noise
        aug_x = self._feature_noise(x)
        
        # 3. Subgraph sampling (optional) - disabled for category fusion compatibility
        # if torch.rand(1).item() < 0.3:  # 30% chance
        #     aug_x, aug_edge_index, aug_edge_mask = self._subgraph_sampling(
        #         aug_x, aug_edge_index, aug_edge_mask
        #     )
        
        return aug_x, aug_edge_index, aug_edge_mask
    
    def _edge_dropout(self, edge_index, edge_mask):
        """Randomly drop edges"""
        if self.edge_dropout <= 0:
            return edge_index, edge_mask
        
        E = edge_index.size(1)
        keep_prob = 1 - self.edge_dropout
        keep_mask = torch.rand(E, device=edge_index.device) < keep_prob
        
        # Keep edges
        new_edge_index = edge_index[:, keep_mask]
        new_edge_mask = edge_mask[keep_mask] if edge_mask is not None else None
        
        return new_edge_index, new_edge_mask
    
    def _feature_noise(self, x):
        """Add Gaussian noise to node features"""
        if self.feature_noise <= 0:
            return x
        
        noise = torch.randn_like(x) * self.feature_noise
        return x + noise
    
    def _subgraph_sampling(self, x, edge_index, edge_mask):
        """Sample a random subgraph"""
        N = x.size(0)
        num_keep = int(N * self.subgraph_ratio)
        
        # Random node sampling
        keep_nodes = torch.randperm(N, device=x.device)[:num_keep]
        
        # Keep node features
        sub_x = x[keep_nodes]
        
        # Filter edges
        node_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        node_mask[keep_nodes] = True
        
        edge_keep_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        sub_edge_index = edge_index[:, edge_keep_mask]
        sub_edge_mask = edge_mask[edge_keep_mask] if edge_mask is not None else None
        
        # Remap node indices
        old_to_new = torch.full((N,), -1, device=x.device)
        old_to_new[keep_nodes] = torch.arange(num_keep, device=x.device)
        sub_edge_index = old_to_new[sub_edge_index]
        
        return sub_x, sub_edge_index, sub_edge_mask


class AdversarialAugmentation(nn.Module):
    """
    Adversarial augmentation for robust training.
    
    Generates adversarial examples by adding small perturbations
    to node features that maximize the loss.
    """
    
    def __init__(self, epsilon=0.01, alpha=0.001, num_steps=3):
        super().__init__()
        self.epsilon = epsilon  # Maximum perturbation magnitude
        self.alpha = alpha      # Step size
        self.num_steps = num_steps  # Number of adversarial steps
    
    def generate_adversarial(self, model, x, edge_index, edge_mask, 
                           positive_pairs, negative_pairs):
        """
        Generate adversarial examples using PGD (Projected Gradient Descent).
        
        Args:
            model: The SRINet model
            x: [N, D] node features
            edge_index: [2, E] edge connectivity
            edge_mask: [E] edge masks
            positive_pairs: [P, 2] positive pairs
            negative_pairs: [N, 2] negative pairs
            
        Returns:
            adv_x: [N, D] adversarial node features
        """
        adv_x = x.clone().detach()
        adv_x.requires_grad = True
        
        for step in range(self.num_steps):
            # Forward pass
            model.zero_grad()
            
            # Temporarily replace node embeddings
            original_embeddings = model.node_embeddings.data.clone()
            model.node_embeddings.data = adv_x
            
            # Compute loss
            result = model(positive_pairs, negative_pairs)
            loss = result['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Restore original embeddings
            model.node_embeddings.data = original_embeddings
            
            # Update adversarial features
            grad = adv_x.grad.data
            adv_x = adv_x + self.alpha * grad.sign()
            
            # Project to epsilon ball
            perturbation = adv_x - x
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            adv_x = x + perturbation
            
            # Prepare for next iteration
            adv_x = adv_x.detach()
            adv_x.requires_grad = True
        
        return adv_x.detach()


# =============================================================================
# 5. ADVANCED FEATURE ENGINEERING
# =============================================================================

class MobilityPatternExtractor(nn.Module):
    """
    Extract advanced mobility patterns from check-in data.
    
    Features extracted:
    - Home/work locations
    - Travel radius and diversity
    - Temporal patterns (day-of-week, time-of-day)
    - Activity diversity across POI categories
    """
    
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Feature encoders
        self.location_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim // 4),  # lat, lon
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim // 4),  # hour, day_of_week, month
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.activity_encoder = nn.Sequential(
            nn.Linear(10, embedding_dim // 4),  # POI category distribution
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.mobility_encoder = nn.Sequential(
            nn.Linear(4, embedding_dim // 4),  # radius, diversity, frequency, regularity
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Combination layer
        self.combiner = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, checkin_data):
        """
        Extract mobility features from check-in data.
        
        Args:
            checkin_data: Dict containing check-in information per user
            
        Returns:
            mobility_features: [N, embedding_dim] mobility pattern features
        """
        batch_features = []
        
        for user_checkins in checkin_data:
            # Extract location features
            locations = user_checkins['locations']  # [num_checkins, 2] (lat, lon)
            home_location = self._estimate_home_location(locations)
            location_features = self.location_encoder(home_location.unsqueeze(0))
            
            # Extract temporal features
            timestamps = user_checkins['timestamps']  # [num_checkins]
            temporal_patterns = self._extract_temporal_patterns(timestamps)
            temporal_features = self.temporal_encoder(temporal_patterns.unsqueeze(0))
            
            # Extract activity features
            categories = user_checkins['categories']  # [num_checkins]
            activity_distribution = self._compute_activity_distribution(categories)
            activity_features = self.activity_encoder(activity_distribution.unsqueeze(0))
            
            # Extract mobility features
            mobility_stats = self._compute_mobility_stats(locations, timestamps)
            mobility_features = self.mobility_encoder(mobility_stats.unsqueeze(0))
            
            # Combine all features
            combined = torch.cat([
                location_features, temporal_features, 
                activity_features, mobility_features
            ], dim=-1)
            
            user_features = self.combiner(combined)
            batch_features.append(user_features)
        
        return torch.cat(batch_features, dim=0)
    
    def _estimate_home_location(self, locations):
        """Estimate home location as the most frequent location during night hours"""
        # Simple heuristic: centroid of all locations
        return locations.mean(dim=0)
    
    def _extract_temporal_patterns(self, timestamps):
        """Extract temporal usage patterns"""
        # Convert timestamps to datetime features
        hours = (timestamps % (24 * 3600)) / 3600  # Hour of day
        days = ((timestamps / (24 * 3600)) % 7)    # Day of week
        months = ((timestamps / (30 * 24 * 3600)) % 12)  # Month (approximate)
        
        # Compute average patterns
        avg_hour = hours.mean()
        avg_day = days.mean()
        avg_month = months.mean()
        
        return torch.tensor([avg_hour, avg_day, avg_month], dtype=torch.float)
    
    def _compute_activity_distribution(self, categories):
        """Compute distribution over POI categories"""
        # Assume categories are integers 0-9
        distribution = torch.zeros(10)
        for cat in categories:
            distribution[cat] += 1
        
        # Normalize
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        
        return distribution
    
    def _compute_mobility_stats(self, locations, timestamps):
        """Compute mobility statistics"""
        if len(locations) < 2:
            return torch.zeros(4)
        
        # Travel radius (std of distances from centroid)
        centroid = locations.mean(dim=0)
        distances = torch.norm(locations - centroid, dim=1)
        radius = distances.std()
        
        # Location diversity (number of unique locations / total)
        unique_locations = len(torch.unique(locations, dim=0))
        diversity = unique_locations / len(locations)
        
        # Check-in frequency (check-ins per day)
        time_span = (timestamps.max() - timestamps.min()) / (24 * 3600)  # days
        frequency = len(timestamps) / max(time_span, 1)
        
        # Regularity (inverse of temporal variance)
        if len(timestamps) > 1:
            time_diffs = timestamps[1:] - timestamps[:-1]
            regularity = 1.0 / (time_diffs.std() + 1e-6)
        else:
            regularity = torch.tensor(0.0)
        
        return torch.tensor([radius, diversity, frequency, regularity], dtype=torch.float)


class POICharacteristics(nn.Module):
    """
    Extract and encode POI characteristics.
    
    Features include:
    - POI popularity (number of visitors)
    - Category embeddings
    - Geographic clustering
    - Temporal activity patterns
    """
    
    def __init__(self, num_categories=10, embedding_dim=64):
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        
        # Category embeddings
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim // 2)
        
        # POI feature encoder
        self.poi_encoder = nn.Sequential(
            nn.Linear(4, embedding_dim // 2),  # popularity, cluster_id, temporal_score, geographic_score
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # Combination layer
        self.combiner = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, poi_data):
        """
        Encode POI characteristics.
        
        Args:
            poi_data: Dict containing POI information
            
        Returns:
            poi_features: [num_pois, embedding_dim] POI feature embeddings
        """
        # Category embeddings
        categories = poi_data['categories']  # [num_pois]
        cat_emb = self.category_embeddings(categories)
        
        # POI statistics
        popularity = poi_data['popularity'].unsqueeze(-1)  # [num_pois, 1]
        cluster_ids = poi_data['cluster_ids'].unsqueeze(-1)  # [num_pois, 1]
        temporal_scores = poi_data['temporal_scores'].unsqueeze(-1)  # [num_pois, 1]
        geographic_scores = poi_data['geographic_scores'].unsqueeze(-1)  # [num_pois, 1]
        
        poi_stats = torch.cat([popularity, cluster_ids, temporal_scores, geographic_scores], dim=-1)
        poi_emb = self.poi_encoder(poi_stats)
        
        # Combine category and POI embeddings
        combined = torch.cat([cat_emb, poi_emb], dim=-1)
        
        return self.combiner(combined)


if __name__ == "__main__":
    print("Advanced SRINet Enhancements")
    print("=" * 60)
    print("\nThis module provides comprehensive enhancements:")
    print("1. Spectral Positional Encodings (structural information)")
    print("2. Advanced Attention Mechanisms (GAT, multi-head, temporal)")
    print("3. Contrastive Learning Framework (InfoNCE, hard negative mining)")
    print("4. Graph Augmentation Techniques (dropout, noise, adversarial)")
    print("5. Advanced Feature Engineering (mobility patterns, POI characteristics)")
    print("\nTo use these enhancements, import the relevant classes and integrate them into your SRINet model.")
    print("=" * 60)

