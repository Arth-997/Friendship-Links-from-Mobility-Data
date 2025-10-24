"""
Graph Builder Module for SRINet

This module handles the construction of multiplex user meeting graphs
from preprocessed check-in data.
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
import os


class GraphBuilder:
    """Build multiplex user meeting graphs from check-in data
    
    Constructs graphs where users are nodes and edges represent
    co-occurrence at POIs within time windows, separated by category.
    
    Args:
        config: Configuration object with graph building parameters
    """
    
    def __init__(self, config):
        self.config = config
        self.time_window = config.time_window_hours * 3600  # Convert to seconds
        
    def build_meeting_graphs(self, checkin_df):
        """Build meeting graphs for each POI category
        
        Args:
            checkin_df (pd.DataFrame): Processed check-in data
            
        Returns:
            dict: Meeting graphs per category with edge lists and statistics
        """
        print("Building multiplex user meeting graphs...")
        print(f"Time window: {self.config.time_window_hours} hours")
        
        categories = checkin_df['category'].unique()
        meeting_graphs = {}
        category_stats = {}
        
        for category in tqdm(categories, desc="Processing categories"):
            # Filter check-ins for this category
            cat_checkins = checkin_df[checkin_df['category'] == category].copy()
            
            if len(cat_checkins) < 10:  # Skip categories with too few check-ins
                continue
            
            # Build meeting events for this category
            meetings, stats = self._compute_meetings(cat_checkins, category)
            
            # Filter categories with insufficient meetings
            if len(meetings) >= self.config.min_meetings_per_category:
                meeting_graphs[category] = meetings
                category_stats[category] = stats
                print(f"  {category}: {len(meetings)} meeting events, {stats['unique_users']} users")
            else:
                print(f"  {category}: {len(meetings)} meetings (filtered out - too few)")
        
        print(f"Built graphs for {len(meeting_graphs)} categories")
        return meeting_graphs, category_stats
    
    def _compute_meetings(self, checkins, category):
        """Compute user meeting events within time window for a category
        
        Args:
            checkins (pd.DataFrame): Check-ins for specific category
            category (str): POI category name
            
        Returns:
            tuple: (meetings list, statistics dict)
        """
        meetings = []
        poi_meeting_counts = defaultdict(int)
        user_meeting_counts = defaultdict(int)
        
        # Group check-ins by POI
        for poi_idx, poi_group in checkins.groupby('poi_idx'):
            # Sort by timestamp for efficient sliding window
            poi_checkins = poi_group.sort_values('unix_timestamp')
            
            if len(poi_checkins) < 2:
                continue
            
            # Find meetings within time window using sliding window
            poi_meetings = 0
            checkin_list = poi_checkins.to_dict('records')
            
            for i, checkin1 in enumerate(checkin_list):
                for j in range(i + 1, len(checkin_list)):
                    checkin2 = checkin_list[j]
                    
                    time_diff = checkin2['unix_timestamp'] - checkin1['unix_timestamp']
                    
                    if time_diff > self.time_window:
                        break  # No more meetings possible for checkin1
                    
                    # Skip if same user
                    if checkin1['user_idx'] == checkin2['user_idx']:
                        continue
                    
                    # Record meeting
                    user1 = min(checkin1['user_idx'], checkin2['user_idx'])
                    user2 = max(checkin1['user_idx'], checkin2['user_idx'])
                    
                    meetings.append({
                        'user1': user1,
                        'user2': user2,
                        'poi_idx': poi_idx,
                        'time_diff': time_diff,
                        'timestamp1': checkin1['unix_timestamp'],
                        'timestamp2': checkin2['unix_timestamp']
                    })
                    
                    poi_meetings += 1
                    user_meeting_counts[user1] += 1
                    user_meeting_counts[user2] += 1
            
            poi_meeting_counts[poi_idx] = poi_meetings
        
        # Compute statistics
        unique_users = len(set([m['user1'] for m in meetings] + [m['user2'] for m in meetings]))
        
        stats = {
            'category': category,
            'total_meetings': len(meetings),
            'unique_users': unique_users,
            'unique_pois': len(poi_meeting_counts),
            'avg_meetings_per_poi': np.mean(list(poi_meeting_counts.values())) if poi_meeting_counts else 0,
            'avg_meetings_per_user': np.mean(list(user_meeting_counts.values())) if user_meeting_counts else 0,
            'max_poi_meetings': max(poi_meeting_counts.values()) if poi_meeting_counts else 0
        }
        
        return meetings, stats
    
    def create_adjacency_matrices(self, meeting_graphs, num_users):
        """Create sparse adjacency matrices from meeting events
        
        Args:
            meeting_graphs (dict): Meeting graphs per category
            num_users (int): Total number of users
            
        Returns:
            tuple: (adjacency_matrices dict, edge_data dict)
        """
        print("Creating adjacency matrices...")
        
        adjacency_matrices = {}
        edge_data = {}
        
        for category, meetings in meeting_graphs.items():
            # Count meetings between user pairs
            edge_weights = defaultdict(int)
            for meeting in meetings:
                pair = (meeting['user1'], meeting['user2'])
                edge_weights[pair] += 1
            
            if not edge_weights:
                continue
            
            # Create edge lists (make symmetric for undirected graph)
            edges = []
            weights = []
            for (u1, u2), count in edge_weights.items():
                edges.extend([(u1, u2), (u2, u1)])  # Both directions
                weights.extend([count, count])
            
            # Convert to PyTorch tensors
            edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
            edge_weights_tensor = torch.tensor(weights, dtype=torch.float)
            
            # Store adjacency matrix data
            adjacency_matrices[category] = {
                'edge_index': edge_index,
                'edge_weights': edge_weights_tensor,
                'num_edges': len(edges) // 2,  # Undirected, so count unique edges
                'num_nodes': num_users,
                'density': (len(edges) // 2) / (num_users * (num_users - 1) / 2) if num_users > 1 else 0.0
            }
            
            # Store edge data for analysis
            edge_data[category] = {
                'edges': [(u1, u2) for (u1, u2), _ in edge_weights.items()],
                'weights': [count for _, count in edge_weights.items()],
                'meeting_events': meetings
            }
            
            print(f"  {category}: {len(edges)//2} unique edges, density: {adjacency_matrices[category]['density']:.6f}")
        
        return adjacency_matrices, edge_data
    
    def analyze_graph_properties(self, adjacency_matrices):
        """Analyze properties of the constructed graphs
        
        Args:
            adjacency_matrices (dict): Adjacency matrices per category
            
        Returns:
            dict: Graph analysis results
        """
        print("Analyzing graph properties...")
        
        analysis = {}
        
        for category, data in adjacency_matrices.items():
            edge_index = data['edge_index']
            edge_weights = data['edge_weights']
            num_nodes = data['num_nodes']
            
            # Basic statistics
            num_edges = data['num_edges']
            density = data['density']
            
            # Degree statistics
            degrees = torch.zeros(num_nodes, dtype=torch.float)
            degrees.scatter_add_(0, edge_index[0], edge_weights)
            degrees = degrees[degrees > 0]  # Only consider connected nodes
            
            # Weight statistics
            unique_weights = edge_weights[::2]  # Undirected, so take every other edge
            
            analysis[category] = {
                'num_edges': num_edges,
                'num_nodes': num_nodes,
                'connected_nodes': len(degrees),
                'density': density,
                'avg_degree': degrees.mean().item() if len(degrees) > 0 else 0.0,
                'max_degree': degrees.max().item() if len(degrees) > 0 else 0.0,
                'degree_std': degrees.std().item() if len(degrees) > 0 else 0.0,
                'avg_edge_weight': unique_weights.mean().item(),
                'max_edge_weight': unique_weights.max().item(),
                'weight_std': unique_weights.std().item()
            }
        
        # Print summary
        for category, stats in analysis.items():
            print(f"\n{category}:")
            print(f"  Edges: {stats['num_edges']:,}")
            print(f"  Connected nodes: {stats['connected_nodes']:,} / {stats['num_nodes']:,}")
            print(f"  Density: {stats['density']:.6f}")
            print(f"  Avg degree: {stats['avg_degree']:.2f}")
            print(f"  Avg edge weight: {stats['avg_edge_weight']:.2f}")
        
        return analysis
    
    def save_graphs(self, adjacency_matrices, edge_data, category_stats, save_dir):
        """Save graph data to files
        
        Args:
            adjacency_matrices (dict): Adjacency matrices per category
            edge_data (dict): Edge data per category
            category_stats (dict): Category statistics
            save_dir (str): Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save adjacency matrices as PyTorch tensors
        torch.save(adjacency_matrices, f'{save_dir}/adjacency_matrices.pt')
        
        # Save edge data for inspection
        with open(f'{save_dir}/edge_data.pkl', 'wb') as f:
            pickle.dump(edge_data, f)
        
        # Save category statistics
        with open(f'{save_dir}/category_stats.pkl', 'wb') as f:
            pickle.dump(category_stats, f)
        
        # Save edge lists as CSV for each category (for inspection)
        for category, data in edge_data.items():
            edge_df = pd.DataFrame({
                'user1': [edge[0] for edge in data['edges']],
                'user2': [edge[1] for edge in data['edges']],
                'weight': data['weights']
            })
            edge_df.to_csv(f'{save_dir}/edges_{category.lower().replace(" ", "_")}.csv', index=False)
        
        # Save graph summary
        graph_summary = {
            'num_categories': len(adjacency_matrices),
            'categories': list(adjacency_matrices.keys()),
            'total_edges': sum(data['num_edges'] for data in adjacency_matrices.values()),
            'time_window_hours': self.config.time_window_hours,
            'min_meetings_per_category': self.config.min_meetings_per_category,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(f'{save_dir}/graph_summary.json', 'w') as f:
            json.dump(graph_summary, f, indent=2)
        
        print(f"✓ Saved graph data to {save_dir}")
    
    def load_graphs(self, save_dir):
        """Load previously saved graph data
        
        Args:
            save_dir (str): Directory containing saved graph data
            
        Returns:
            tuple: (adjacency_matrices, edge_data, category_stats)
        """
        print(f"Loading graph data from {save_dir}...")
        
        # Load adjacency matrices
        adjacency_matrices = torch.load(f'{save_dir}/adjacency_matrices.pt')
        
        # Load edge data
        with open(f'{save_dir}/edge_data.pkl', 'rb') as f:
            edge_data = pickle.load(f)
        
        # Load category statistics
        with open(f'{save_dir}/category_stats.pkl', 'rb') as f:
            category_stats = pickle.load(f)
        
        print(f"✓ Loaded graphs for {len(adjacency_matrices)} categories")
        return adjacency_matrices, edge_data, category_stats
    
    def filter_graphs(self, adjacency_matrices, min_edges=None, max_edges=None, 
                     min_density=None, max_density=None):
        """Filter graphs based on properties
        
        Args:
            adjacency_matrices (dict): Input adjacency matrices
            min_edges (int): Minimum number of edges
            max_edges (int): Maximum number of edges  
            min_density (float): Minimum graph density
            max_density (float): Maximum graph density
            
        Returns:
            dict: Filtered adjacency matrices
        """
        print("Filtering graphs...")
        
        filtered = {}
        
        for category, data in adjacency_matrices.items():
            keep = True
            
            if min_edges is not None and data['num_edges'] < min_edges:
                keep = False
            if max_edges is not None and data['num_edges'] > max_edges:
                keep = False
            if min_density is not None and data['density'] < min_density:
                keep = False
            if max_density is not None and data['density'] > max_density:
                keep = False
            
            if keep:
                filtered[category] = data
            else:
                print(f"  Filtered out {category}: {data['num_edges']} edges, density {data['density']:.6f}")
        
        print(f"Kept {len(filtered)}/{len(adjacency_matrices)} categories after filtering")
        return filtered