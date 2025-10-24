#!/usr/bin/env python3
"""
Test script to verify SRINet implementation with Gowalla dataset
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
impo        # Load and process data
        print("1. Loading and processing data...")
        config = SRINetConfig()
        processor = DataProcessor(config)torch.nn as nn
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test data loading and basic preprocessing"""
    print("=" * 50)
    print("Testing Data Loading...")
    print("=" * 50)
    
    try:
        # Load a sample of the data for testing
        print("Loading sample data...")
        data = pd.read_csv('data/Gowalla_cleanCheckins.csv', nrows=10000)
        print(f"Loaded {len(data)} rows")
        print(f"Columns: {list(data.columns)}")
        print(f"Data shape: {data.shape}")
        
        # Check for missing values
        print(f"Missing values: {data.isnull().sum().sum()}")
        
        # Basic statistics
        print("\nBasic statistics:")
        print(f"Unique users: {data['user'].nunique()}")
        print(f"Unique locations: {data['location id'].nunique()}")
        print(f"Latitude range: {data['latitude'].min():.4f} to {data['latitude'].max():.4f}")
        print(f"Longitude range: {data['longitude'].min():.4f} to {data['longitude'].max():.4f}")
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_data_processing():
    """Test data processing pipeline"""
    print("\n" + "=" * 50)
    print("Testing Data Processing...")
    print("=" * 50)
    
    try:
        from data_processing import DataProcessor
        from utils import SRINetConfig
        
        config = SRINetConfig()
        processor = DataProcessor(config)
        print("DataProcessor initialized successfully")
        
        # Load sample data
        data = pd.read_csv('data/Gowalla_cleanCheckins.csv', nrows=1000)
        
        # Test preprocessing
        processed_data = processor.preprocess(data)
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed columns: {list(processed_data.columns)}")
        
        return processed_data
        
    except Exception as e:
        print(f"Error in data processing: {e}")
        return None

def test_graph_builder():
    """Test graph construction"""
    print("\n" + "=" * 50)
    print("Testing Graph Builder...")
    print("=" * 50)
    
    try:
        from graph_builder import MultiplexGraphBuilder
        from data_processing import DataProcessor
        from utils import SRINetConfig
        
        # Process sample data
        config = SRINetConfig()
        processor = DataProcessor(config)
        data = pd.read_csv('data/Gowalla_cleanCheckins.csv', nrows=500)
        processed_data = processor.preprocess(data)
        
        # Build graph
        builder = MultiplexGraphBuilder(k_neighbors=5)
        graph = builder.build_graph(processed_data)
        
        print(f"Graph nodes: {graph.num_nodes}")
        print(f"Graph edges: {graph.num_edges}")
        print(f"Node features shape: {graph.x.shape}")
        
        return graph
        
    except Exception as e:
        print(f"Error in graph building: {e}")
        return None

def test_model_components():
    """Test individual model components"""
    print("\n" + "=" * 50)
    print("Testing Model Components...")
    print("=" * 50)
    
    try:
        from models.mask_module import BinaryConcreteMask
        from models.gnn_layers import MaskedGCNLayer, MaskedGATLayer
        from models.srinet import SRINet
        from utils import SRINetConfig
        
        # Test Binary Concrete Mask
        print("Testing Binary Concrete Mask...")
        mask = BinaryConcreteMask(num_edges=100, temperature=1.0)
        edges = torch.randn(100, 2)
        mask_weights = mask(edges)
        print(f"Mask weights shape: {mask_weights.shape}")
        print(f"Mask weights range: {mask_weights.min():.4f} to {mask_weights.max():.4f}")
        
        # Test Masked GCN Layer
        print("\nTesting Masked GCN Layer...")
        gcn = MaskedGCNLayer(16, 32)
        x = torch.randn(50, 16)
        edge_index = torch.randint(0, 50, (2, 100))
        edge_weights = torch.rand(100)
        out = gcn(x, edge_index, edge_weights)
        print(f"GCN output shape: {out.shape}")
        
        # Test SRINet model
        print("\nTesting SRINet model...")
        config = SRINetConfig(
            input_dim=16,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            dropout=0.1
        )
        model = SRINet(config)
        
        # Create dummy graph data
        batch_size = 10
        num_nodes = 50
        num_edges = 100
        
        x = torch.randn(batch_size, num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
        
        output = model(x, edge_index)
        print(f"Model output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error in model testing: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n" + "=" * 50)
    print("Testing Full Pipeline...")
    print("=" * 50)
    
    try:
        from data_processing import DataProcessor
        from graph_builder import MultiplexGraphBuilder
        from models.srinet import SRINet
        from utils import SRINetConfig
        
        # Load and process data
        print("1. Loading and processing data...")
        config = SRINetConfig()
        processor = DataProcessor(config)
        data = pd.read_csv('data/Gowalla_cleanCheckins.csv', nrows=200)
        processed_data = processor.preprocess(data)
        
        # Build graph
        print("2. Building graph...")
        builder = MultiplexGraphBuilder(k_neighbors=3)
        graph = builder.build_graph(processed_data)
        
        # Create model
        print("3. Creating model...")
        config = SRINetConfig(
            input_dim=graph.x.shape[1],
            hidden_dim=64,
            output_dim=2,
            num_layers=2,
            dropout=0.1
        )
        model = SRINet(config)
        
        # Forward pass
        print("4. Running forward pass...")
        with torch.no_grad():
            # Reshape for batch processing
            x = graph.x.unsqueeze(0)  # Add batch dimension
            edge_index = graph.edge_index.unsqueeze(0)  # Add batch dimension
            
            output = model(x, edge_index)
            print(f"Output shape: {output.shape}")
            
        print("✅ Full pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in full pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting SRINet Implementation Tests")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        print("❌ Data loading failed")
        return
    
    # Test data processing
    processed_data = test_data_processing()
    if processed_data is None:
        print("❌ Data processing failed")
        return
    
    # Test graph building
    graph = test_graph_builder()
    if graph is None:
        print("❌ Graph building failed")
        return
    
    # Test model components
    if not test_model_components():
        print("❌ Model component testing failed")
        return
    
    # Test full pipeline
    if not test_full_pipeline():
        print("❌ Full pipeline test failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! SRINet implementation is working!")
    print("=" * 50)

if __name__ == "__main__":
    main()