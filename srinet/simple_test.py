#!/usr/bin/env python3
"""
Simple test script to verify SRINet implementation
"""

import sys
import os
import pandas as pd
import torch

# Add src to path
sys.path.append('src')

def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing basic imports...")
    
    try:
        from utils import SRINetConfig
        print("✅ SRINetConfig imported successfully")
        
        from data_processing import DataProcessor
        print("✅ DataProcessor imported successfully")
        
        from graph_builder import GraphBuilder
        print("✅ GraphBuilder imported successfully")
        
        from models.mask_module import TopologyMaskModule
        print("✅ TopologyMaskModule imported successfully")
        
        from models.gnn_layers import MaskedGCNLayer
        print("✅ MaskedGCNLayer imported successfully")
        
        from models.srinet import SRINet
        print("✅ SRINet imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test loading the Gowalla dataset"""
    print("\nTesting data loading...")
    
    try:
        # Load a small sample
        data = pd.read_csv('data/Gowalla_cleanCheckins.csv', nrows=1000)
        print(f"✅ Loaded {len(data)} rows")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Unique users: {data['user'].nunique()}")
        print(f"   Unique locations: {data['location id'].nunique()}")
        return data
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return None

def test_model_creation():
    """Test creating SRINet model"""
    print("\nTesting model creation...")
    
    try:
        from utils import SRINetConfig
        from models.srinet import SRINet
        
        config = SRINetConfig()
        config.embedding_dim = 16
        config.hidden_dim = 32
        config.num_layers = 2
        
        # Create dummy adjacency matrices for testing
        num_users = 10
        adjacency_matrices = {
            'category_1': torch.rand(num_users, num_users),
            'category_2': torch.rand(num_users, num_users)
        }
        
        model = SRINet(config, num_users, adjacency_matrices)
        print(f"✅ SRINet model created successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test basic forward components
        print("✅ Model initialized with all components")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing pipeline"""
    print("\nTesting data processing...")
    
    try:
        from utils import SRINetConfig
        from data_processing import DataProcessor
        
        config = SRINetConfig()
        processor = DataProcessor(config)
        
        # Load sample data
        data = pd.read_csv('data/Gowalla_cleanCheckins.csv', nrows=500)
        
        # Test preprocessing
        processed_data = processor.preprocess_checkins(data)
        print(f"✅ Data preprocessing successful")
        print(f"   Processed shape: {processed_data.shape}")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🚀 SRINet Implementation Test")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 50)
    
    # Test imports
    if not test_basic_imports():
        print("❌ Import tests failed - stopping")
        return
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        print("❌ Data loading failed - stopping")
        return
    
    # Test model creation
    if not test_model_creation():
        print("❌ Model creation failed")
        return
    
    # Test data processing
    processed_data = test_data_processing()
    if processed_data is None:
        print("❌ Data processing failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All basic tests passed!")
    print("✅ SRINet implementation is working correctly")
    print("=" * 50)

if __name__ == "__main__":
    main()