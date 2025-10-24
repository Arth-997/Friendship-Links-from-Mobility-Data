"""
Data Processing Module for SRINet

This module handles the ingestion and preprocessing of check-in data
into the format required for SRINet training and evaluation.
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Data preprocessing pipeline for SRINet
    
    Handles loading, cleaning, and preprocessing of check-in data from
    location-based social networks.
    
    Args:
        config: Configuration object with data processing parameters
    """
    
    def __init__(self, config):
        self.config = config
        
    def load_checkin_data(self, filepath):
        """Load check-in data from file
        
        Expected format: CSV with columns [user_id, poi_id, timestamp, category]
        or similar structure that can be mapped.
        
        Args:
            filepath (str): Path to check-in data file
            
        Returns:
            pd.DataFrame: Loaded check-in data
        """
        print(f"Loading check-in data from {filepath}...")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.pkl'):
            df = pd.read_pickle(filepath)
        else:
            raise ValueError("Supported formats: .csv, .pkl")
        
        print(f"Loaded {len(df)} check-ins")
        return df
    
    def create_synthetic_data(self, num_users=1000, num_pois=500, num_checkins=10000, 
                            save_path=None):
        """Create synthetic check-in data for testing
        
        Generates realistic synthetic check-in patterns with temporal
        and spatial correlations.
        
        Args:
            num_users (int): Number of users to generate
            num_pois (int): Number of POIs to generate  
            num_checkins (int): Number of check-ins to generate
            save_path (str): Optional path to save generated data
            
        Returns:
            tuple: (checkin_df, poi_df) DataFrames
        """
        print("Creating synthetic check-in data...")
        
        # Define POI categories with realistic distributions
        categories = [
            'Restaurant', 'Shopping', 'Entertainment', 'Transport', 'Education',
            'Healthcare', 'Sports', 'Office', 'Home', 'Other'
        ]
        
        category_weights = [0.25, 0.15, 0.12, 0.1, 0.08, 0.05, 0.08, 0.1, 0.05, 0.02]
        
        # Generate POIs with categories and locations
        pois = []
        for i in range(num_pois):
            category = np.random.choice(categories, p=category_weights)
            # Simulate city coordinates (e.g., around NYC)
            lat = 40.7 + np.random.normal(0, 0.1)
            lon = -74.0 + np.random.normal(0, 0.1)
            
            pois.append({
                'poi_id': f'poi_{i}',
                'category': category,
                'lat': lat,
                'lon': lon
            })
        
        poi_df = pd.DataFrame(pois)
        
        # Generate check-ins with temporal patterns
        checkins = []
        base_time = datetime(2024, 1, 1)
        
        # Create user preferences (some users prefer certain categories)
        user_preferences = {}
        for user_id in range(num_users):
            # Each user has preferences for 2-3 categories
            preferred_cats = np.random.choice(categories, size=np.random.randint(2, 4), replace=False)
            user_preferences[user_id] = preferred_cats
        
        for _ in range(num_checkins):
            user_id = np.random.randint(0, num_users)
            
            # Bias POI selection towards user preferences
            if user_id in user_preferences:
                preferred_pois = poi_df[poi_df['category'].isin(user_preferences[user_id])]
                if len(preferred_pois) > 0 and np.random.random() < 0.7:
                    poi_id = np.random.choice(preferred_pois['poi_id'])
                else:
                    poi_id = np.random.choice(poi_df['poi_id'])
            else:
                poi_id = np.random.choice(poi_df['poi_id'])
            
            # Generate realistic timestamps (weekday/weekend patterns)
            day_offset = np.random.randint(0, 30)  # 30 days of data
            hour = np.random.choice(range(24), p=self._get_hourly_weights())
            minute = np.random.randint(0, 60)
            
            timestamp = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
            
            checkins.append({
                'user_id': user_id,
                'poi_id': poi_id,
                'timestamp': timestamp,
                'unix_timestamp': timestamp.timestamp()
            })
        
        checkin_df = pd.DataFrame(checkins)
        
        # Merge with POI categories
        checkin_df = checkin_df.merge(poi_df[['poi_id', 'category']], on='poi_id')
        
        print(f"Generated {len(checkin_df)} check-ins for {num_users} users at {num_pois} POIs")
        print(f"Categories: {checkin_df['category'].value_counts().to_dict()}")
        
        if save_path:
            checkin_df.to_csv(f"{save_path}/synthetic_checkins.csv", index=False)
            poi_df.to_csv(f"{save_path}/synthetic_pois.csv", index=False)
            print(f"✓ Saved synthetic data to {save_path}")
        
        return checkin_df, poi_df
    
    def _get_hourly_weights(self):
        """Get realistic hourly check-in probability weights"""
        # Higher probability during daytime hours
        weights = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM
            0.03, 0.05, 0.07, 0.08, 0.09, 0.10,  # 6-11 AM  
            0.12, 0.11, 0.10, 0.09, 0.08, 0.07,  # 12-5 PM
            0.09, 0.10, 0.08, 0.06, 0.04, 0.02   # 6-11 PM
        ])
        return weights / weights.sum()
    
    def preprocess_checkins(self, checkin_df):
        """Preprocess check-in data
        
        Performs cleaning, filtering, and index mapping.
        
        Args:
            checkin_df (pd.DataFrame): Raw check-in data
            
        Returns:
            tuple: (processed_checkin_df, user_to_idx, poi_to_idx)
        """
        print("Preprocessing check-in data...")
        
        # Ensure required columns exist
        required_cols = ['user_id', 'poi_id', 'timestamp', 'category']
        missing_cols = set(required_cols) - set(checkin_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to unix timestamp if needed
        if 'unix_timestamp' not in checkin_df.columns:
            if pd.api.types.is_datetime64_any_dtype(checkin_df['timestamp']):
                checkin_df['unix_timestamp'] = checkin_df['timestamp'].astype(int) // 10**9
            else:
                checkin_df['unix_timestamp'] = pd.to_datetime(checkin_df['timestamp']).astype(int) // 10**9
        
        # Sort by user and time
        checkin_df = checkin_df.sort_values(['user_id', 'unix_timestamp']).copy()
        
        # Filter users with minimum check-ins
        user_counts = checkin_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_checkins_per_user].index
        checkin_df = checkin_df[checkin_df['user_id'].isin(valid_users)]
        print(f"Filtered to {len(valid_users)} users with ≥{self.config.min_checkins_per_user} check-ins")
        
        # Remove duplicate check-ins (same user, POI, within short time window)
        checkin_df = self._remove_duplicates(checkin_df)
        
        # Create user and POI mappings to integer indices
        unique_users = sorted(checkin_df['user_id'].unique())
        unique_pois = sorted(checkin_df['poi_id'].unique())
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        poi_to_idx = {poi: idx for idx, poi in enumerate(unique_pois)}
        
        # Map to integer indices
        checkin_df['user_idx'] = checkin_df['user_id'].map(user_to_idx)
        checkin_df['poi_idx'] = checkin_df['poi_id'].map(poi_to_idx)
        
        # Validate mappings
        assert checkin_df['user_idx'].isna().sum() == 0, "Failed to map some users"
        assert checkin_df['poi_idx'].isna().sum() == 0, "Failed to map some POIs"
        
        print(f"Processed data:")
        print(f"  Users: {len(unique_users)}")
        print(f"  POIs: {len(unique_pois)}")
        print(f"  Check-ins: {len(checkin_df)}")
        print(f"  Categories: {len(checkin_df['category'].unique())}")
        print(f"  Time range: {checkin_df['timestamp'].min()} to {checkin_df['timestamp'].max()}")
        
        return checkin_df, user_to_idx, poi_to_idx
    
    def _remove_duplicates(self, checkin_df, time_threshold_minutes=5):
        """Remove duplicate check-ins within short time windows"""
        print("Removing duplicate check-ins...")
        
        initial_count = len(checkin_df)
        
        # Sort by user, POI, timestamp
        checkin_df = checkin_df.sort_values(['user_id', 'poi_id', 'unix_timestamp'])
        
        # Mark duplicates: same user-POI within time threshold
        checkin_df['time_diff'] = checkin_df.groupby(['user_id', 'poi_id'])['unix_timestamp'].diff()
        checkin_df['is_duplicate'] = (checkin_df['time_diff'] < time_threshold_minutes * 60) & (checkin_df['time_diff'].notna())
        
        # Keep only non-duplicates
        checkin_df = checkin_df[~checkin_df['is_duplicate']].copy()
        checkin_df = checkin_df.drop(['time_diff', 'is_duplicate'], axis=1)
        
        final_count = len(checkin_df)
        print(f"Removed {initial_count - final_count} duplicate check-ins")
        
        return checkin_df
    
    def create_train_test_split(self, checkin_df, test_ratio=0.2, temporal_split=True):
        """Create train/test split of check-in data
        
        Args:
            checkin_df (pd.DataFrame): Processed check-in data
            test_ratio (float): Fraction of data for testing
            temporal_split (bool): If True, split by time; if False, split randomly
            
        Returns:
            tuple: (train_df, test_df)
        """
        print(f"Creating train/test split (test_ratio={test_ratio}, temporal={temporal_split})")
        
        if temporal_split:
            # Split by timestamp (earlier data for training)
            split_time = checkin_df['unix_timestamp'].quantile(1 - test_ratio)
            train_df = checkin_df[checkin_df['unix_timestamp'] < split_time].copy()
            test_df = checkin_df[checkin_df['unix_timestamp'] >= split_time].copy()
        else:
            # Random split stratified by user
            train_df, test_df = train_test_split(
                checkin_df, test_size=test_ratio, stratify=checkin_df['user_idx'], random_state=42
            )
        
        print(f"Split: {len(train_df)} train, {len(test_df)} test check-ins")
        return train_df, test_df
    
    def save_processed_data(self, checkin_df, user_to_idx, poi_to_idx, save_dir):
        """Save processed data to files
        
        Args:
            checkin_df (pd.DataFrame): Processed check-in data
            user_to_idx (dict): User ID to index mapping
            poi_to_idx (dict): POI ID to index mapping  
            save_dir (str): Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save mappings
        with open(f'{save_dir}/user_mapping.pkl', 'wb') as f:
            pickle.dump(user_to_idx, f)
        
        with open(f'{save_dir}/poi_mapping.pkl', 'wb') as f:
            pickle.dump(poi_to_idx, f)
        
        # Save processed check-ins
        checkin_df.to_csv(f'{save_dir}/checkins.csv', index=False)
        
        # Save summary statistics
        stats = {
            'num_users': len(user_to_idx),
            'num_pois': len(poi_to_idx),
            'num_checkins': len(checkin_df),
            'categories': checkin_df['category'].value_counts().to_dict(),
            'time_range': {
                'start': checkin_df['timestamp'].min().isoformat() if hasattr(checkin_df['timestamp'].min(), 'isoformat') else str(checkin_df['timestamp'].min()),
                'end': checkin_df['timestamp'].max().isoformat() if hasattr(checkin_df['timestamp'].max(), 'isoformat') else str(checkin_df['timestamp'].max())
            },
            'processed_at': datetime.now().isoformat()
        }
        
        import json
        with open(f'{save_dir}/data_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved processed data to {save_dir}")
    
    def load_processed_data(self, save_dir):
        """Load previously processed data
        
        Args:
            save_dir (str): Directory containing processed data
            
        Returns:
            tuple: (checkin_df, user_to_idx, poi_to_idx)
        """
        print(f"Loading processed data from {save_dir}...")
        
        # Load check-ins
        checkin_df = pd.read_csv(f'{save_dir}/checkins.csv')
        
        # Load mappings
        with open(f'{save_dir}/user_mapping.pkl', 'rb') as f:
            user_to_idx = pickle.load(f)
        
        with open(f'{save_dir}/poi_mapping.pkl', 'rb') as f:
            poi_to_idx = pickle.load(f)
        
        print(f"✓ Loaded {len(checkin_df)} check-ins for {len(user_to_idx)} users")
        return checkin_df, user_to_idx, poi_to_idx