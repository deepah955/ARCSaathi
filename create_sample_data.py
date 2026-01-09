"""
Utility script to create sample datasets for testing the ML Recommender
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def create_classification_dataset(n_samples=1000, n_features=20, n_classes=2, output_file='sample_classification.csv'):
    """Create a sample classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    df['category'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
    
    # Add target
    df['target'] = y
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, np.random.choice(feature_names[:5], size=1)[0]] = np.nan
    
    df.to_csv(output_file, index=False)
    print(f"Created classification dataset: {output_file}")
    print(f"  Samples: {n_samples}, Features: {n_features + 2}, Classes: {n_classes}")
    return df

def create_regression_dataset(n_samples=1000, n_features=15, output_file='sample_regression.csv'):
    """Create a sample regression dataset"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        noise=10.0,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    df['category'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    
    # Add target
    df['target'] = y
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, np.random.choice(feature_names[:3], size=1)[0]] = np.nan
    
    df.to_csv(output_file, index=False)
    print(f"Created regression dataset: {output_file}")
    print(f"  Samples: {n_samples}, Features: {n_features + 1}")
    return df

def create_imbalanced_dataset(n_samples=1000, output_file='sample_imbalanced.csv'):
    """Create an imbalanced classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    feature_names = [f'feature_{i+1}' for i in range(12)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    df.to_csv(output_file, index=False)
    print(f"Created imbalanced dataset: {output_file}")
    print(f"  Samples: {n_samples}, Class distribution: {df['target'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    print("Creating sample datasets for testing...\n")
    
    # Create various sample datasets
    create_classification_dataset(n_samples=2000, n_features=25, n_classes=3, 
                                 output_file='sample_multiclass.csv')
    create_classification_dataset(n_samples=500, n_features=10, n_classes=2,
                                 output_file='sample_small_classification.csv')
    create_regression_dataset(n_samples=1500, n_features=20,
                             output_file='sample_regression.csv')
    create_imbalanced_dataset(n_samples=1000, output_file='sample_imbalanced.csv')
    
    print("\n✅ Sample datasets created successfully!")
    print("You can now load these in the ML Recommender application.")

