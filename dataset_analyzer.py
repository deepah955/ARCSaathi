"""
Advanced Dataset Analyzer for ML Algorithm Recommendation
Analyzes dataset characteristics to inform algorithm selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class DatasetAnalyzer:
    """Comprehensive dataset analysis for ML algorithm recommendation"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if applicable)
        
        Returns:
            Dictionary containing all analysis results
        """
        self.analysis_results = {
            'basic_stats': self._get_basic_stats(df),
            'data_quality': self._analyze_data_quality(df),
            'feature_analysis': self._analyze_features(df, target_column),
            'target_analysis': self._analyze_target(df, target_column) if target_column else None,
            'complexity_metrics': self._calculate_complexity_metrics(df, target_column),
            'recommendations': self._generate_preliminary_recommendations(df, target_column)
        }
        
        return self.analysis_results
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic dataset statistics"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            'boolean_columns': len(df.select_dtypes(include=['bool']).columns)
        }
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        return {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': (missing_data > 0).sum(),
            'high_missing_columns': missing_percent[missing_percent > 50].to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    
    def _analyze_features(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Analyze feature characteristics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        feature_analysis = {
            'numeric_features': {},
            'categorical_features': {},
            'feature_correlation': None,
            'high_correlation_pairs': []
        }
        
        # Analyze numeric features
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            feature_analysis['numeric_features'] = {
                'count': len(numeric_cols),
                'mean_skewness': numeric_df.skew().abs().mean(),
                'high_skewness_features': numeric_df.skew().abs()[numeric_df.skew().abs() > 2].to_dict(),
                'mean_kurtosis': numeric_df.kurtosis().abs().mean(),
                'variance_ratio': (numeric_df.var() / numeric_df.mean().abs()).mean() if numeric_df.mean().abs().sum() > 0 else 0
            }
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = numeric_df.corr().abs()
                feature_analysis['feature_correlation'] = corr_matrix.mean().mean()
                
                # Find highly correlated pairs
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.8:
                            high_corr.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_matrix.iloc[i, j]
                            })
                feature_analysis['high_correlation_pairs'] = high_corr
        
        # Analyze categorical features
        if len(categorical_cols) > 0:
            categorical_df = df[categorical_cols]
            cardinality = {}
            for col in categorical_cols:
                cardinality[col] = df[col].nunique()
            
            feature_analysis['categorical_features'] = {
                'count': len(categorical_cols),
                'mean_cardinality': np.mean(list(cardinality.values())),
                'high_cardinality_features': {k: v for k, v in cardinality.items() if v > 50},
                'low_cardinality_features': {k: v for k, v in cardinality.items() if v <= 5}
            }
        
        return feature_analysis
    
    def _analyze_target(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable characteristics"""
        if target_column not in df.columns:
            return None
        
        target = df[target_column]
        target_analysis = {
            'type': None,
            'distribution': {},
            'class_imbalance': None,
            'regression_metrics': {}
        }
        
        # Determine if classification or regression
        if target.dtype in ['object', 'category', 'bool'] or target.nunique() < 20:
            target_analysis['type'] = 'classification'
            target_analysis['n_classes'] = target.nunique()
            value_counts = target.value_counts()
            target_analysis['distribution'] = value_counts.to_dict()
            
            # Calculate class imbalance
            if len(value_counts) > 1:
                max_class = value_counts.max()
                min_class = value_counts.min()
                imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                target_analysis['class_imbalance'] = {
                    'ratio': imbalance_ratio,
                    'is_imbalanced': imbalance_ratio > 5,
                    'majority_class_pct': (max_class / len(target)) * 100
                }
        else:
            target_analysis['type'] = 'regression'
            target_analysis['regression_metrics'] = {
                'mean': target.mean(),
                'std': target.std(),
                'skewness': target.skew(),
                'kurtosis': target.kurtosis(),
                'range': target.max() - target.min(),
                'coefficient_of_variation': target.std() / target.mean() if target.mean() != 0 else 0
            }
        
        return target_analysis
    
    def _calculate_complexity_metrics(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Calculate dataset complexity metrics"""
        n_samples = len(df)
        n_features = len(df.columns) - (1 if target_column else 0)
        
        # Feature-to-sample ratio
        feature_sample_ratio = n_features / n_samples if n_samples > 0 else 0
        
        # Dimensionality
        if n_samples > 0 and n_features > 0:
            dimensionality = 'high' if feature_sample_ratio > 0.1 else 'medium' if feature_sample_ratio > 0.01 else 'low'
        else:
            dimensionality = 'unknown'
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_sample_ratio': feature_sample_ratio,
            'dimensionality': dimensionality,
            'dataset_size': 'small' if n_samples < 1000 else 'medium' if n_samples < 100000 else 'large',
            'complexity_score': self._calculate_complexity_score(df, target_column)
        }
    
    def _calculate_complexity_score(self, df: pd.DataFrame, target_column: str = None) -> float:
        """Calculate overall dataset complexity score (0-1)"""
        score = 0.0
        
        # Size complexity
        n_samples = len(df)
        if n_samples > 100000:
            score += 0.2
        elif n_samples > 10000:
            score += 0.1
        
        # Feature complexity
        n_features = len(df.columns) - (1 if target_column else 0)
        if n_features > 100:
            score += 0.2
        elif n_features > 50:
            score += 0.1
        
        # Missing data complexity
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            score += 0.2
        elif missing_pct > 5:
            score += 0.1
        
        # Feature type complexity
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        if categorical_cols > numeric_cols:
            score += 0.2
        elif categorical_cols > 0:
            score += 0.1
        
        # Correlation complexity
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_mean = numeric_df.corr().abs().mean().mean()
            if corr_mean > 0.7:
                score += 0.2
            elif corr_mean > 0.5:
                score += 0.1
        
        return min(score, 1.0)
    
    def _generate_preliminary_recommendations(self, df: pd.DataFrame, target_column: str = None) -> List[str]:
        """Generate preliminary recommendations based on analysis"""
        recommendations = []
        
        # Missing data recommendations
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            recommendations.append("High missing data detected - consider imputation strategies")
        elif missing_pct > 5:
            recommendations.append("Moderate missing data - imputation may be beneficial")
        
        # Feature engineering recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            high_skew = numeric_df.skew().abs()[numeric_df.skew().abs() > 2]
            if len(high_skew) > 0:
                recommendations.append(f"{len(high_skew)} features with high skewness - consider transformation")
        
        # Correlation recommendations
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = (corr_matrix > 0.8).sum().sum() - len(corr_matrix)
            if high_corr_pairs > 0:
                recommendations.append("High feature correlation detected - consider feature selection")
        
        # Categorical encoding recommendations
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
            if len(high_cardinality) > 0:
                recommendations.append(f"{len(high_cardinality)} high cardinality categorical features - consider encoding strategies")
        
        return recommendations

