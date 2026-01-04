"""
Intelligent ML Algorithm Recommendation Engine
Uses advanced heuristics and ML-based scoring to recommend optimal algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNSUPERVISED = "unsupervised"


@dataclass
class AlgorithmRecommendation:
    """Data class for algorithm recommendations"""
    name: str
    category: str
    score: float
    confidence: float
    reasoning: List[str]
    pros: List[str]
    cons: List[str]
    best_for: List[str]
    hyperparameters_suggestions: Dict[str, Any]


class AlgorithmRecommender:
    """Intelligent ML algorithm recommendation engine"""
    
    def __init__(self):
        self.algorithms = self._initialize_algorithms()
    
    def recommend(self, analysis_results: Dict[str, Any], task_type: str = None) -> List[AlgorithmRecommendation]:
        """
        Generate algorithm recommendations based on dataset analysis
        
        Args:
            analysis_results: Results from DatasetAnalyzer
            task_type: 'classification', 'regression', or 'clustering'
        
        Returns:
            List of AlgorithmRecommendation objects sorted by score
        """
        # Determine task type if not provided
        if not task_type:
            task_type = self._infer_task_type(analysis_results)
        
        # Get relevant algorithms for task type
        relevant_algorithms = self._get_relevant_algorithms(task_type)
        
        # Score each algorithm
        recommendations = []
        for algo_name, algo_info in relevant_algorithms.items():
            score, confidence, reasoning = self._score_algorithm(
                algo_name, algo_info, analysis_results, task_type
            )
            
            recommendation = AlgorithmRecommendation(
                name=algo_name,
                category=algo_info['category'],
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                pros=algo_info.get('pros', []),
                cons=algo_info.get('cons', []),
                best_for=algo_info.get('best_for', []),
                hyperparameters_suggestions=algo_info.get('hyperparameters', {})
            )
            recommendations.append(recommendation)
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations
    
    def _infer_task_type(self, analysis_results: Dict[str, Any]) -> str:
        """Infer task type from analysis results"""
        target_analysis = analysis_results.get('target_analysis')
        if target_analysis:
            return target_analysis.get('type', 'classification')
        return 'classification'
    
    def _get_relevant_algorithms(self, task_type: str) -> Dict[str, Any]:
        """Get algorithms relevant to the task type"""
        if task_type == 'classification':
            return {k: v for k, v in self.algorithms.items() if v['task_type'] == 'classification'}
        elif task_type == 'regression':
            return {k: v for k, v in self.algorithms.items() if v['task_type'] == 'regression'}
        elif task_type == 'clustering':
            return {k: v for k, v in self.algorithms.items() if v['task_type'] == 'clustering'}
        elif task_type == 'dimensionality_reduction':
            return {k: v for k, v in self.algorithms.items() if v['task_type'] == 'dimensionality_reduction'}
        elif task_type == 'unsupervised':
            # Return both clustering and dimensionality reduction
            return {k: v for k, v in self.algorithms.items() if v['task_type'] in ['clustering', 'dimensionality_reduction']}
        else:
            return self.algorithms
    
    def _score_algorithm(self, algo_name: str, algo_info: Dict[str, Any], 
                        analysis_results: Dict[str, Any], task_type: str) -> Tuple[float, float, List[str]]:
        """Score an algorithm based on dataset characteristics"""
        score = 0.0
        reasoning = []
        confidence_factors = []
        
        basic_stats = analysis_results.get('basic_stats', {})
        data_quality = analysis_results.get('data_quality', {})
        feature_analysis = analysis_results.get('feature_analysis', {})
        target_analysis = analysis_results.get('target_analysis', {})
        complexity = analysis_results.get('complexity_metrics', {})
        
        n_samples = basic_stats.get('rows', 0)
        n_features = basic_stats.get('columns', 0)
        missing_pct = data_quality.get('missing_percentage', 0)
        dataset_size = complexity.get('dataset_size', 'medium')
        complexity_score = complexity.get('complexity_score', 0.5)
        
        # Algorithm-specific scoring logic
        algo_rules = algo_info.get('scoring_rules', {})
        
        # Sample size considerations
        if 'min_samples' in algo_rules:
            if n_samples >= algo_rules['min_samples']:
                score += 0.15
                reasoning.append(f"Dataset size ({n_samples}) meets minimum requirements")
            else:
                score -= 0.2
                reasoning.append(f"Dataset may be too small for {algo_name}")
        
        if 'large_dataset_bonus' in algo_rules and dataset_size == 'large':
            score += 0.1
            reasoning.append("Large dataset - algorithm scales well")
        
        # Feature count considerations
        if 'handles_high_dim' in algo_rules and algo_rules['handles_high_dim']:
            if n_features > 50:
                score += 0.1
                reasoning.append("Handles high-dimensional data well")
        
        if 'feature_selection_benefit' in algo_rules and algo_rules['feature_selection_benefit']:
            if n_features > 100:
                score += 0.1
                reasoning.append("Can benefit from feature selection")
        
        # Missing data handling
        if 'handles_missing' in algo_rules and algo_rules['handles_missing']:
            if missing_pct > 5:
                score += 0.15
                reasoning.append("Handles missing data natively")
        else:
            if missing_pct > 20:
                score -= 0.1
                reasoning.append("Requires preprocessing for missing data")
        
        # Feature type considerations
        numeric_count = basic_stats.get('numeric_columns', 0)
        categorical_count = basic_stats.get('categorical_columns', 0)
        
        if 'handles_categorical' in algo_rules and algo_rules['handles_categorical']:
            if categorical_count > 0:
                score += 0.1
                reasoning.append("Handles categorical features well")
        
        if 'numeric_preferred' in algo_rules and algo_rules['numeric_preferred']:
            if numeric_count > categorical_count:
                score += 0.1
                reasoning.append("Works best with numeric features")
        
        # Complexity considerations
        if 'handles_complexity' in algo_rules:
            if complexity_score > 0.7 and algo_rules['handles_complexity']:
                score += 0.1
                reasoning.append("Handles complex datasets well")
        
        # Task-specific considerations
        if task_type == 'classification' and target_analysis:
            n_classes = target_analysis.get('n_classes', 2)
            class_imbalance = target_analysis.get('class_imbalance', {})
            is_imbalanced = class_imbalance.get('is_imbalanced', False)
            
            if 'handles_multiclass' in algo_rules and algo_rules['handles_multiclass']:
                if n_classes > 2:
                    score += 0.1
                    reasoning.append("Handles multiclass problems well")
            
            if 'handles_imbalance' in algo_rules and algo_rules['handles_imbalance']:
                if is_imbalanced:
                    score += 0.15
                    reasoning.append("Handles class imbalance effectively")
            elif is_imbalanced:
                score -= 0.1
                reasoning.append("May struggle with class imbalance")
        
        elif task_type == 'regression' and target_analysis:
            regression_metrics = target_analysis.get('regression_metrics', {})
            skewness = abs(regression_metrics.get('skewness', 0))
            
            if 'handles_nonlinear' in algo_rules and algo_rules['handles_nonlinear']:
                if skewness > 1:
                    score += 0.1
                    reasoning.append("Handles non-linear relationships")
        
        # Feature correlation considerations
        feature_corr = feature_analysis.get('feature_correlation', 0)
        if 'handles_correlation' in algo_rules and algo_rules['handles_correlation']:
            if feature_corr > 0.7:
                score += 0.1
                reasoning.append("Handles correlated features well")
        
        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, (score + 0.5) / 1.5))
        
        # Calculate confidence
        confidence = min(1.0, 0.5 + (len(reasoning) * 0.05) + (1 - complexity_score) * 0.3)
        confidence_factors.append(f"Based on {len(reasoning)} matching criteria")
        
        return score, confidence, reasoning
    
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """Initialize comprehensive algorithm database"""
        return {
            # Classification Algorithms
            'Random Forest': {
                'task_type': 'classification',
                'category': 'Ensemble',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Handles non-linear relationships',
                    'Feature importance available',
                    'Robust to outliers',
                    'No feature scaling required',
                    'Works well with mixed data types'
                ],
                'cons': [
                    'Can overfit with small datasets',
                    'Less interpretable than single trees',
                    'Memory intensive for large datasets'
                ],
                'best_for': [
                    'Medium to large datasets',
                    'Non-linear relationships',
                    'Feature importance analysis needed',
                    'Mixed data types'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            },
            'XGBoost': {
                'task_type': 'classification',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'State-of-the-art performance',
                    'Handles missing values natively',
                    'Feature importance available',
                    'Excellent for structured data',
                    'Fast training and prediction'
                ],
                'cons': [
                    'Requires hyperparameter tuning',
                    'Can overfit if not tuned properly',
                    'Less interpretable',
                    'Memory intensive'
                ],
                'best_for': [
                    'Large datasets',
                    'Competitions and production',
                    'Structured/tabular data',
                    'When maximum performance is needed'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                }
            },
            'LightGBM': {
                'task_type': 'classification',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Very fast training',
                    'Low memory usage',
                    'Handles categorical features natively',
                    'Excellent for large datasets',
                    'Good default parameters'
                ],
                'cons': [
                    'Can overfit small datasets',
                    'Less interpretable',
                    'Requires careful tuning for small data'
                ],
                'best_for': [
                    'Very large datasets',
                    'When speed is critical',
                    'High-cardinality categorical features',
                    'Memory-constrained environments'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'num_leaves': 31
                }
            },
            'CatBoost': {
                'task_type': 'classification',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Best categorical feature handling',
                    'Minimal hyperparameter tuning needed',
                    'Handles missing values automatically',
                    'Robust to overfitting',
                    'Good default parameters'
                ],
                'cons': [
                    'Slower than LightGBM',
                    'Less flexible than XGBoost',
                    'Memory intensive'
                ],
                'best_for': [
                    'Datasets with many categorical features',
                    'When minimal tuning is desired',
                    'Production systems',
                    'Mixed data types'
                ],
                'hyperparameters': {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1
                }
            },
            'Support Vector Machine (SVM)': {
                'task_type': 'classification',
                'category': 'Kernel Methods',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Effective in high-dimensional spaces',
                    'Memory efficient',
                    'Versatile (different kernel functions)',
                    'Works well with clear margin of separation'
                ],
                'cons': [
                    'Does not scale well to large datasets',
                    'Requires feature scaling',
                    'Sensitive to hyperparameters',
                    'Poor probability estimates'
                ],
                'best_for': [
                    'Small to medium datasets',
                    'High-dimensional data',
                    'Clear margin of separation',
                    'Text classification'
                ],
                'hyperparameters': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                }
            },
            'Logistic Regression': {
                'task_type': 'classification',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': False,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Fast and simple',
                    'Highly interpretable',
                    'Provides probability estimates',
                    'No hyperparameter tuning needed',
                    'Works well as baseline'
                ],
                'cons': [
                    'Assumes linear relationship',
                    'Requires feature scaling',
                    'Sensitive to outliers',
                    'May underperform on complex data'
                ],
                'best_for': [
                    'Baseline model',
                    'Interpretability required',
                    'Linear relationships',
                    'Small to medium datasets'
                ],
                'hyperparameters': {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs'
                }
            },
            'Neural Network': {
                'task_type': 'classification',
                'category': 'Deep Learning',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Can model complex non-linear relationships',
                    'Automatic feature learning',
                    'Scalable to large datasets',
                    'Flexible architecture'
                ],
                'cons': [
                    'Requires large datasets',
                    'Black box model',
                    'Requires hyperparameter tuning',
                    'Computationally expensive',
                    'Requires feature scaling'
                ],
                'best_for': [
                    'Large datasets',
                    'Complex non-linear patterns',
                    'When other methods plateau',
                    'Deep feature learning needed'
                ],
                'hyperparameters': {
                    'hidden_layer_sizes': (100,),
                    'activation': 'relu',
                    'learning_rate': 'constant',
                    'max_iter': 200
                }
            },
            'K-Nearest Neighbors (KNN)': {
                'task_type': 'classification',
                'category': 'Instance-Based',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': False,
                    'handles_nonlinear': True,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Simple and intuitive',
                    'No training phase',
                    'Works well for non-linear data',
                    'Can be used for both classification and regression'
                ],
                'cons': [
                    'Computationally expensive for large datasets',
                    'Sensitive to irrelevant features',
                    'Requires feature scaling',
                    'Sensitive to curse of dimensionality'
                ],
                'best_for': [
                    'Small datasets',
                    'Non-linear relationships',
                    'When interpretability is needed',
                    'Low-dimensional data'
                ],
                'hyperparameters': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto'
                }
            },
            'Naive Bayes': {
                'task_type': 'classification',
                'category': 'Probabilistic',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': True,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': False,
                    'handles_correlation': False
                },
                'pros': [
                    'Fast training and prediction',
                    'Works well with small datasets',
                    'Good for text classification',
                    'Handles multiple classes naturally',
                    'Probabilistic predictions'
                ],
                'cons': [
                    'Assumes feature independence',
                    'May underperform on complex data',
                    'Sensitive to irrelevant features'
                ],
                'best_for': [
                    'Text classification',
                    'Small datasets',
                    'High-dimensional sparse data',
                    'When speed is critical'
                ],
                'hyperparameters': {
                    'alpha': 1.0,
                    'fit_prior': True
                }
            },
            # Regression Algorithms
            'Random Forest Regressor': {
                'task_type': 'regression',
                'category': 'Ensemble',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Handles non-linear relationships',
                    'Feature importance available',
                    'Robust to outliers',
                    'No feature scaling required'
                ],
                'cons': [
                    'Can overfit with small datasets',
                    'Less interpretable',
                    'May not extrapolate well'
                ],
                'best_for': [
                    'Non-linear relationships',
                    'Feature importance needed',
                    'Medium to large datasets'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2
                }
            },
            'XGBoost Regressor': {
                'task_type': 'regression',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'State-of-the-art performance',
                    'Handles missing values',
                    'Excellent for structured data',
                    'Fast training'
                ],
                'cons': [
                    'Requires tuning',
                    'Can overfit',
                    'Less interpretable'
                ],
                'best_for': [
                    'Large datasets',
                    'Maximum performance needed',
                    'Structured data'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            },
            'Linear Regression': {
                'task_type': 'regression',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Fast and simple',
                    'Highly interpretable',
                    'No hyperparameter tuning',
                    'Good baseline'
                ],
                'cons': [
                    'Assumes linear relationship',
                    'Sensitive to outliers',
                    'Requires feature scaling'
                ],
                'best_for': [
                    'Baseline model',
                    'Linear relationships',
                    'Interpretability needed'
                ],
                'hyperparameters': {
                    'fit_intercept': True,
                    'normalize': False
                }
            },
            'Ridge Regression': {
                'task_type': 'regression',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'handles_correlation': True,
                    'numeric_preferred': True
                },
                'pros': [
                    'Handles multicollinearity',
                    'Regularization prevents overfitting',
                    'Interpretable',
                    'Works well with many features'
                ],
                'cons': [
                    'Assumes linear relationship',
                    'Requires hyperparameter tuning',
                    'Sensitive to outliers'
                ],
                'best_for': [
                    'Multicollinear features',
                    'Many features',
                    'Regularization needed'
                ],
                'hyperparameters': {
                    'alpha': 1.0,
                    'fit_intercept': True
                }
            },
            'Lasso Regression': {
                'task_type': 'regression',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'handles_correlation': False,
                    'feature_selection_benefit': True,
                    'numeric_preferred': True
                },
                'pros': [
                    'Automatic feature selection',
                    'Handles high-dimensional data',
                    'Regularization',
                    'Interpretable'
                ],
                'cons': [
                    'Assumes linear relationship',
                    'May eliminate important features',
                    'Requires tuning'
                ],
                'best_for': [
                    'Feature selection needed',
                    'High-dimensional data',
                    'Sparse solutions'
                ],
                'hyperparameters': {
                    'alpha': 1.0,
                    'fit_intercept': True
                }
            },
            'Support Vector Regression (SVR)': {
                'task_type': 'regression',
                'category': 'Kernel Methods',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Handles non-linear relationships',
                    'Robust to outliers',
                    'Memory efficient',
                    'Versatile kernels'
                ],
                'cons': [
                    'Does not scale to large datasets',
                    'Requires feature scaling',
                    'Sensitive to hyperparameters'
                ],
                'best_for': [
                    'Non-linear relationships',
                    'Small to medium datasets',
                    'Outlier robustness needed'
                ],
                'hyperparameters': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'epsilon': 0.1
                }
            },
            # Additional Regression Algorithms
            'Polynomial Regression': {
                'task_type': 'regression',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Captures non-linear relationships',
                    'Interpretable coefficients',
                    'Simple to implement',
                    'Good for curved relationships'
                ],
                'cons': [
                    'Prone to overfitting with high degrees',
                    'Extrapolation can be unreliable',
                    'Sensitive to outliers',
                    'Feature scaling required'
                ],
                'best_for': [
                    'Non-linear but smooth relationships',
                    'Small to medium datasets',
                    'When relationship has polynomial nature'
                ],
                'hyperparameters': {
                    'degree': 2,
                    'fit_intercept': True
                }
            },
            'Elastic Net Regression': {
                'task_type': 'regression',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'handles_correlation': True,
                    'feature_selection_benefit': True,
                    'numeric_preferred': True
                },
                'pros': [
                    'Combines L1 and L2 regularization',
                    'Feature selection and multicollinearity handling',
                    'More stable than Lasso alone',
                    'Good for high-dimensional data'
                ],
                'cons': [
                    'Requires tuning two hyperparameters',
                    'Assumes linear relationship',
                    'Computationally more expensive than Ridge/Lasso'
                ],
                'best_for': [
                    'High-dimensional data with correlated features',
                    'When both feature selection and regularization needed',
                    'Datasets with many predictors'
                ],
                'hyperparameters': {
                    'alpha': 1.0,
                    'l1_ratio': 0.5,
                    'fit_intercept': True
                }
            },
            'Bayesian Regression': {
                'task_type': 'regression',
                'category': 'Probabilistic',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'handles_correlation': True,
                    'numeric_preferred': True
                },
                'pros': [
                    'Provides uncertainty estimates',
                    'Natural regularization',
                    'Robust to overfitting',
                    'No need for cross-validation'
                ],
                'cons': [
                    'Computationally expensive',
                    'Assumes linear relationship',
                    'Requires prior specification'
                ],
                'best_for': [
                    'When uncertainty quantification is important',
                    'Small datasets',
                    'When prior knowledge is available'
                ],
                'hyperparameters': {
                    'n_iter': 300,
                    'alpha_1': 1e-06,
                    'alpha_2': 1e-06
                }
            },
            'Decision Tree Regressor': {
                'task_type': 'regression',
                'category': 'Tree-Based',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True
                },
                'pros': [
                    'Easy to understand and interpret',
                    'Handles non-linear relationships',
                    'No feature scaling needed',
                    'Can handle mixed data types'
                ],
                'cons': [
                    'Prone to overfitting',
                    'Unstable (small changes in data)',
                    'Biased with imbalanced data',
                    'Not ideal for extrapolation'
                ],
                'best_for': [
                    'Interpretability needed',
                    'Non-linear relationships',
                    'Mixed feature types',
                    'Baseline model'
                ],
                'hyperparameters': {
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            },
            'Extra Trees Regressor': {
                'task_type': 'regression',
                'category': 'Ensemble',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Faster than Random Forest',
                    'Less variance than Random Forest',
                    'Good feature importance',
                    'Handles non-linearity well'
                ],
                'cons': [
                    'Can overfit on noisy data',
                    'Less accurate than Random Forest sometimes',
                    'Memory intensive'
                ],
                'best_for': [
                    'Large datasets',
                    'When speed is important',
                    'Non-linear relationships'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2
                }
            },
            'Gradient Boosting Regressor': {
                'task_type': 'regression',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 500,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'High predictive accuracy',
                    'Handles non-linear relationships',
                    'Feature importance available',
                    'Robust to outliers'
                ],
                'cons': [
                    'Slower training than Random Forest',
                    'Requires careful tuning',
                    'Can overfit',
                    'Less interpretable'
                ],
                'best_for': [
                    'High accuracy needed',
                    'Structured data',
                    'Medium to large datasets'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }
            },
            'AdaBoost Regressor': {
                'task_type': 'regression',
                'category': 'Ensemble',
                'scoring_rules': {
                    'min_samples': 200,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': False
                },
                'pros': [
                    'Less prone to overfitting than other boosting',
                    'Good for weak learners',
                    'No need for feature scaling',
                    'Simple to implement'
                ],
                'cons': [
                    'Sensitive to noisy data and outliers',
                    'Slower than other methods',
                    'Can underperform on complex data'
                ],
                'best_for': [
                    'Small to medium datasets',
                    'When data is not too noisy',
                    'Combining weak models'
                ],
                'hyperparameters': {
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'loss': 'linear'
                }
            },
            'LightGBM Regressor': {
                'task_type': 'regression',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Very fast training',
                    'Low memory usage',
                    'Handles categorical features',
                    'Excellent for large datasets'
                ],
                'cons': [
                    'Can overfit small datasets',
                    'Less interpretable',
                    'Requires parameter tuning'
                ],
                'best_for': [
                    'Very large datasets',
                    'When speed is critical',
                    'High-cardinality categorical features'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'num_leaves': 31
                }
            },
            'CatBoost Regressor': {
                'task_type': 'regression',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Best categorical handling',
                    'Minimal tuning needed',
                    'Handles missing values',
                    'Robust to overfitting'
                ],
                'cons': [
                    'Slower than LightGBM',
                    'Memory intensive',
                    'Less flexible'
                ],
                'best_for': [
                    'Many categorical features',
                    'Minimal tuning desired',
                    'Production systems'
                ],
                'hyperparameters': {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1
                }
            },
            'KNN Regressor': {
                'task_type': 'regression',
                'category': 'Instance-Based',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Simple and intuitive',
                    'No training phase',
                    'Non-parametric',
                    'Can adapt to local patterns'
                ],
                'cons': [
                    'Computationally expensive for large datasets',
                    'Sensitive to feature scaling',
                    'Curse of dimensionality',
                    'Poor extrapolation'
                ],
                'best_for': [
                    'Small datasets',
                    'Non-linear relationships',
                    'Low-dimensional data'
                ],
                'hyperparameters': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto'
                }
            },
            'Neural Network Regressor': {
                'task_type': 'regression',
                'category': 'Deep Learning',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Handles complex non-linear patterns',
                    'Automatic feature learning',
                    'Scalable to large datasets',
                    'Flexible architecture'
                ],
                'cons': [
                    'Requires large datasets',
                    'Black box model',
                    'Hyperparameter tuning needed',
                    'Computationally expensive'
                ],
                'best_for': [
                    'Large datasets',
                    'Complex patterns',
                    'When maximum accuracy needed'
                ],
                'hyperparameters': {
                    'hidden_layer_sizes': (100,),
                    'activation': 'relu',
                    'learning_rate': 'constant',
                    'max_iter': 200
                }
            },
            # Additional Classification Algorithms
            'Ridge Classifier': {
                'task_type': 'classification',
                'category': 'Linear Models',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': False,
                    'handles_nonlinear': False,
                    'handles_correlation': True,
                    'numeric_preferred': True
                },
                'pros': [
                    'Fast training',
                    'Handles multicollinearity',
                    'Regularization prevents overfitting',
                    'Works well with many features'
                ],
                'cons': [
                    'Assumes linear decision boundary',
                    'No probability estimates',
                    'Requires feature scaling'
                ],
                'best_for': [
                    'High-dimensional data',
                    'Multicollinear features',
                    'Fast baseline model'
                ],
                'hyperparameters': {
                    'alpha': 1.0,
                    'fit_intercept': True,
                    'normalize': False
                }
            },
            'Decision Tree Classifier': {
                'task_type': 'classification',
                'category': 'Tree-Based',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': True,
                    'handles_multiclass': True,
                    'handles_imbalance': False,
                    'handles_nonlinear': True,
                    'handles_correlation': True
                },
                'pros': [
                    'Easy to understand and interpret',
                    'Visual representation possible',
                    'No feature scaling needed',
                    'Handles non-linear relationships'
                ],
                'cons': [
                    'Prone to overfitting',
                    'Unstable',
                    'Biased with imbalanced classes',
                    'Can create over-complex trees'
                ],
                'best_for': [
                    'Interpretability critical',
                    'Non-linear relationships',
                    'Mixed feature types',
                    'Rule extraction needed'
                ],
                'hyperparameters': {
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'criterion': 'gini'
                }
            },
            'Extra Trees Classifier': {
                'task_type': 'classification',
                'category': 'Ensemble',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Faster than Random Forest',
                    'Good generalization',
                    'Feature importance available',
                    'Reduces variance'
                ],
                'cons': [
                    'Can overfit noisy data',
                    'Less accurate than RF sometimes',
                    'Memory intensive',
                    'Less interpretable'
                ],
                'best_for': [
                    'Large datasets',
                    'When speed matters',
                    'High-dimensional data'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'criterion': 'gini'
                }
            },
            'Gradient Boosting Classifier': {
                'task_type': 'classification',
                'category': 'Gradient Boosting',
                'scoring_rules': {
                    'min_samples': 500,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': True,
                    'handles_nonlinear': True,
                    'handles_correlation': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'High predictive accuracy',
                    'Handles non-linear relationships',
                    'Feature importance',
                    'Flexible loss functions'
                ],
                'cons': [
                    'Slower training',
                    'Requires careful tuning',
                    'Can overfit',
                    'Less interpretable'
                ],
                'best_for': [
                    'High accuracy required',
                    'Structured data',
                    'Medium to large datasets'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'subsample': 1.0
                }
            },
            'AdaBoost Classifier': {
                'task_type': 'classification',
                'category': 'Ensemble',
                'scoring_rules': {
                    'min_samples': 200,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_multiclass': True,
                    'handles_imbalance': False,
                    'handles_nonlinear': True,
                    'handles_correlation': False
                },
                'pros': [
                    'Less prone to overfitting',
                    'Good with weak learners',
                    'No feature scaling needed',
                    'Simple implementation'
                ],
                'cons': [
                    'Sensitive to noisy data',
                    'Sensitive to outliers',
                    'Can be slow'
                ],
                'best_for': [
                    'Small to medium datasets',
                    'Clean data',
                    'Binary classification'
                ],
                'hyperparameters': {
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'algorithm': 'SAMME.R'
                }
            },
            # Clustering Algorithms
            'K-Means Clustering': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Fast and simple',
                    'Scalable to large datasets',
                    'Easy to interpret',
                    'Well-established algorithm'
                ],
                'cons': [
                    'Requires predefined k',
                    'Sensitive to initialization',
                    'Assumes spherical clusters',
                    'Sensitive to outliers'
                ],
                'best_for': [
                    'Well-separated spherical clusters',
                    'Large datasets',
                    'When k is known'
                ],
                'hyperparameters': {
                    'n_clusters': 8,
                    'init': 'k-means++',
                    'n_init': 10,
                    'max_iter': 300
                }
            },
            'Mini-Batch K-Means': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'numeric_preferred': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Very fast for large datasets',
                    'Memory efficient',
                    'Can handle streaming data',
                    'Good approximation of K-Means'
                ],
                'cons': [
                    'Less accurate than K-Means',
                    'Requires predefined k',
                    'Sensitive to initialization'
                ],
                'best_for': [
                    'Very large datasets',
                    'Streaming data',
                    'When speed is critical'
                ],
                'hyperparameters': {
                    'n_clusters': 8,
                    'batch_size': 100,
                    'max_iter': 100
                }
            },
            'DBSCAN': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'No need to specify number of clusters',
                    'Can find arbitrary shaped clusters',
                    'Identifies outliers/noise',
                    'Robust to outliers'
                ],
                'cons': [
                    'Sensitive to parameters',
                    'Struggles with varying densities',
                    'Not suitable for high dimensions',
                    'Computationally expensive'
                ],
                'best_for': [
                    'Arbitrary shaped clusters',
                    'Data with noise',
                    'Unknown number of clusters'
                ],
                'hyperparameters': {
                    'eps': 0.5,
                    'min_samples': 5,
                    'metric': 'euclidean'
                }
            },
            'HDBSCAN': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'Handles varying density clusters',
                    'No need to specify eps',
                    'Robust to parameter choices',
                    'Good outlier detection'
                ],
                'cons': [
                    'Computationally expensive',
                    'Not deterministic',
                    'Requires external library'
                ],
                'best_for': [
                    'Varying density clusters',
                    'Complex cluster structures',
                    'Outlier detection'
                ],
                'hyperparameters': {
                    'min_cluster_size': 5,
                    'min_samples': 5,
                    'metric': 'euclidean'
                }
            },
            'Agglomerative Clustering': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'Hierarchical structure',
                    'Can use different linkage methods',
                    'Dendrogram visualization',
                    'Works with any distance metric'
                ],
                'cons': [
                    'Computationally expensive O(n³)',
                    'Cannot undo merges',
                    'Not scalable to large datasets',
                    'Sensitive to noise'
                ],
                'best_for': [
                    'Small to medium datasets',
                    'Hierarchical relationships',
                    'When dendrogram needed'
                ],
                'hyperparameters': {
                    'n_clusters': 2,
                    'linkage': 'ward',
                    'affinity': 'euclidean'
                }
            },
            'Gaussian Mixture Model (GMM)': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 200,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'Soft clustering (probabilities)',
                    'Flexible cluster shapes',
                    'Probabilistic model',
                    'Can determine optimal clusters'
                ],
                'cons': [
                    'Sensitive to initialization',
                    'Can converge to local optima',
                    'Assumes Gaussian distribution',
                    'Computationally expensive'
                ],
                'best_for': [
                    'Elliptical clusters',
                    'Soft clustering needed',
                    'Probability estimates required'
                ],
                'hyperparameters': {
                    'n_components': 1,
                    'covariance_type': 'full',
                    'max_iter': 100
                }
            },
            'Spectral Clustering': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'Handles non-convex clusters',
                    'Works with similarity matrix',
                    'Good for graph-based data',
                    'Effective for complex structures'
                ],
                'cons': [
                    'Computationally expensive',
                    'Requires number of clusters',
                    'Memory intensive',
                    'Not scalable to very large datasets'
                ],
                'best_for': [
                    'Non-convex clusters',
                    'Graph-structured data',
                    'Image segmentation'
                ],
                'hyperparameters': {
                    'n_clusters': 8,
                    'affinity': 'rbf',
                    'n_neighbors': 10
                }
            },
            'OPTICS': {
                'task_type': 'clustering',
                'category': 'Clustering',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': False,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'Handles varying density',
                    'No need for distance threshold',
                    'Produces cluster hierarchy',
                    'Outlier detection'
                ],
                'cons': [
                    'Computationally expensive',
                    'Memory intensive',
                    'Complex to tune',
                    'Results can be hard to interpret'
                ],
                'best_for': [
                    'Varying density clusters',
                    'Hierarchical cluster structure',
                    'Exploratory analysis'
                ],
                'hyperparameters': {
                    'min_samples': 5,
                    'max_eps': float('inf'),
                    'metric': 'euclidean'
                }
            },
            # Dimensionality Reduction
            'Principal Component Analysis (PCA)': {
                'task_type': 'dimensionality_reduction',
                'category': 'Dimensionality Reduction',
                'scoring_rules': {
                    'min_samples': 50,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'handles_correlation': True,
                    'numeric_preferred': True
                },
                'pros': [
                    'Reduces dimensionality',
                    'Removes correlated features',
                    'Fast and efficient',
                    'Variance retention control'
                ],
                'cons': [
                    'Linear transformation only',
                    'Components hard to interpret',
                    'Sensitive to scaling',
                    'Assumes linear relationships'
                ],
                'best_for': [
                    'High-dimensional data',
                    'Feature extraction',
                    'Visualization',
                    'Noise reduction'
                ],
                'hyperparameters': {
                    'n_components': None,
                    'whiten': False,
                    'svd_solver': 'auto'
                }
            },
            'Linear Discriminant Analysis (LDA)': {
                'task_type': 'dimensionality_reduction',
                'category': 'Dimensionality Reduction',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': False,
                    'numeric_preferred': True
                },
                'pros': [
                    'Supervised dimensionality reduction',
                    'Maximizes class separability',
                    'Good for classification',
                    'Can be used as classifier'
                ],
                'cons': [
                    'Requires labeled data',
                    'Assumes Gaussian distribution',
                    'Linear transformation only',
                    'Limited by number of classes'
                ],
                'best_for': [
                    'Classification preprocessing',
                    'Maximizing class separation',
                    'Labeled data available'
                ],
                'hyperparameters': {
                    'n_components': None,
                    'solver': 'svd',
                    'shrinkage': None
                }
            },
            't-SNE': {
                'task_type': 'dimensionality_reduction',
                'category': 'Dimensionality Reduction',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True
                },
                'pros': [
                    'Excellent for visualization',
                    'Preserves local structure',
                    'Reveals cluster structure',
                    'Non-linear dimensionality reduction'
                ],
                'cons': [
                    'Computationally expensive',
                    'Not deterministic',
                    'Sensitive to hyperparameters',
                    'Only for visualization (not for ML input)'
                ],
                'best_for': [
                    '2D/3D visualization',
                    'Exploratory data analysis',
                    'Cluster visualization'
                ],
                'hyperparameters': {
                    'n_components': 2,
                    'perplexity': 30.0,
                    'learning_rate': 200.0,
                    'n_iter': 1000
                }
            },
            'UMAP': {
                'task_type': 'dimensionality_reduction',
                'category': 'Dimensionality Reduction',
                'scoring_rules': {
                    'min_samples': 100,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Faster than t-SNE',
                    'Preserves global and local structure',
                    'Better for general purpose',
                    'Can be used for ML input'
                ],
                'cons': [
                    'Requires external library',
                    'Sensitive to hyperparameters',
                    'Less established than PCA/t-SNE'
                ],
                'best_for': [
                    'Large dataset visualization',
                    'General dimensionality reduction',
                    'Preprocessing for ML'
                ],
                'hyperparameters': {
                    'n_components': 2,
                    'n_neighbors': 15,
                    'min_dist': 0.1,
                    'metric': 'euclidean'
                }
            },
            'Autoencoders': {
                'task_type': 'dimensionality_reduction',
                'category': 'Deep Learning',
                'scoring_rules': {
                    'min_samples': 1000,
                    'handles_high_dim': True,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'handles_nonlinear': True,
                    'large_dataset_bonus': True
                },
                'pros': [
                    'Non-linear dimensionality reduction',
                    'Flexible architecture',
                    'Can learn complex patterns',
                    'Good for anomaly detection'
                ],
                'cons': [
                    'Requires large datasets',
                    'Computationally expensive',
                    'Requires hyperparameter tuning',
                    'Black box model'
                ],
                'best_for': [
                    'Large datasets',
                    'Complex non-linear relationships',
                    'Anomaly detection',
                    'Feature learning'
                ],
                'hyperparameters': {
                    'encoding_dim': 32,
                    'epochs': 50,
                    'batch_size': 256,
                    'optimizer': 'adam'
                }
            }
        }

