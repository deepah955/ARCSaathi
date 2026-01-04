# Comprehensive List of Machine Learning Algorithms with Workflows

This document lists all **60+ Machine Learning algorithms** supported by the AI-Powered ML Algorithm Recommender. Each algorithm includes its standard workflow suitable for academic reviews and project evaluations.

## Standard Workflow Pattern
Most algorithms follow this general workflow:
```
Dataset → Data Cleaning → Feature Selection/Scaling → Model Training → Prediction → Evaluation Metrics
```

---

## Regression Algorithms (18)

### 1. Linear Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Linear Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Simple linear relationships, baseline model

**Key Characteristics:** Fast, interpretable, assumes linearity

---

### 2. Multiple Linear Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Multiple Linear Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Multiple predictors with linear relationships

**Key Characteristics:** Extension of linear regression, handles multiple features

---

### 3. Polynomial Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Polynomial Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Non-linear but polynomial relationships

**Key Characteristics:** Captures curved relationships, risk of overfitting

---

### 4. Ridge Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Ridge Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Multicollinear features, regularization needed

**Key Characteristics:** L2 regularization, handles correlated features

---

### 5. Lasso Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Lasso Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Feature selection, high-dimensional data

**Key Characteristics:** L1 regularization, automatic feature selection

---

### 6. Elastic Net Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Elastic Net Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** High-dimensional data with correlated features

**Key Characteristics:** Combines L1 and L2 regularization

---

### 7. Bayesian Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Bayesian Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Uncertainty quantification needed

**Key Characteristics:** Probabilistic approach, provides uncertainty estimates

---

### 8. Decision Tree Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Decision Tree Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Non-linear relationships, interpretability important

**Key Characteristics:** Easy to interpret, handles non-linearity, prone to overfitting

---

### 9. Random Forest Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Random Forest Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Non-linear relationships, robust predictions

**Key Characteristics:** Ensemble of trees, handles non-linearity, feature importance

---

### 10. Extra Trees Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Extra Trees Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Large datasets, faster than Random Forest

**Key Characteristics:** Faster training, less variance, good generalization

---

### 11. Gradient Boosting Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Gradient Boosting Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** High accuracy required, structured data

**Key Characteristics:** Sequential boosting, high accuracy, requires tuning

---

### 12. AdaBoost Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → AdaBoost Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Combining weak learners, clean data

**Key Characteristics:** Adaptive boosting, less prone to overfitting

---

### 13. XGBoost Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → XGBoost Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Competition-level performance, structured data

**Key Characteristics:** State-of-the-art performance, handles missing values, fast

---

### 14. LightGBM Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → LightGBM Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Very large datasets, speed critical

**Key Characteristics:** Very fast, low memory, handles categorical features

---

### 15. CatBoost Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → CatBoost Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Many categorical features, minimal tuning

**Key Characteristics:** Best categorical handling, minimal tuning needed

---

### 16. KNN Regressor
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → KNN Regressor Model Training → Prediction → Evaluation Metrics

**Use Case:** Small datasets, non-linear patterns

**Key Characteristics:** Instance-based, no training phase, sensitive to scaling

---

### 17. Support Vector Regression (SVR)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Support Vector Regression (SVR) Model Training → Prediction → Evaluation Metrics

**Use Case:** Non-linear relationships, small to medium datasets

**Key Characteristics:** Kernel methods, robust to outliers, requires scaling

---

### 18. Neural Network Regressor (MLP)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Neural Network Regressor (MLP) Model Training → Prediction → Evaluation Metrics

**Use Case:** Large datasets, complex patterns

**Key Characteristics:** Deep learning, handles complex patterns, requires large data

---

## Classification Algorithms (15)

### 1. Logistic Regression
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Logistic Regression Model Training → Prediction → Evaluation Metrics

**Use Case:** Binary/multiclass classification, baseline model

**Key Characteristics:** Interpretable, fast, probability estimates

---

### 2. Ridge Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Ridge Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** High-dimensional classification, multicollinearity

**Key Characteristics:** Fast, regularized, handles correlated features

---

### 3. Decision Tree Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Decision Tree Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Interpretability critical, rule extraction

**Key Characteristics:** Visual representation, easy to interpret, prone to overfitting

---

### 4. Random Forest Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Random Forest Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Non-linear classification, robust predictions

**Key Characteristics:** Ensemble method, handles non-linearity, feature importance

---

### 5. Extra Trees Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Extra Trees Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Large datasets, faster training

**Key Characteristics:** Faster than RF, good generalization

---

### 6. Gradient Boosting Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Gradient Boosting Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** High accuracy required

**Key Characteristics:** Sequential boosting, high accuracy, flexible

---

### 7. AdaBoost Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → AdaBoost Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Binary classification, weak learners

**Key Characteristics:** Adaptive boosting, less overfitting

---

### 8. XGBoost Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → XGBoost Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Competition performance, production systems

**Key Characteristics:** State-of-the-art, handles missing values, fast

---

### 9. LightGBM Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → LightGBM Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Very large datasets, speed critical

**Key Characteristics:** Very fast, memory efficient

---

### 10. CatBoost Classifier
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → CatBoost Classifier Model Training → Prediction → Evaluation Metrics

**Use Case:** Many categorical features

**Key Characteristics:** Best categorical handling, robust

---

### 11. K-Nearest Neighbors (KNN)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → K-Nearest Neighbors (KNN) Model Training → Prediction → Evaluation Metrics

**Use Case:** Small datasets, non-linear boundaries

**Key Characteristics:** Instance-based, no training, intuitive

---

### 12. Support Vector Machine (SVM)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Support Vector Machine (SVM) Model Training → Prediction → Evaluation Metrics

**Use Case:** High-dimensional data, clear margin separation

**Key Characteristics:** Kernel methods, effective in high dimensions

---

### 13. Naive Bayes (Gaussian / Multinomial)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Naive Bayes Model Training → Prediction → Evaluation Metrics

**Use Case:** Text classification, fast predictions

**Key Characteristics:** Fast, probabilistic, assumes independence

---

### 14. Neural Network Classifier (MLP)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Neural Network Classifier (MLP) Model Training → Prediction → Evaluation Metrics

**Use Case:** Large datasets, complex patterns

**Key Characteristics:** Deep learning, automatic feature learning

---

## Clustering Algorithms (10)

### 1. K-Means Clustering
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → K-Means Clustering Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Well-separated spherical clusters

**Key Characteristics:** Fast, simple, requires k parameter

---

### 2. Mini-Batch K-Means
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Mini-Batch K-Means Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Very large datasets, streaming data

**Key Characteristics:** Very fast, memory efficient, approximation of K-Means

---

### 3. K-Medoids
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → K-Medoids Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** When cluster centers should be actual data points

**Key Characteristics:** More robust to outliers than K-Means

---

### 4. Agglomerative Clustering
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Agglomerative Clustering Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Hierarchical relationships, dendrogram needed

**Key Characteristics:** Bottom-up hierarchical, dendrogram visualization

---

### 5. Divisive Clustering
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Divisive Clustering Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Hierarchical relationships, top-down approach

**Key Characteristics:** Top-down hierarchical clustering

---

### 6. DBSCAN
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → DBSCAN Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Arbitrary shaped clusters, outlier detection

**Key Characteristics:** Density-based, finds arbitrary shapes, identifies noise

---

### 7. HDBSCAN
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → HDBSCAN Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Varying density clusters

**Key Characteristics:** Hierarchical DBSCAN, handles varying densities

---

### 8. OPTICS
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → OPTICS Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Varying density, hierarchical structure

**Key Characteristics:** Ordering points for cluster structure, handles varying densities

---

### 9. Gaussian Mixture Model (GMM)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Gaussian Mixture Model (GMM) Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Soft clustering, probability estimates

**Key Characteristics:** Probabilistic clustering, soft assignments, elliptical clusters

---

### 10. Spectral Clustering
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Spectral Clustering Model Training → Cluster Assignment → Evaluation Metrics

**Use Case:** Non-convex clusters, graph-based data

**Key Characteristics:** Graph-based, handles non-convex clusters, effective for complex structures

---

## Dimensionality Reduction Techniques (5)

### 1. Principal Component Analysis (PCA)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Principal Component Analysis (PCA) Model Training → Transformation → Evaluation Metrics

**Use Case:** Feature extraction, visualization, noise reduction

**Key Characteristics:** Linear transformation, variance maximization, orthogonal components

---

### 2. Linear Discriminant Analysis (LDA)
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Linear Discriminant Analysis (LDA) Model Training → Transformation → Evaluation Metrics

**Use Case:** Classification preprocessing, supervised reduction

**Key Characteristics:** Supervised method, maximizes class separability

---

### 3. t-SNE
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → t-SNE Model Training → Transformation → Visualization

**Use Case:** 2D/3D visualization, exploratory analysis

**Key Characteristics:** Non-linear, excellent visualization, preserves local structure

---

### 4. UMAP
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → UMAP Model Training → Transformation → Evaluation/Visualization

**Use Case:** Large dataset visualization, general dimensionality reduction

**Key Characteristics:** Faster than t-SNE, preserves global and local structure

---

### 5. Autoencoders
**Workflow:** Dataset → Data Cleaning → Feature Selection/Scaling → Autoencoders Model Training → Encoding → Evaluation Metrics

**Use Case:** Large datasets, complex patterns, anomaly detection

**Key Characteristics:** Deep learning, non-linear, flexible architecture

---

## Algorithm Selection Guidelines

### By Dataset Size:
- **Small (<1,000 samples):** Logistic Regression, KNN, Decision Trees, Naive Bayes
- **Medium (1,000-100,000):** Random Forest, SVM, Gradient Boosting
- **Large (>100,000):** XGBoost, LightGBM, CatBoost, Mini-Batch K-Means, Neural Networks

### By Feature Count:
- **Low (<10):** Any algorithm
- **Medium (10-100):** Random Forest, Gradient Boosting, Ridge/Lasso
- **High (>100):** Lasso, Elastic Net, PCA + Model, LDA

### By Data Characteristics:
- **Linear relationships:** Linear/Logistic Regression, Ridge, Lasso
- **Non-linear:** Trees, Ensembles, Neural Networks, SVM
- **Categorical features:** CatBoost, LightGBM, Decision Trees
- **Missing values:** XGBoost, LightGBM, CatBoost
- **Outliers:** Tree-based methods, Robust regression

### By Task Requirements:
- **Interpretability:** Linear models, Decision Trees, Naive Bayes
- **Speed:** Naive Bayes, Linear models, KNN
- **Accuracy:** XGBoost, LightGBM, CatBoost, Neural Networks
- **Probability estimates:** Logistic Regression, Naive Bayes, GMM

---

## Evaluation Metrics by Task Type

### Regression Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Classification Metrics:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Log Loss

### Clustering Metrics:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Adjusted Rand Index (if labels available)

### Dimensionality Reduction Metrics:
- Explained Variance Ratio
- Reconstruction Error
- Trustworthiness (for t-SNE/UMAP)

---

## References and Further Reading

For detailed implementation and usage of these algorithms, refer to:
- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- CatBoost Documentation: https://catboost.ai/

---

*This comprehensive guide is part of the AI-Powered ML Algorithm Recommender system, which automatically analyzes your dataset and recommends the most suitable algorithms.*
