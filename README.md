# AI-Powered ML Algorithm Recommender

An intelligent, advanced GUI application that automatically analyzes datasets and recommends the best machine learning algorithms based on comprehensive dataset characteristics.

## 🚀 Features

### Advanced Dataset Analysis
- **Comprehensive Profiling**: Analyzes dataset size, feature types, data quality, and complexity metrics
- **Feature Analysis**: Examines correlations, distributions, skewness, and cardinality
- **Data Quality Assessment**: Detects missing values, duplicates, and data quality issues
- **Target Variable Analysis**: Automatically determines task type (classification/regression) and analyzes target characteristics

### Intelligent Algorithm Recommendation
- **Multi-Factor Scoring**: Uses advanced heuristics to score algorithms based on:
  - Dataset size and dimensionality
  - Feature types and characteristics
  - Data quality metrics
  - Target variable properties
  - Complexity indicators
- **Ranked Recommendations**: Provides top algorithms sorted by relevance score
- **Detailed Reasoning**: Explains why each algorithm is recommended
- **Confidence Metrics**: Shows confidence levels for each recommendation

### Comprehensive Algorithm Database
Supports **60+ algorithms** across multiple categories:

**Regression Algorithms (18):**
- Linear, Multiple Linear, Polynomial, Ridge, Lasso, Elastic Net, Bayesian
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost
- KNN, SVR, Neural Network

**Classification Algorithms (15):**
- Logistic Regression, Ridge Classifier
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost
- KNN, SVM, Naive Bayes, Neural Network

**Clustering Algorithms (10):**
- K-Means, Mini-Batch K-Means, K-Medoids
- Agglomerative, Divisive, DBSCAN, HDBSCAN, OPTICS
- Gaussian Mixture Model (GMM), Spectral Clustering

**Dimensionality Reduction (5):**
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE, UMAP, Autoencoders

### Rich Information for Each Algorithm
- **Pros & Cons**: Detailed advantages and limitations
- **Best Use Cases**: When to use each algorithm
- **Hyperparameter Suggestions**: Recommended starting parameters
- **Category Classification**: Algorithm type and category
- **Workflow Guidance**: Standard ML workflow for each algorithm

### Modern GUI
- **Dark Theme**: Beautiful, modern dark interface
- **Interactive Visualization**: Score comparisons and confidence plots
- **Tabbed Information**: Organized display of recommendations
- **Real-time Analysis**: Progress tracking during analysis
- **File Support**: Load CSV and Excel files
- **Multiple Task Types**: Classification, Regression, Clustering, Dimensionality Reduction

## 📋 Requirements

- Python 3.8 or higher
- See `requirements.txt` for all dependencies

## 🛠️ Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. **Run the application**:
```bash
python main.py
```

2. **Load your dataset**:
   - Click "📁 Load Dataset"
   - Select a CSV or Excel file
   - The dataset will be loaded and analyzed

3. **Configure settings** (optional):
   - Select target column from dropdown (if applicable)
   - Choose task type: Auto-detect, Classification, or Regression

4. **Analyze & Get Recommendations**:
   - Click "🔍 Analyze & Recommend"
   - Wait for analysis to complete
   - View ranked algorithm recommendations with detailed information

5. **Explore Recommendations**:
   - View top 5 algorithms with scores and confidence
   - Explore tabs for each algorithm:
     - **Reasoning**: Why this algorithm is recommended
     - **Pros & Cons**: Advantages and limitations
     - **Best For**: Ideal use cases
     - **Hyperparameters**: Suggested starting parameters
   - Check visualizations for score comparisons

## 📊 How It Works

### Analysis Pipeline

1. **Dataset Loading**: Reads CSV/Excel files into pandas DataFrame

2. **Basic Statistics**: Calculates rows, columns, memory usage, feature types

3. **Data Quality Analysis**:
   - Missing value detection and percentage
   - Duplicate row identification
   - Data completeness metrics

4. **Feature Analysis**:
   - Numeric feature statistics (skewness, kurtosis, variance)
   - Categorical feature cardinality
   - Feature correlation analysis
   - High correlation pair detection

5. **Target Variable Analysis** (if provided):
   - Task type detection (classification vs regression)
   - Class distribution and imbalance detection
   - Regression metrics (mean, std, skewness, etc.)

6. **Complexity Metrics**:
   - Feature-to-sample ratio
   - Dimensionality assessment
   - Overall complexity score

7. **Algorithm Scoring**:
   - Each algorithm is scored based on dataset characteristics
   - Multiple factors contribute to the score:
     - Sample size compatibility
     - Feature count handling
     - Missing data tolerance
     - Feature type support
     - Task-specific capabilities
   - Confidence is calculated based on matching criteria

8. **Recommendation Generation**:
   - Algorithms are ranked by score
   - Top recommendations are displayed with full details
   - Visualizations show score distributions

## 🎨 Algorithm Selection Criteria

The system considers:

- **Dataset Size**: Small (<1K), Medium (1K-100K), Large (>100K)
- **Feature Count**: Low (<10), Medium (10-50), High (>50)
- **Missing Data**: Percentage and handling requirements
- **Feature Types**: Numeric vs categorical balance
- **Target Characteristics**: 
  - Classification: Number of classes, class imbalance
  - Regression: Distribution, skewness, variance
- **Complexity**: Overall dataset complexity score
- **Correlations**: Feature correlation patterns

## 📈 Example Use Cases

- **Small Dataset Classification**: Recommends algorithms like Logistic Regression, Naive Bayes, or KNN
- **Large Dataset with Many Features**: Suggests XGBoost, LightGBM, or Random Forest
- **High Missing Data**: Recommends algorithms that handle missing values natively (XGBoost, LightGBM, CatBoost)
- **Class Imbalance**: Suggests algorithms with built-in imbalance handling
- **Many Categorical Features**: Recommends CatBoost or algorithms with good categorical support
- **High-Dimensional Data**: Suggests algorithms that scale well (SVM, Linear models, Tree-based)

## 🔧 Technical Details

### Architecture
- **dataset_analyzer.py**: Comprehensive dataset analysis engine
- **algorithm_recommender.py**: Intelligent recommendation engine with algorithm database
- **main.py**: Modern GUI application using CustomTkinter

### Key Technologies
- **CustomTkinter**: Modern, customizable Tkinter-based GUI
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: ML algorithm knowledge base

## 🎯 Future Enhancements

Potential additions:
- Export recommendations to PDF/CSV
- Integration with scikit-learn for quick model training
- Additional algorithm support
- Custom algorithm scoring rules
- Batch analysis for multiple datasets
- Performance benchmarking

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more algorithms to the database
- Improve scoring heuristics
- Enhance the GUI
- Add new features

## 💡 Tips

1. **For best results**: Provide a target column if you have one - this enables more accurate task type detection
2. **Large datasets**: The analysis may take a moment - be patient for comprehensive results
3. **Missing data**: Algorithms that handle missing data natively (XGBoost, LightGBM, CatBoost) score higher when missing data is present
4. **Feature engineering**: Review the preliminary recommendations for feature engineering suggestions

---

**Built with ❤️ for the ML community**

