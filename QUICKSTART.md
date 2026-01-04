# Quick Start Guide

## Installation (5 minutes)

1. **Install Python 3.8+** (if not already installed)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

## Quick Test

1. **Create sample data** (optional):
```bash
python create_sample_data.py
```

2. **Load a dataset**:
   - Click "📁 Load Dataset"
   - Select any CSV file (or use the generated samples)

3. **Get recommendations**:
   - Select target column (if applicable)
   - Click "🔍 Analyze & Recommend"
   - View your personalized algorithm recommendations!

## What You'll See

- **Top 5 Algorithm Recommendations** with:
  - Recommendation scores (0-1)
  - Confidence levels
  - Detailed reasoning
  - Pros and cons
  - Best use cases
  - Hyperparameter suggestions

- **Visualizations**:
  - Score comparison bar chart
  - Score vs Confidence scatter plot

## Example Workflow

1. Load `sample_multiclass.csv` (from create_sample_data.py)
2. Select "target" as target column
3. Choose "Auto-detect" for task type
4. Click "Analyze & Recommend"
5. Explore the recommendations!

## Tips

- **Best results**: Provide a target column when available
- **Large datasets**: Analysis may take 10-30 seconds
- **Missing data**: Algorithms that handle missing data score higher
- **Many features**: Tree-based algorithms (XGBoost, Random Forest) excel

Enjoy exploring ML algorithms! 🚀

