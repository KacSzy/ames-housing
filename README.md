# Housing Sale Price Prediction

Machine learning project for predicting Ames housing sale prices with regression models, feature preprocessing, and
dimensionality reduction experiments.

## Overview

This project explores how well different models can predict the final sale price (`SalePrice`) of homes in the Ames
Housing dataset. The workflow includes:

- data cleaning and preprocessing
- target handling and feature preparation
- dimensionality reduction with PCA and random projection
- model comparison across multiple regressors
- evaluation using RMSE, MAE, and R²

## Models Tested

- XGBoost (`XGBRegressor`)
- Ridge Regression
- Random Forest Regressor
- Support Vector Regressor (`SVR`)
- K-Nearest Neighbors Regressor
- Decision Tree Regressor

## Dimensionality Reduction

The notebook compares model performance using:

- PCA
- Random Projection
- K-Means-based reduction experiments

## Best Result

From the saved comparison table in `data_prepared/model_results_comparison.csv`, the best recorded configuration was:

- **Dataset:** PCA
- **Model:** `XGBRegressor`
- **Optimal K / Components:** 62
- **RMSE:** 22800.42
- **MAE:** 14125.08
- **R²:** 0.9216

## Project Structure

```text
HousingSalePrice/
├── data/
│   └── AmesHousing.csv
├── data_prepared/
│   ├── Ames_pca.csv
│   ├── Ames_rp_*.csv
│   └── model_results_comparison.csv
├── workflow.ipynb
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.12+
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `ipykernel`

## Getting Started

### 1. Install dependencies

If you use `uv`:

```bash
uv sync
```

Or install manually with your preferred environment manager.

### 2. Open the notebook

Run the analysis in `workflow.ipynb`.

### 3. Review prepared outputs

The processed datasets and model comparison results are stored in `data_prepared/`.
