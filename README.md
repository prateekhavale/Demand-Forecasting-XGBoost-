# Demand Forecasting using XGBoost (Time Series)

An advanced time series demand forecasting system that predicts product demand across multiple stores using feature engineering + XGBoost regression.

This project focuses on:

* Understanding demand patterns
* Building predictive models using historical data
* Applying lag features, rolling statistics, and temporal features
* Improving performance with hyperparameter tuning (GridSearch + Optuna)

---

## Features

* Time-based feature extraction (day, month, year, week)
* Lag-based demand modeling (2, 3, 7, 14, 30 days)
* Rolling statistics (mean, std)
* Multi-store & multi-product forecasting
* Exploratory Data Analysis (EDA)
* XGBoost regression model
* Hyperparameter tuning:
 **GridSearchCV
 **Optuna optimization
* Evaluation using MAE (Mean Absolute Error)

---
## Project Workflow

```
Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering
   ↓
EDA & Visualization
   ↓
Train-Test Split (Time-based)
   ↓
Model Training (XGBoost)
   ↓
Hyperparameter Tuning
   ↓
Evaluation (MAE)
   ↓
Prediction & Comparison
```

---

## Dataset

* Input file: sales_data.csv
* Key columns:
** Date
** Store ID
** Product ID
** Demand
** Price
** Promotion
** Inventory Level
** Weather Condition

---

## Exploratory Data Analysis

The project performs:

* Overall demand trends over time
* Product-wise demand comparison
* Monthly demand patterns
* Promotion impact analysis
* Weather impact on demand
* Correlation heatmap

---

## Feature Engineering

* Time Features
** Day, Month, Year
** Day of Week
** Week of Year

* Lag Features
** lag_2, lag_3, lag_7, lag_14, lag_30

* Rolling Features
** rolling_mean_7
** rolling_mean_14
** rolling_std_7

---

## Model

### Base Model
* XGBoost Regressor
* Key parameters:
** n_estimators = 300
** max_depth = 8
** learning_rate = 0.05

### Evaluation Metric
* Mean Absolute Error (MAE)

---

## Hyperparameter Tuning

### GridSearchCV

* TimeSeriesSplit cross-validation
* Searches across:
* Depth
* Learning rate
* Subsample
* Estimators

### Optuna Optimization

* Advanced tuning using Bayesian optimization
* 20 trials for best parameters

---

## Results

* Base Model MAE
* Tuned Model MAE
* Optuna Improved MAE

## Visualization includes:

* Actual vs Predicted demand
* Model comparison plots
* Feature importance ranking

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna
```

---

## Usage

* Place dataset: sales_data.csv
* Run the notebook/script
* Outputs:
** Demand predictions
** Model evaluation
** Visual insights

---

## Conclusion

This project demonstrates how classical ML + feature engineering can outperform complex models when applied correctly to time series forecasting.

---

## Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---
