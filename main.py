
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('sales_data.csv')

df.info()

df.isnull().sum()

df['Date']= pd.to_datetime(df['Date'])

df['day'] = df['Date'].dt.day
df['month'] =  df['Date'].dt.month
df['year'] =  df['Date'].dt.year

df = df.set_index('Date')

df = df.sort_index()

for col in df.columns:
  print(f' column: {col}, unique values: {df[col].nunique()}\n')

df.groupby(['Store ID', 'Product ID']).size()

def check_gaps(group):
  return group.index.inferred_freq

df.groupby(['Store ID', 'Product ID']).apply(check_gaps)

df.groupby(['Store ID', 'Product ID']).size().describe()

overall = df.groupby('Date')['Demand'].sum()

plt.figure(figsize=(12,5))
overall.plot()
plt.title("Overall Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.show()

product_demand = df.groupby('Product ID')['Demand'].sum().sort_values(ascending=False)

product_demand.plot(kind='bar', figsize=(12,5))
plt.title("Overall Demand by Product")
plt.ylabel("Total Demand")
plt.show()

monthly_pattern = df.groupby('month')['Demand'].mean()

monthly_pattern.plot(kind='bar', figsize=(10,5))
plt.title("Average Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Demand")
plt.show()

df.groupby('Promotion')['Demand'].mean().plot(kind='bar', title="Promotion Impact")
plt.ylabel("Average Demand")
plt.show()

plt.scatter(df['Price'],df['Demand'])
plt.xlabel("Price")
plt.ylabel("Demand")
plt.title("Price vs Demand")
plt.show()

plt.scatter(df['Inventory Level'], df['Demand'])
plt.xlabel("Inventory")
plt.ylabel("Demand")
plt.title("Inventory vs Demand")
plt.show()

df.groupby('Weather Condition')['Demand'].mean().plot(kind='bar', title="Weather Impact")
plt.show()

corr = df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(10,5))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

df['lag_2']  = df.groupby(['Store ID','Product ID'])['Demand'].shift(2)
df['lag_3']  = df.groupby(['Store ID','Product ID'])['Demand'].shift(3)
df['lag_7']  = df.groupby(['Store ID','Product ID'])['Demand'].shift(7)
df['lag_14'] = df.groupby(['Store ID','Product ID'])['Demand'].shift(14)
df['lag_30'] = df.groupby(['Store ID','Product ID'])['Demand'].shift(30)

df['price_promo'] = df['Price'] * df['Promotion']

df['day_of_week'] = df.index.dayofweek
df['week_of_year'] = df.index.isocalendar().week

df['rolling_mean_7'] = df.groupby(['Store ID','Product ID'])['Demand'].transform(lambda x: x.rolling(7).mean())
df['rolling_mean_14'] = df.groupby(['Store ID','Product ID'])['Demand'].transform(lambda x: x.rolling(14).mean())
df['rolling_std_7'] = df.groupby(['Store ID','Product ID'])['Demand'].transform(lambda x: x.rolling(7).std())

df = df.dropna()

df = df.sort_values(['Store ID', 'Product ID', 'Date'])

train_list = []
test_list = []

for _, group in df.groupby(['Store ID', 'Product ID']):

  split = int(len(group)* 0.8)

  train_list.append(group.iloc[:split])
  test_list.append(group.iloc[split:])

train_df = pd.concat(train_list).reset_index()
test_df  = pd.concat(test_list).reset_index()

X_train = train_df.drop(columns=['Demand','Date','Units Sold', 'Units Ordered'])
y_train = train_df['Demand']

X_test = test_df.drop(columns=['Demand','Date','Units Sold', 'Units Ordered'])
y_test = test_df['Demand']

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f'MAE: {mae}')

plt.figure(figsize=(12,5))
plt.plot(y_test.values[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Demand")
plt.show()

importance = pd.Series(model.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)

print(importance.head(15))

importance.head(10).plot(kind='barh', title="Top Features")
plt.gca().invert_yaxis()
plt.show()

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

model = XGBRegressor(random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("Tuned MAE:", mae)

pip install optuna

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    model = XGBRegressor(**params)

    tscv = TimeSeriesSplit(n_splits=3)

    maes = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)

    return sum(maes) / len(maes)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(study.best_params)

best_model = XGBRegressor(**study.best_params)

best_model.fit(X_train, y_train)

y_pred_optuna = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_optuna)
print("Optuna MAE:", mae)

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'XGBoost_Base': y_pred,
    'XGBoost_Optuna': y_pred_optuna
})

import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

plt.plot(results_df['Actual'][:200], label='Actual', linewidth=2)
plt.plot(results_df['XGBoost_Base'][:200], label='XGB Base', alpha=0.7)
plt.plot(results_df['XGBoost_Optuna'][:200], label='XGB Optuna', alpha=0.7)

plt.legend()
plt.title("Model Comparison (First 200 Points)")
plt.show()

from sklearn.metrics import mean_absolute_error

mae_base = mean_absolute_error(y_test, y_pred)
mae_optuna = mean_absolute_error(y_test, y_pred_optuna)

print("XGB Base MAE:", mae_base)
print("XGB Optuna MAE:", mae_optuna)
