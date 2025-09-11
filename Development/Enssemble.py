import os
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

CSV_PATH = "C:/Users/LENOVO/Documents/Projects/Capstone-Project-ML1/Data/Cleaned/movies_enriched.csv"
MODEL_OUTPUT = "C:/Users/LENOVO/Documents/Projects/Capstone-Project-ML1/models/ensemble_movies.pkl"
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

df = pd.read_csv(CSV_PATH)
if 'vote_average' not in df.columns:
    raise ValueError("Expected column 'vote_average' in CSV")

y = df['vote_average'].copy()
X = df.drop(columns=['vote_average'])

binary_prefixes = ('genre_', 'director_', 'writer_', 'cast_')
binary_cols = [c for c in X.columns if c.startswith(binary_prefixes)]
numeric_cols = [c for c in X.columns if c not in binary_cols]

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[binary_cols] = X[binary_cols].fillna(0)

scaler = RobustScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lgbm = LGBMRegressor(
    n_estimators=1200,
    num_leaves=40,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=30,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(50, verbose=False), log_evaluation(100)]
)

param_grid = {
    'n_estimators': [300],
    'max_depth': [20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5]
}

rf = ExtraTreesRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
extratrees = grid_search.best_estimator_

print("Best ExtraTrees params:", grid_search.best_params_)

weight_lgbm = 0.8
weight_rf = 0.2

y_train_pred_avg = weight_lgbm * lgbm.predict(X_train) + weight_rf * extratrees.predict(X_train)
y_test_pred_avg = weight_lgbm * lgbm.predict(X_test) + weight_rf * extratrees.predict(X_test)

print("\n=== Weighted Averaging Performance ===")
print("Train MAE:", mean_absolute_error(y_train, y_train_pred_avg))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred_avg)))
print("Train R²:", r2_score(y_train, y_train_pred_avg))

print("Test MAE:", mean_absolute_error(y_test, y_test_pred_avg))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred_avg)))
print("Test R²:", r2_score(y_test, y_test_pred_avg))

estimators = [
    ('lgbm', lgbm),
    ('extratrees', extratrees)
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)

stacking.fit(X_train, y_train)

y_train_pred_stack = stacking.predict(X_train)
y_test_pred_stack = stacking.predict(X_test)

print("\n=== Stacking Regressor Performance ===")
print("Train MAE:", mean_absolute_error(y_train, y_train_pred_stack))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred_stack)))
print("Train R²:", r2_score(y_train, y_train_pred_stack))

print("Test MAE:", mean_absolute_error(y_test, y_test_pred_stack))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred_stack)))
print("Test R²:", r2_score(y_test, y_test_pred_stack))

joblib.dump(stacking, MODEL_OUTPUT)
print(f"\n✅ Saved final ensemble model to {MODEL_OUTPUT}")
