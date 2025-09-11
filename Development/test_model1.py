import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../Data/Cleaned/movies_enriched.csv")
df = pd.read_csv(file_path)

if 'vote_average' not in df.columns:
    raise ValueError("Expected column 'vote_average' in CSV")

y = df['vote_average'].copy()
X = df.drop(columns=['vote_average'])

binary_prefixes = ('genre_', 'director_', 'writer_', 'cast_')
binary_cols = [c for c in X.columns if c.startswith(binary_prefixes)]
numeric_cols = [c for c in X.columns if c not in binary_cols]

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
print(f"Binary features (count={len(binary_cols)}): sample {binary_cols[:8]}")

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[binary_cols] = X[binary_cols].fillna(0)

scaler = RobustScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

joblib.dump(scaler, "C:/Users/LENOVO/Documents/Projects/Capstone-Project-ML1/models/robustscaler_extratrees.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 7],     
    'min_samples_split': [20, 30],
    'min_samples_leaf': [15, 20] 
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

print("Best Hyperparameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print("\nTrain Performance:")
print(f"MAE: {train_mae:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"R²: {train_r2:.2f}")

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Performance:")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R²: {test_r2:.2f}")

joblib.dump(best_rf, "C:/Users/LENOVO/Documents/Projects/Capstone-Project-ML1/models/extratrees_movies_enriched.pkl")
print("Saved ExtraTrees model to C:/Users/LENOVO/Documents/Projects/Capstone-Project-ML1/models/extratrees_movies_enriched.pkl")
