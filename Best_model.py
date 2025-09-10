import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import early_stopping, log_evaluation

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception as e:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm") from e

CSV_PATH = "./Data/Cleaned/movies_enriched.csv"  
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.2   
N_TRIALS = 40    
RANDOMIZED_ITERS = 50  
MODEL_OUTPUT = "./models/lgbm_movies_enriched.pkl"
IMPORTANCE_OUTPUT = "./models/feature_importances.csv"
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Loaded dataframe shape:", df.shape)

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

joblib.dump(scaler, os.path.join(os.path.dirname(MODEL_OUTPUT), "robustscaler.pkl"))

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VALID_SIZE, random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape, "Test shape:", X_test.shape)

def objective_optuna(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, **param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50), log_evaluation(50)],
    )

    preds = model.predict(X_val)
    return r2_score(y_val, preds)

if OPTUNA_AVAILABLE:
    print("Running Optuna tuning...")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_optuna, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    print("Optuna best params:", best_params)
    best_params.setdefault('n_estimators', 1000)
    best_params.setdefault('learning_rate', 0.01)

else:
    print("Optuna not available — using RandomizedSearchCV fallback.")
    param_dist = {
        'n_estimators': [200, 400, 600, 800, 1000],
        'num_leaves': [31, 50, 80, 120],
        'max_depth': [5, 8, 12, 16],
        'learning_rate': [0.01, 0.02, 0.05],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'reg_alpha': [0.0, 0.1, 1.0],
        'reg_lambda': [0.0, 0.1, 1.0],
        'min_child_samples': [5, 10, 20, 50]
    }

    lgbm = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=RANDOMIZED_ITERS,
        scoring='r2',
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print("RandomizedSearchCV best params:", best_params)

print("Training final model with best params...")

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1, random_state=RANDOM_STATE
)

final_model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)

final_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val_sub, y_val_sub)],
    eval_metric='rmse',
    callbacks=[early_stopping(50), log_evaluation(50)]
)

y_train_pred = final_model.predict(X_train_full)
y_test_pred = final_model.predict(X_test)

print("\nTrain Performance:")
print(f"MAE: {mean_absolute_error(y_train_full, y_train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train_full, y_train_pred)):.4f}")
print(f"R²: {r2_score(y_train_full, y_train_pred):.4f}")

print("\nTest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_test_pred):.4f}")

fi = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
fi.to_csv(IMPORTANCE_OUTPUT, index=False)
print(f"Saved feature importances to {IMPORTANCE_OUTPUT}")

joblib.dump(final_model, MODEL_OUTPUT)
print(f"Saved model to {MODEL_OUTPUT}")

print("\nTop 20 features by importance:")
print(fi.head(20).to_string(index=False))
