import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import ast

df = pd.read_csv("../Data/Cleaned/movies_cleaned.csv")

df['genres'] = df['genres'].apply(ast.literal_eval)

all_genres = [genre for sublist in df['genres'] for genre in sublist]
top_genres = pd.Series(all_genres).value_counts().head(20).index.tolist()

top_directors = df['director'].value_counts().head(50).index.tolist()

numerical_features = ['budget', 'popularity', 'revenue', 'runtime', 'release_year']
categorical_features = ['genres', 'director']

for genre in top_genres:
    df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)

for director in top_directors:
    df[f'director_{director}'] = df['director'].apply(lambda x: 1 if director == x else 0)

X = df[numerical_features + [f'genre_{g}' for g in top_genres] + [f'director_{d}' for d in top_directors]]
y = df['vote_average']

X = X.fillna(X.median())
y = y.fillna(y.median())

scaler = RobustScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 7],     
    'min_samples_split': [20, 30],
    'min_samples_leaf': [15, 20] 
}

rf = ExtraTreesRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           scoring='r2', cv=3, n_jobs=-1, verbose=1)
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