from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("models/lgbm_movies_enriched.pkl")
    all_columns = joblib.load("models/feature_columns.pkl")
except FileNotFoundError as e:
    print(f"Error: Model files not found. Please ensure 'lgbm_movies_enriched.pkl' and 'feature_columns.pkl' are in the 'models' directory.")
    print(e)
    model = None
    all_columns = None

def load_names_from_file(filename):
    """Loads a list of names from a text file."""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
        return []

@app.route("/")
def home():
    directors = load_names_from_file("directors.txt")
    cast = load_names_from_file("cast.txt")
    genres = load_names_from_file("genres.txt")
    writers = load_names_from_file("writers.txt")
    return render_template('Predict.html', directors=directors, cast=cast, genres=genres, writers=writers)

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not all_columns:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.json

    input_dict = {col: 0 for col in all_columns}

    try:
        budget = float(data.get("budget", 0))
        popularity = float(data.get("popularity", 0))
        runtime = float(data.get("runtime", 0))
        vote_count = float(data.get("vote_count", 0))
        revenue = float(data.get("revenue", 0))
        release_year = float(data.get("release_year", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input for numerical fields"}), 400

    input_dict["budget"] = budget
    input_dict["popularity"] = popularity
    input_dict["runtime"] = runtime
    input_dict["vote_count"] = vote_count
    input_dict["revenue"] = revenue
    input_dict["release_year"] = release_year

    input_dict["profit"] = revenue - budget
    input_dict["roi"] = (revenue / budget) if budget > 0 else 0
    input_dict["profit_margin"] = (revenue - budget) / revenue if revenue > 0 else 0
    input_dict["log_budget"] = np.log1p(budget)
    input_dict["log_revenue"] = np.log1p(revenue)
    input_dict["log_vote_count"] = np.log1p(vote_count)
    input_dict["log_profit"] = np.log1p(revenue - budget) if revenue > budget else 0
    input_dict["log_roi"] = np.log1p(input_dict["roi"]) if input_dict["roi"] > 0 else 0
    input_dict["revenue_per_vote"] = revenue / vote_count if vote_count > 0 else 0
    input_dict["popularity_per_vote"] = popularity / vote_count if vote_count > 0 else 0

    if budget < 0 or revenue < 0 or runtime < 0:
        return jsonify({"error": "Budget, revenue, and runtime must be non-negative"}), 400

    selected_genre = f"genre_{data.get('genre')}"
    if selected_genre in input_dict:
        input_dict[selected_genre] = 1

    X = pd.DataFrame([input_dict], columns=all_columns)
    prediction = model.predict(X)[0]

    return jsonify({"predicted_vote_average": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
