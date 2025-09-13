from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model and feature columns
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

    # Add numerical features and convert them to float
    try:
        input_dict["budget"] = float(data.get("budget", 0))
        input_dict["popularity"] = float(data.get("popularity", 0))
        input_dict["runtime"] = float(data.get("runtime", 0))
        input_dict["vote_count"] = float(data.get("vote_count", 0))
        input_dict["revenue"] = float(data.get("revenue", 0))
        input_dict["release_year"] = float(data.get("release_year", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input for numerical fields. Please ensure they are numbers."}), 400

    # Add categorical features (one-hot encoded)
    selected_genre = f"genre_{data.get('genre')}"
    if selected_genre in input_dict:
        input_dict[selected_genre] = 1

    selected_director = f"director_{data.get('director')}"
    if selected_director in input_dict:
        input_dict[selected_director] = 1

    selected_cast = f"cast_{data.get('cast')}"
    if selected_cast in input_dict:
        input_dict[selected_cast] = 1
        
    selected_writer = f"writer_{data.get('writer')}"
    if selected_writer in input_dict:
        input_dict[selected_writer] = 1

    # Create DataFrame and predict
    X = pd.DataFrame([input_dict], columns=all_columns)
    prediction = model.predict(X)[0]

    return jsonify({"predicted_vote_average": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
