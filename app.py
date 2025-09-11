from flask import Flask, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/lgbm_movies_enriched.pkl")
all_columns = joblib.load("models/feature_columns.pkl")

@app.route("/")
def home():
    return "Movie Rating Prediction API is running!"

@app.route("/predict")
def predict():
    input_dict = {col: 0 for col in all_columns}

    input_dict["budget"] = 165000000.0
    input_dict["popularity"] = 150.3
    input_dict["runtime"] = 148.0
    input_dict["vote_count"] = 20000
    input_dict["revenue"] = 830000000.0
    input_dict["release_year"] = 2010.0

    if "genre_Action" in input_dict:
        input_dict["genre_Action"] = 1
    if "director_Christopher Nolan" in input_dict:
        input_dict["director_Christopher Nolan"] = 1
    if "cast_Leonardo DiCaprio" in input_dict:
        input_dict["cast_Leonardo DiCaprio"] = 1

    X = pd.DataFrame([input_dict], columns=all_columns)

    prediction = model.predict(X)[0]

    return jsonify({"predicted_vote_average": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
