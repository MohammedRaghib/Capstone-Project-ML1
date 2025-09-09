# Movie Rating Prediction

## Problem Statement
The goal of this project is to develop a machine learning model that predicts a movie's rating based on its metadata. Many platforms and production companies use such predictions to assess the potential success of a movie before release. By analyzing features such as budget, genres, runtime, release date, and popularity, this project aims to forecast a movie's rating (e.g., vote average) and provide insights into which factors most influence audience reception.

## Overview
This project predicts movie ratings using metadata from the TMDb 5000 Movie Dataset.  
It demonstrates fundamental Machine Learning concepts including data preprocessing, feature engineering, model building, and evaluation.

## Dataset
- **Source**: [TMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Features include:
  - Budget
  - Genres
  - Runtime
  - Release date
  - Popularity
  - Vote average (target variable)

## Steps Implemented
1. Data Exploration (EDA)
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training (Linear Regression, Random Forest)
5. Model Evaluation (RMSE, MAE, R²)
6. Insights and Visualizations

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MohammedRaghib/Capstone-Project-ML1.git
   cd Capstone-Project-ML1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## License
MIT License

```
MIT License

Copyright (c) 2025 Mohammed Raghib

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Author
**Mohammed Raghib**