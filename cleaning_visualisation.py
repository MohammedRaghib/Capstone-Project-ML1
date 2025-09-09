import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

movies = pd.read_csv('./Data/Uncleaned/tmdb_5000_movies.csv')
credits = pd.read_csv('./Data/Uncleaned/tmdb_5000_credits.csv')

print("Original Movies shape:", movies.shape)
print("Original Credits shape:", credits.shape)

df = movies.merge(credits, left_on='id', right_on='movie_id', how='inner')
print("Merged dataset shape:", df.shape)

columns_to_drop = [
    'homepage', 'original_language', 'original_title', 'overview',
    'tagline', 'status', 'keywords', 'production_companies',
    'production_countries', 'spoken_languages', 'title_y'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

df.rename(columns={'title_x': 'title'}, inplace=True)

# Handle missing runtime
df['runtime'].fillna(df['runtime'].median(), inplace=True)
df.dropna(subset=['release_date', 'vote_average'], inplace=True)

# Extract release year
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Fill missing/zero budgets with median
df['budget'] = df['budget'].replace(0, np.nan)
df['budget'].fillna(df['budget'].median(), inplace=True)

# Fill missing/zero revenues with median
df['revenue'] = df['revenue'].replace(0, np.nan)
df['revenue'].fillna(df['revenue'].median(), inplace=True)

# Extract genres
def extract_genres(x):
    try:
        genres = [i['name'] for i in ast.literal_eval(x)]
        return genres if genres else ['Unknown']
    except:
        return ['Unknown']

df['genres'] = df['genres'].apply(extract_genres)

# Extract director
def get_director(crew_data):
    try:
        crew_list = ast.literal_eval(crew_data)
        for person in crew_list:
            if person.get('job') == 'Director':
                return person.get('name')
        return 'Unknown'
    except:
        return 'Unknown'

df['director'] = df['crew'].apply(get_director)

features = [
    'budget', 'popularity', 'runtime', 'vote_count', 'revenue',
    'release_year', 'genres', 'director', 'vote_average'
]
df = df[features]

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDataset Summary:")
print(df.describe())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['vote_average'], bins=20, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='budget', y='vote_average', data=df, alpha=0.5)
plt.title("Budget vs Rating")
plt.xlabel("Budget")
plt.ylabel("Rating")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='popularity', y='vote_average', data=df, alpha=0.5)
plt.title("Popularity vs Rating")
plt.xlabel("Popularity")
plt.ylabel("Rating")
plt.show()

print("\nCleaned dataset shape:", df.shape)

df.to_csv('./Data/Cleaned/movies_cleaned.csv', index=False)