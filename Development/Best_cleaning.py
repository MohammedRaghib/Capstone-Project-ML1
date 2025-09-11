import pandas as pd
import numpy as np
import ast
from textblob import TextBlob

movies = pd.read_csv('../Data/Uncleaned/tmdb_5000_movies.csv')
credits = pd.read_csv('../Data/Uncleaned/tmdb_5000_credits.csv')

print("Original Movies shape:", movies.shape)
print("Original Credits shape:", credits.shape)

df = movies.merge(credits, left_on='id', right_on='movie_id', how='inner')
print("Merged dataset shape:", df.shape)

columns_to_drop = [
    'homepage', 'original_language', 'original_title',
    'status', 'title_y'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
df.rename(columns={'title_x': 'title'}, inplace=True)

df['runtime'].fillna(df['runtime'].median(), inplace=True)
df.dropna(subset=['release_date', 'vote_average'], inplace=True)

df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['release_decade'] = (df['release_year'] // 10) * 10

df['budget'] = df['budget'].replace(0, np.nan)
df['budget'].fillna(df['budget'].median(), inplace=True)

df['revenue'] = df['revenue'].replace(0, np.nan)
df['revenue'].fillna(df['revenue'].median(), inplace=True)

df['profit'] = df['revenue'] - df['budget']
df['roi'] = df['revenue'] / df['budget']
df['profit_margin'] = df['profit'] / (df['budget'] + 1)

for col in ['budget', 'revenue', 'profit', 'roi', 'vote_count']:
    df[f'log_{col}'] = np.log1p(df[col])

def extract_genres(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)]
    except:
        return ['Unknown']

df['genres'] = df['genres'].apply(extract_genres)
df['genre_count'] = df['genres'].apply(len)

all_genres = [g for sub in df['genres'] for g in sub]
top_genres = pd.Series(all_genres).value_counts().head(20).index.tolist()
for genre in top_genres:
    df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)

def get_director(crew_data):
    try:
        crew_list = ast.literal_eval(crew_data)
        for person in crew_list:
            if person.get('job') == 'Director':
                return person.get('name')
        return 'Unknown'
    except:
        return 'Unknown'

def get_writer(crew_data):
    try:
        crew_list = ast.literal_eval(crew_data)
        writers = [p['name'] for p in crew_list if p.get('department') == 'Writing']
        return writers[0] if writers else 'Unknown'
    except:
        return 'Unknown'

df['director'] = df['crew'].apply(get_director)
df['writer'] = df['crew'].apply(get_writer)

top_directors = df['director'].value_counts().head(50).index.tolist()
top_writers = df['writer'].value_counts().head(50).index.tolist()

for d in top_directors:
    df[f'director_{d}'] = (df['director'] == d).astype(int)

for w in top_writers:
    df[f'writer_{w}'] = (df['writer'] == w).astype(int)

def get_top_cast(cast_data, top_n=3):
    try:
        cast_list = ast.literal_eval(cast_data)
        return [c['name'] for c in cast_list[:top_n]]
    except:
        return []

df['top_cast'] = df['cast'].apply(get_top_cast)
all_cast = [c for sub in df['top_cast'] for c in sub]
top_cast = pd.Series(all_cast).value_counts().head(100).index.tolist()

for actor in top_cast:
    df[f'cast_{actor}'] = df['top_cast'].apply(lambda x: 1 if actor in x else 0)

df['cast_count'] = df['top_cast'].apply(len)

def text_features(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return 0, 0
    blob = TextBlob(text)
    return len(text.split()), blob.sentiment.polarity

df['overview_length'], df['overview_sentiment'] = zip(*df['overview'].map(text_features))
df['tagline_length'], df['tagline_sentiment'] = zip(*df['tagline'].map(text_features))

df['popularity_per_vote'] = df['popularity'] / (df['vote_count'] + 1)
df['revenue_per_vote'] = df['revenue'] / (df['vote_count'] + 1)

features = [
    'budget', 'popularity', 'runtime', 'vote_count', 'revenue',
    'release_year', 'release_decade', 'profit', 'roi', 'profit_margin',
    'log_budget', 'log_revenue', 'log_profit', 'log_roi', 'log_vote_count',
    'genre_count', 'cast_count', 'overview_length', 'overview_sentiment',
    'tagline_length', 'tagline_sentiment',
    'popularity_per_vote', 'revenue_per_vote'
] + [f'genre_{g}' for g in top_genres] \
  + [f'director_{d}' for d in top_directors] \
  + [f'writer_{w}' for w in top_writers] \
  + [f'cast_{c}' for c in top_cast]

cleaned = df[features + ['vote_average']]
print("\nFinal cleaned dataset shape:", cleaned.shape)

cleaned.to_csv('./Data/Cleaned/movies_enriched.csv', index=False)
