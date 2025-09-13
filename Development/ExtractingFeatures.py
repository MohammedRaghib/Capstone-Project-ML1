import pandas as pd
import os

# Load the feature importances data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../Data/Cleaned/feature_importances.csv")
df = pd.read_csv(file_path)

# Extract directors
directors_df = df[df['feature'].str.startswith('director_')].copy()
directors_df['name'] = directors_df['feature'].str.replace('director_', '')

# Extract cast members
cast_df = df[df['feature'].str.startswith('cast_')].copy()
cast_df['name'] = cast_df['feature'].str.replace('cast_', '')

# Extract genres
genre_df = df[df['feature'].str.startswith('genre_')].copy()
genre_df['name'] = genre_df['feature'].str.replace('genre_', '')

# Extract writers
writer_df = df[df['feature'].str.startswith('writer_')].copy()
writer_df['name'] = writer_df['feature'].str.replace('writer_', '')

# Save the names to separate text files
directors_df['name'].to_csv('directors.txt', index=False, header=False)
cast_df['name'].to_csv('cast.txt', index=False, header=False)
genre_df['name'].to_csv('genres.txt', index=False, header=False)
writer_df['name'].to_csv('writers.txt', index=False, header=False)

print("Names saved to text files.")
