import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "../Data/Cleaned/movies_enriched.csv"
OUT_DIR = "../Data/Visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Loaded enriched dataset:", df.shape)

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
plt.close()

pairplot_cols = [
    "vote_average", "budget", "revenue", "runtime",
    "profit", "roi", "popularity"
]
sns.pairplot(df[pairplot_cols], diag_kind="kde", plot_kws={"alpha": 0.3})
plt.savefig(os.path.join(OUT_DIR, "pairplot.png"))
plt.close()

dist_cols = ["vote_average", "budget", "revenue", "runtime", "profit", "roi"]
for col in dist_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=40, kde=True)
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"dist_{col}.png"))
    plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x="release_decade", y="vote_average", data=df)
plt.title("Movie Ratings by Release Decade", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ratings_by_decade.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x="release_decade", y="budget", data=df)
plt.title("Budgets by Release Decade", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "budgets_by_decade.png"))
plt.close()

scatter_pairs = [
    ("budget", "vote_average"),
    ("revenue", "vote_average"),
    ("popularity", "vote_average"),
    ("profit", "vote_average"),
]
for x, y in scatter_pairs:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x, y=y, data=df, alpha=0.4)
    plt.title(f"{x} vs {y}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"scatter_{x}_vs_{y}.png"))
    plt.close()

genre_cols = [c for c in df.columns if c.startswith("genre_")]
mean_ratings = {g: df.loc[df[g] == 1, "vote_average"].mean() for g in genre_cols}
genre_df = pd.Series(mean_ratings).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_df.index.str.replace("genre_", ""), y=genre_df.values)
plt.title("Average Rating by Genre", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "avg_rating_by_genre.png"))
plt.close()

print(f"✅ Visualizations saved to {OUT_DIR}")
