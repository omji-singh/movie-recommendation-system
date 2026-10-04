import os, zipfile, urllib.request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================

#         DATA INGESTION & SETUP

# ==========================================

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
if not os.path.exists("ml-latest-small"):

    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(URL, "ml.zip")
    with zipfile.ZipFile("ml.zip") as z:

        z.extractall(".")
movies = pd.read_csv("ml-latest-small/movies.csv")

ratings = pd.read_csv("ml-latest-small/ratings.csv")
print(f"Dataset loaded: {len(movies)} movies, {len(ratings)} ratings, {ratings.userId.nunique()} users")


# ==========================================

#    CONTENT-BASED PREPARATION (TF-IDF)

# ==========================================

# Format genres by replacing the '|' separator with spaces

movies_content = movies.copy()
movies_content['genre_features'] = movies_content['genres'].str.replace('|', ' ', regex=False).str.lower()

# Convert text genres into a mathematical matrix

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_content['genre_features'])
content_similarity_matrix = cosine_similarity(tfidf_matrix)


# ==========================================

#    COLLABORATIVE FILTERING PREPARATION

# ==========================================

# Filter out movies with too few ratings to avoid skewed data

rating_counts = ratings.groupby('movieId').size()
popular_movies = rating_counts[rating_counts >= 20].index
filtered_ratings = ratings[ratings.movieId.isin(popular_movies)]

# Create a User-Item Matrix

user_item_matrix = filtered_ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Mean-center the ratings to remove user bias (generous vs. harsh raters)

centered_matrix = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0).fillna(0)

# Calculate similarity between movies based on user rating patterns

item_similarity_matrix = pd.DataFrame(
    cosine_similarity(centered_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# Helper dictionaries for fast ID/Title lookups

id_to_title = movies.set_index('movieId')['title']
title_to_id = movies.set_index('title')['movieId']


# ==========================================

#      CORE RECOMMENDATION ENGINE

# ==========================================

def find_title_index(query):
    """Fuzzy search to find a movie title index based on a partial text query."""
    words = query.lower().split()
    mask = movies_content['title'].str.lower().apply(lambda t: all(w in t for w in words))
    hits = movies_content[mask]
    if hits.empty:
        raise ValueError(f"No title matching '{query}' found.")
    return hits.index[0]

def recommend_similar_movies(movie_query, n=10, method='content'):
    """Recommends movies using either Content-Based (genres) or Collaborative (ratings) filtering."""
    idx = find_title_index(movie_query)
    resolved_title = movies_content.loc[idx, 'title']
    print(f"Because you liked: {resolved_title} (Method: {method.title()})\n")
  
    if method == 'content':
        scores = list(enumerate(content_similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        results = movies_content.iloc[[i for i, _ in scores]][['title', 'genres']].copy()
        results['similarity_score'] = [round(s, 3) for _, s in scores]

    elif method == 'collab':
        movie_id = title_to_id[resolved_title]
        if movie_id not in item_similarity_matrix.index:
            raise ValueError(f"'{resolved_title}' has too few ratings for collaborative filtering.")

        sims = item_similarity_matrix[movie_id].drop(movie_id).sort_values(ascending=False).head(n)
        results = pd.DataFrame({
            'title': [id_to_title[i] for i in sims.index],
            'similarity_score': sims.round(3).values
        })

    return results.reset_index(drop=True)

def predict_user_preferences(user_id, n=10):
    """Predicts top movies for a specific user based on their historical ratings."""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"Unknown user ID: {user_id}")
    user_ratings = user_item_matrix.loc[user_id].dropna()
    seen_movies = set(user_ratings.index)

    # Calculate weighted average of similarities
  
    sims = item_similarity_matrix.loc[list(seen_movies)]
    weights = user_ratings.values.reshape(-1, 1)
    numerator = (sims.values * weights).sum(axis=0)
    denominator = np.abs(sims.values).sum(axis=0)

    # Avoid division by zero
  
    predicted_scores = pd.Series(
        numerator / np.where(denominator == 0, 1e-9, denominator),
        index=item_similarity_matrix.columns
    )

    # Remove movies the user has already seen

    predicted_scores = predicted_scores.drop(index=list(seen_movies), errors='ignore')
    top_predictions = predicted_scores.sort_values(ascending=False).head(n)
    return pd.DataFrame({
        'title': [id_to_title[i] for i in top_predictions.index],
        'predicted_rating': top_predictions.round(2).values
    })


# ==========================================

#          EXECUTION & TESTING

# ==========================================

print(recommend_similar_movies("3 idiots", method='content'))
