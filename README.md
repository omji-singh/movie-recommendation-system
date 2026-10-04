# Hybrid Movie Recommendation System

A dual-approach Movie Recommendation Engine built in Python using the **MovieLens (Small)** dataset. The system supports both **Content-Based Filtering** (using TF-IDF text vectorization) and
**Item-Based Collaborative Filtering** (using user rating patterns and mean-centered cosine similarity).

## Features

- **Automated Data Ingestion**: Downloads and extracts the dataset dynamically.
- **Content-Based Engine**: Matches movies based on genre feature vectors extracted with TF-IDF.
- **Collaborative Filtering Engine**: Recommends movies based on user rating patterns while neutralizing rater bias through mean-centering.
- **User Preference Prediction**: Calculates expected rating scores for unseen movies for a specific user ID.
- **Fuzzy Search**: Handles partial titles and MovieLens title formatting.
