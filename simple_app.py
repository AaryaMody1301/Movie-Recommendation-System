"""
Simple Movie Recommendation functions.
This module provides basic functions for the movie recommendation system.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Global variables
movies_df = None
tfidf_matrix = None
tfidf = None

def load_movies(max_movies=None):
    """Load movies from CSV file."""
    global movies_df
    try:
        df = pd.read_csv('movies.csv')
        if max_movies:
            df = df.head(max_movies)
        movies_df = df
        print(f"Loaded {len(movies_df)} movies from movies.csv")
        return df
    except Exception as e:
        print(f"Error loading movies: {e}")
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

def build_tfidf_matrix():
    """Build TF-IDF matrix for content-based recommendations."""
    global tfidf_matrix, tfidf, movies_df
    
    if movies_df is None:
        movies_df = load_movies()
    
    # Create a text representation for TF-IDF
    movies_df['text'] = movies_df['title'] + ' ' + movies_df['genres'].fillna('')
    
    if tfidf_matrix is None:
        try:
            # Create a TF-IDF Vectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(movies_df['text'])
            print("TF-IDF matrix built successfully")
        except Exception as e:
            print(f"Error building TF-IDF matrix: {e}")

def get_movie_recommendations(movie_title, n=10):
    """Get movie recommendations based on title similarity."""
    global tfidf_matrix, movies_df
    
    if movies_df is None:
        movies_df = load_movies()
    
    if tfidf_matrix is None:
        build_tfidf_matrix()
    
    # Find movies with matching title
    matches = movies_df[movies_df['title'].str.contains(movie_title, case=False)]
    
    if len(matches) == 0:
        return []
    
    # Get the first matching movie
    movie_idx = matches.index[0]
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
    
    # Get the indices of movies with highest similarity scores
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # exclude the movie itself
    
    # Return the top n most similar movies with their scores
    recommendations = []
    for idx, score in sim_scores:
        movie = movies_df.iloc[idx]
        recommendations.append({
            'movie': {
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres']
            },
            'score': float(score)
        })
    
    return recommendations

def get_unique_genres():
    """Get a list of unique movie genres from the dataset."""
    global movies_df
    
    if movies_df is None:
        movies_df = load_movies()
    
    all_genres = set()
    for genres in movies_df['genres']:
        if isinstance(genres, str):
            all_genres.update(genres.split('|'))
    
    # Remove empty strings
    if '' in all_genres:
        all_genres.remove('')
        
    # Sort alphabetically
    return sorted(list(all_genres)) 