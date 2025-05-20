import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import gc  # Garbage collection
import os
import logging
from typing import List, Dict, Optional

# Setup logger
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF on movie genres and titles.
    """

    def __init__(self):
        """Initialize the recommender."""
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.movie_indices = None

    def fit(self, movies_df: pd.DataFrame):
        """
        Fit the recommender with movie data. Builds the TF-IDF matrix.

        Args:
            movies_df: DataFrame containing movie data ('movieId', 'title', 'genres').
        """
        if movies_df is None or movies_df.empty:
            logger.error("Cannot fit recommender: movies_df is empty or None.")
            return

        logger.info(f"Fitting ContentBasedRecommender with {len(movies_df)} movies.")
        self.movies_df = movies_df.copy()

        # Ensure required columns exist
        required_cols = ['movieId', 'title', 'genres']
        if not all(col in self.movies_df.columns for col in required_cols):
            logger.error(f"Movies DataFrame must contain columns: {required_cols}")
            self.movies_df = None # Invalidate state
            return

        # Preprocess data
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        # Combine title and genres for TF-IDF
        # Replace '|' with space in genres for better tokenization
        self.movies_df['content'] = self.movies_df['title'] + ' ' + self.movies_df['genres'].str.replace('|', ' ', regex=False)

        # Build TF-IDF matrix
        try:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            # Handle potential NaN values in 'content' column just in case
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['content'].fillna(''))
            logger.info("TF-IDF matrix built successfully.")

            # Create a mapping from title to index for quick lookup
            # Use title as index, assuming titles are unique enough for this purpose
            # If titles are not unique, this might map to the first occurrence.
            # Consider using movieId if titles can be duplicates.
            self.movies_df = self.movies_df.reset_index(drop=True) # Ensure index is 0-based sequential
            self.movie_indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])

        except Exception as e:
            logger.error(f"Error building TF-IDF matrix: {e}")
            # Reset state if fitting fails
            self.movies_df = None
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            self.movie_indices = None

    def get_recommendations(self, movie_title: str, top_n: int = 10) -> List[Dict]:
        """
        Get movie recommendations based on content similarity.

        Args:
            movie_title: The title of the movie to get recommendations for.
            top_n: The number of recommendations to return.

        Returns:
            A list of recommended movie dictionaries, each containing
            'movieId', 'title', 'genres', and 'similarity_score'.
            Returns an empty list if the movie title is not found or the model is not fitted.
        """
        if self.tfidf_matrix is None or self.movie_indices is None or self.movies_df is None:
            logger.warning("Recommender not fitted. Call fit() first.")
            return []

        # Find the index of the movie
        if movie_title not in self.movie_indices:
            # Try a case-insensitive partial match if exact title not found
            matches = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False, na=False)]
            if matches.empty:
                logger.warning(f"Movie title '{movie_title}' not found in the dataset.")
                return []
            # Use the index of the first match
            movie_idx = matches.index[0]
            actual_title = self.movies_df.loc[movie_idx, 'title']
            logger.info(f"Exact title '{movie_title}' not found. Using first match: '{actual_title}'")
        else:
            # Get the index from the precomputed series
            movie_idx = self.movie_indices[movie_title]
            # Handle potential duplicates if multiple movies have the same title
            if isinstance(movie_idx, pd.Series):
                 movie_idx = movie_idx.iloc[0] # Take the first one

        if movie_idx >= self.tfidf_matrix.shape[0]:
             logger.error(f"Movie index {movie_idx} out of bounds for TF-IDF matrix.")
             return []


        # Calculate cosine similarity between the input movie and all others
        # linear_kernel is usually faster for TF-IDF
        cosine_sim = linear_kernel(self.tfidf_matrix[movie_idx:movie_idx+1], self.tfidf_matrix).flatten()

        # Get similarity scores for all movies, sorted
        # enumerate adds index, sort by score (x[1]), descending
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)

        # Get scores of the top_n most similar movies (excluding the movie itself, index 0 is the movie itself)
        sim_scores = sim_scores[1:top_n + 1]

        # Get the movie indices from the similarity scores
        movie_indices = [i[0] for i in sim_scores]

        # Get the corresponding movie details
        recommendations_df = self.movies_df.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
        recommendations_df['similarity_score'] = [i[1] for i in sim_scores]

        return recommendations_df.to_dict('records')


# Example usage (optional, for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing ContentBasedRecommender...")

    # Create dummy data for testing
    data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Movie A (2020)', 'Movie B (2021)', 'Movie C (2020)', 'Movie D (2022)', 'Movie A (2019)'], # Note duplicate title base
        'genres': ['Action|Adventure', 'Comedy', 'Action|Thriller', 'Comedy|Romance', 'Action|Sci-Fi']
    }
    dummy_movies_df = pd.DataFrame(data)

    # Initialize and fit recommender
    recommender = ContentBasedRecommender()
    recommender.fit(dummy_movies_df)

    # Get recommendations
    if recommender.tfidf_matrix is not None:
        recommendations = recommender.get_recommendations('Movie A (2020)', top_n=2)
        logger.info(f"Recommendations for 'Movie A (2020)':\n{recommendations}")

        recommendations_partial = recommender.get_recommendations('Movie B', top_n=2)
        logger.info(f"Recommendations for 'Movie B':\n{recommendations_partial}")

        recommendations_not_found = recommender.get_recommendations('Unknown Movie', top_n=2)
        logger.info(f"Recommendations for 'Unknown Movie':\n{recommendations_not_found}")
    else:
        logger.error("Recommender fitting failed.") 