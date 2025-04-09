"""
Enhanced content-based filtering using sentence-transformers.
"""
import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-based recommender using Sentence Transformers for embedding generation.
    
    This recommender creates embeddings from movie metadata (genre, title, overview)
    and recommends similar movies based on cosine similarity.
    """
    
    def __init__(self, transformer_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the content-based recommender.
        
        Args:
            transformer_model: Name or path of the sentence transformer model.
        """
        self.transformer_model = transformer_model
        self.model = None
        self.movies_df = None
        self.movie_embeddings = None
        self.id_to_index = None
        self.index_to_id = None
    
    def fit(self, movies_df: pd.DataFrame):
        """
        Fit the model with movie data.
        
        Args:
            movies_df: DataFrame containing movie data.
        """
        logger.info("Fitting content-based recommender...")
        self.movies_df = movies_df.copy()
        
        # Create a mapping from movie ID to index and vice versa
        self.id_to_index = {id: idx for idx, id in enumerate(self.movies_df['movieId'])}
        self.index_to_id = {idx: id for id, idx in self.id_to_index.items()}
        
        # Load sentence transformer model
        logger.info(f"Loading transformer model: {self.transformer_model}")
        self.model = SentenceTransformer(self.transformer_model)
        
        # Create content features for embedding
        logger.info("Creating content features...")
        content_features = self._create_content_features()
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        self.movie_embeddings = self._generate_embeddings(content_features)
        
        logger.info("Content-based recommender fitted successfully")
        return self
    
    def _create_content_features(self) -> List[str]:
        """
        Create content features from movie metadata.
        
        Returns:
            List of content feature strings.
        """
        # Check required columns
        for col in ['genres', 'title', 'clean_title']:
            if col not in self.movies_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        content_features = []
        
        # Use clean_title, genres and overview (if available) to create content features
        for _, row in self.movies_df.iterrows():
            feature = f"{row['clean_title']}. "
            
            # Add genres
            feature += f"Genres: {row['genres'].replace('|', ', ')}. "
            
            # Add overview if available
            if 'overview' in self.movies_df.columns and row['overview']:
                feature += f"Overview: {row['overview']}"
            
            content_features.append(feature)
        
        return content_features
    
    def _generate_embeddings(self, content_features: List[str]) -> np.ndarray:
        """
        Generate embeddings for content features.
        
        Args:
            content_features: List of content feature strings.
            
        Returns:
            Array of embeddings.
        """
        # Generate embeddings in batches to avoid memory issues
        batch_size = 128
        embeddings = []
        
        for i in range(0, len(content_features), batch_size):
            batch = content_features[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar movies for a given movie ID.
        
        Args:
            movie_id: Movie ID.
            top_n: Number of similar movies to return.
            
        Returns:
            List of tuples (movie_id, similarity_score).
        """
        if self.movie_embeddings is None:
            raise ValueError("Model not fitted yet")
        
        # Get index for movie_id
        if movie_id not in self.id_to_index:
            raise ValueError(f"Movie ID {movie_id} not found")
        
        movie_idx = self.id_to_index[movie_id]
        
        # Get embedding for the movie
        movie_embedding = self.movie_embeddings[movie_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all other movies
        similarity_scores = cosine_similarity(movie_embedding, self.movie_embeddings).flatten()
        
        # Get indices of top similar movies (excluding the movie itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        # Return movie IDs and similarity scores
        return [(self.index_to_id[idx], similarity_scores[idx]) for idx in similar_indices]
    
    def get_recommendations(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """
        Get movie recommendations with details.
        
        Args:
            movie_id: Movie ID.
            top_n: Number of recommendations.
            
        Returns:
            List of dictionaries with movie details and similarity scores.
        """
        similar_movies = self.get_similar_movies(movie_id, top_n)
        
        recommendations = []
        for similar_id, score in similar_movies:
            movie = self.movies_df[self.movies_df['movieId'] == similar_id].iloc[0]
            recommendation = {
                'movie': {
                    'movieId': movie.movieId,
                    'title': movie.title,
                    'genres': movie.genres
                },
                'score': score,
                'reason': self._generate_recommendation_reason(movie_id, similar_id, score)
            }
            
            # Add additional fields if available
            for field in ['year', 'overview', 'poster_url']:
                if field in movie and movie[field]:
                    recommendation['movie'][field] = movie[field]
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_recommendation_reason(self, movie_id: int, similar_id: int, score: float) -> str:
        """
        Generate a reason for recommendation.
        
        Args:
            movie_id: Original movie ID.
            similar_id: Similar movie ID.
            score: Similarity score.
            
        Returns:
            Reason string.
        """
        if score > 0.9:
            strength = "very similar"
        elif score > 0.8:
            strength = "similar"
        elif score > 0.6:
            strength = "somewhat similar"
        else:
            strength = "slightly similar"
        
        # Get movie titles
        movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
        similar_movie = self.movies_df[self.movies_df['movieId'] == similar_id].iloc[0]
        
        # Check if they share genres
        movie_genres = set(movie['genres'].split('|'))
        similar_genres = set(similar_movie['genres'].split('|'))
        common_genres = movie_genres.intersection(similar_genres)
        
        if common_genres:
            genre_text = ", ".join(common_genres)
            return f"This movie is {strength} to {movie['title']} and shares these genres: {genre_text}."
        else:
            return f"This movie is {strength} to {movie['title']} based on content analysis."
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model.
        """
        data = {
            'transformer_model': self.transformer_model,
            'movies_df': self.movies_df,
            'movie_embeddings': self.movie_embeddings,
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Content-based model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ContentBasedRecommender':
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded ContentBasedRecommender instance.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(transformer_model=data['transformer_model'])
        recommender.movies_df = data['movies_df']
        recommender.movie_embeddings = data['movie_embeddings']
        recommender.id_to_index = data['id_to_index']
        recommender.index_to_id = data['index_to_id']
        
        # Load the transformer model
        recommender.model = SentenceTransformer(recommender.transformer_model)
        
        logger.info(f"Content-based model loaded from {path}")
        return recommender 