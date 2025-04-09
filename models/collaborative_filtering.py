"""
Collaborative filtering using Matrix Factorization with Surprise.
"""
import os
import logging
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """
    Collaborative filtering using Matrix Factorization with Surprise.
    
    This recommender uses the SVD algorithm to factorize the user-item rating matrix
    and predict ratings for unseen items.
    """
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005, 
                 reg_all: float = 0.02, random_state: int = 42):
        """
        Initialize the collaborative recommender.
        
        Args:
            n_factors: Number of latent factors.
            n_epochs: Number of SGD epochs.
            lr_all: Learning rate for all parameters.
            reg_all: Regularization term for all parameters.
            random_state: Random seed for reproducibility.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        
        # Initialize model
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state
        )
        
        # Data containers
        self.movies_df = None
        self.trainset = None
        self.testset = None
        self.user_ids = set()
        self.movie_ids = set()
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, test_size: float = 0.2):
        """
        Fit the model with ratings data.
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating'].
            movies_df: DataFrame with movies metadata.
            test_size: Fraction of data to use for testing.
            
        Returns:
            Self instance.
        """
        logger.info("Fitting collaborative recommender...")
        self.movies_df = movies_df.copy()
        
        # Check required columns
        required_cols = ['userId', 'movieId', 'rating']
        missing_cols = [col for col in required_cols if col not in ratings_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ratings DataFrame: {missing_cols}")
        
        # Convert ratings to Surprise format
        reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
        data = Dataset.load_from_df(ratings_df[required_cols], reader)
        
        # Split into train and test sets
        self.trainset, self.testset = train_test_split(data, test_size=test_size, random_state=self.random_state)
        
        # Store unique user and movie IDs
        self.user_ids = set(ratings_df['userId'].unique())
        self.movie_ids = set(ratings_df['movieId'].unique())
        
        # Train the model
        logger.info(f"Training SVD model with {self.n_factors} factors and {self.n_epochs} epochs...")
        self.model.fit(self.trainset)
        
        logger.info("Collaborative recommender fitted successfully")
        return self
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        if self.model is None or self.testset is None:
            raise ValueError("Model not fitted yet")
        
        # Make predictions
        predictions = self.model.test(self.testset)
        
        # Calculate metrics
        rmse_score = rmse(predictions)
        mae_score = mae(predictions)
        
        logger.info(f"Evaluation metrics - RMSE: {rmse_score:.4f}, MAE: {mae_score:.4f}")
        
        return {
            'rmse': rmse_score,
            'mae': mae_score
        }
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID.
            movie_id: Movie ID.
            
        Returns:
            Predicted rating.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Convert raw IDs to internal IDs if needed
        try:
            return self.model.predict(user_id, movie_id).est
        except Exception as e:
            logger.warning(f"Error predicting rating for user {user_id}, movie {movie_id}: {str(e)}")
            return 0.0
    
    def get_top_n_recommendations(self, user_id: int, n: int = 10, 
                                  exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Get top N movie recommendations for a user.
        
        Args:
            user_id: User ID.
            n: Number of recommendations.
            exclude_rated: Whether to exclude movies the user has already rated.
            
        Returns:
            List of tuples (movie_id, predicted_rating).
        """
        if self.model is None or self.movies_df is None:
            raise ValueError("Model not fitted yet")
        
        # Check if user exists
        if user_id not in self.user_ids:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        # Get movies the user has already rated
        rated_movies = set()
        if exclude_rated:
            for u, i, r in self.trainset.all_ratings():
                if self.trainset.to_raw_uid(u) == str(user_id):
                    rated_movies.add(int(self.trainset.to_raw_iid(i)))
        
        # Get all movies
        all_movies = set(self.movies_df['movieId'])
        
        # Filter out rated movies
        candidate_movies = all_movies - rated_movies if exclude_rated else all_movies
        
        # Predict ratings for all candidate movies
        predictions = []
        for movie_id in candidate_movies:
            predicted_rating = self.predict_rating(str(user_id), str(movie_id))
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating (descending) and take top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
    def get_recommendations(self, user_id: int, n: int = 10, exclude_rated: bool = True) -> List[Dict]:
        """
        Get movie recommendations with details.
        
        Args:
            user_id: User ID.
            n: Number of recommendations.
            exclude_rated: Whether to exclude movies the user has already rated.
            
        Returns:
            List of dictionaries with movie details and predicted ratings.
        """
        top_n = self.get_top_n_recommendations(user_id, n, exclude_rated)
        
        recommendations = []
        for movie_id, predicted_rating in top_n:
            movie = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie) == 0:
                continue
                
            movie = movie.iloc[0]
            recommendation = {
                'movie': {
                    'movieId': movie.movieId,
                    'title': movie.title,
                    'genres': movie.genres
                },
                'score': predicted_rating,
                'reason': f"Based on your rating patterns, we predict you'll rate this movie {predicted_rating:.1f} out of 5."
            }
            
            # Add additional fields if available
            for field in ['year', 'overview', 'poster_url']:
                if field in movie and movie[field]:
                    recommendation['movie'][field] = movie[field]
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model.
        """
        data = {
            'n_factors': self.n_factors,
            'n_epochs': self.n_epochs,
            'lr_all': self.lr_all,
            'reg_all': self.reg_all,
            'random_state': self.random_state,
            'model': self.model,
            'movies_df': self.movies_df,
            'user_ids': self.user_ids,
            'movie_ids': self.movie_ids
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Collaborative model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CollaborativeRecommender':
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded CollaborativeRecommender instance.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(
            n_factors=data['n_factors'],
            n_epochs=data['n_epochs'],
            lr_all=data['lr_all'],
            reg_all=data['reg_all'],
            random_state=data['random_state']
        )
        
        recommender.model = data['model']
        recommender.movies_df = data['movies_df']
        recommender.user_ids = data['user_ids']
        recommender.movie_ids = data['movie_ids']
        
        logger.info(f"Collaborative model loaded from {path}")
        return recommender 