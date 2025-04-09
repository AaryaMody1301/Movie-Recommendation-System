"""
Hybrid recommender combining content-based and collaborative filtering.
"""
import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    
    This recommender combines recommendations from both approaches using
    a weighted average or ranking fusion.
    """
    
    def __init__(self, content_recommender: ContentBasedRecommender = None,
                 collaborative_recommender: CollaborativeRecommender = None,
                 content_weight: float = 0.5):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_recommender: Content-based recommender instance.
            collaborative_recommender: Collaborative recommender instance.
            content_weight: Weight for content-based recommendations (0-1).
                            Collaborative weight is (1 - content_weight).
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight = content_weight
        self.collab_weight = 1.0 - content_weight
        self.movies_df = None
    
    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None,
            train_content: bool = True, train_collaborative: bool = True,
            test_size: float = 0.2, random_state: int = 42):
        """
        Fit the models with data.
        
        Args:
            movies_df: DataFrame with movie data.
            ratings_df: DataFrame with rating data (required for collaborative filtering).
            train_content: Whether to train the content-based model.
            train_collaborative: Whether to train the collaborative model.
            test_size: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Self instance.
        """
        self.movies_df = movies_df.copy()
        
        # Train content-based model
        if train_content and self.content_recommender is not None:
            logger.info("Training content-based model...")
            self.content_recommender.fit(movies_df)
        
        # Train collaborative model
        if train_collaborative and self.collaborative_recommender is not None:
            if ratings_df is None:
                raise ValueError("Ratings data required for collaborative filtering")
            
            logger.info("Training collaborative model...")
            self.collaborative_recommender.fit(ratings_df, movies_df, test_size=test_size)
        
        logger.info("Hybrid recommender fitted successfully")
        return self
    
    def get_recommendations_for_user(self, user_id: int, n: int = 10,
                                     strategy: str = 'weighted') -> List[Dict]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID.
            n: Number of recommendations.
            strategy: Combination strategy ('weighted' or 'rank').
            
        Returns:
            List of recommendation dictionaries.
        """
        if self.movies_df is None:
            raise ValueError("Model not fitted yet")
        
        # Get collaborative filtering recommendations if available
        collab_recs = []
        if self.collaborative_recommender is not None:
            try:
                collab_recs = self.collaborative_recommender.get_recommendations(user_id, n=n*2)
            except Exception as e:
                logger.warning(f"Error getting collaborative recommendations: {str(e)}")
        
        # Get content-based recommendations from user's highly rated movies
        content_recs = []
        if self.content_recommender is not None:
            try:
                # If we have collaborative data, get user's top-rated movies
                if self.collaborative_recommender is not None:
                    # Get ratings for this user from the training set
                    user_ratings = []
                    if hasattr(self.collaborative_recommender, 'trainset'):
                        trainset = self.collaborative_recommender.trainset
                        for u, i, r in trainset.all_ratings():
                            if trainset.to_raw_uid(u) == str(user_id):
                                movie_id = int(trainset.to_raw_iid(i))
                                user_ratings.append((movie_id, r))
                    
                    # Sort by rating and get top movies
                    if user_ratings:
                        user_ratings.sort(key=lambda x: x[1], reverse=True)
                        top_movies = user_ratings[:5]  # Use top 5 rated movies
                        
                        # Get content-based recommendations for each top movie
                        for movie_id, _ in top_movies:
                            movie_recs = self.content_recommender.get_recommendations(movie_id, top_n=n)
                            content_recs.extend(movie_recs)
                    else:
                        # If no ratings, use a popular movie as seed
                        # This is a fallback for cold-start users
                        popular_movies = self.movies_df.sort_values('movieId').head(1)
                        if len(popular_movies) > 0:
                            seed_movie_id = popular_movies.iloc[0]['movieId']
                            content_recs = self.content_recommender.get_recommendations(seed_movie_id, top_n=n*2)
                
                # If still no content recommendations, use a default approach
                if not content_recs and len(self.movies_df) > 0:
                    seed_movie_id = self.movies_df.iloc[0]['movieId']
                    content_recs = self.content_recommender.get_recommendations(seed_movie_id, top_n=n*2)
            
            except Exception as e:
                logger.warning(f"Error getting content-based recommendations: {str(e)}")
        
        # Combine recommendations based on strategy
        if strategy == 'weighted':
            return self._combine_weighted(content_recs, collab_recs, n)
        elif strategy == 'rank':
            return self._combine_rank_fusion(content_recs, collab_recs, n)
        else:
            raise ValueError(f"Unknown combination strategy: {strategy}")
    
    def get_recommendations_for_movie(self, movie_id: int, user_id: Optional[int] = None,
                                     n: int = 10, strategy: str = 'weighted') -> List[Dict]:
        """
        Get recommendations based on a movie, optionally personalized for a user.
        
        Args:
            movie_id: Movie ID.
            user_id: Optional user ID for personalization.
            n: Number of recommendations.
            strategy: Combination strategy ('weighted' or 'rank').
            
        Returns:
            List of recommendation dictionaries.
        """
        if self.movies_df is None:
            raise ValueError("Model not fitted yet")
        
        # Get content-based recommendations
        content_recs = []
        if self.content_recommender is not None:
            try:
                content_recs = self.content_recommender.get_recommendations(movie_id, top_n=n*2)
            except Exception as e:
                logger.warning(f"Error getting content-based recommendations: {str(e)}")
        
        # If user_id is provided, get collaborative recommendations
        collab_recs = []
        if user_id is not None and self.collaborative_recommender is not None:
            try:
                collab_recs = self.collaborative_recommender.get_recommendations(user_id, n=n*2)
            except Exception as e:
                logger.warning(f"Error getting collaborative recommendations: {str(e)}")
        
        # Combine recommendations based on strategy
        if strategy == 'weighted':
            return self._combine_weighted(content_recs, collab_recs, n)
        elif strategy == 'rank':
            return self._combine_rank_fusion(content_recs, collab_recs, n)
        else:
            raise ValueError(f"Unknown combination strategy: {strategy}")
    
    def _combine_weighted(self, content_recs: List[Dict], collab_recs: List[Dict], n: int) -> List[Dict]:
        """
        Combine recommendations using weighted average of scores.
        
        Args:
            content_recs: Content-based recommendations.
            collab_recs: Collaborative recommendations.
            n: Number of recommendations to return.
            
        Returns:
            List of combined recommendations.
        """
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process content-based recommendations
        for rec in content_recs:
            movie_id = rec['movie']['movieId']
            score = rec['score'] * self.content_weight
            if movie_id in combined_scores:
                combined_scores[movie_id]['score'] += score
                combined_scores[movie_id]['sources'].append('content')
            else:
                combined_scores[movie_id] = {
                    'movie': rec['movie'],
                    'score': score,
                    'sources': ['content'],
                    'content_reason': rec.get('reason', '')
                }
        
        # Process collaborative recommendations
        for rec in collab_recs:
            movie_id = rec['movie']['movieId']
            score = rec['score'] * self.collab_weight
            if movie_id in combined_scores:
                combined_scores[movie_id]['score'] += score
                combined_scores[movie_id]['sources'].append('collab')
                combined_scores[movie_id]['collab_reason'] = rec.get('reason', '')
            else:
                combined_scores[movie_id] = {
                    'movie': rec['movie'],
                    'score': score,
                    'sources': ['collab'],
                    'collab_reason': rec.get('reason', '')
                }
        
        # Convert to list and sort by score
        recommendations = list(combined_scores.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Format the final recommendations
        final_recs = []
        for rec in recommendations[:n]:
            # Generate a combined reason
            reason = self._generate_combined_reason(rec)
            
            final_rec = {
                'movie': rec['movie'],
                'score': rec['score'],
                'reason': reason
            }
            final_recs.append(final_rec)
        
        return final_recs
    
    def _combine_rank_fusion(self, content_recs: List[Dict], collab_recs: List[Dict], n: int) -> List[Dict]:
        """
        Combine recommendations using rank fusion.
        
        Args:
            content_recs: Content-based recommendations.
            collab_recs: Collaborative recommendations.
            n: Number of recommendations to return.
            
        Returns:
            List of combined recommendations.
        """
        # Constant for CombSUM rank fusion
        k = 60  # Fusion constant
        
        # Create dictionaries for ranks
        content_ranks = {rec['movie']['movieId']: i+1 for i, rec in enumerate(content_recs)}
        collab_ranks = {rec['movie']['movieId']: i+1 for i, rec in enumerate(collab_recs)}
        
        # Combine all movie IDs
        all_movie_ids = set(content_ranks.keys()) | set(collab_ranks.keys())
        
        # Calculate fusion scores
        fusion_scores = {}
        for movie_id in all_movie_ids:
            # Get ranks (default to a high rank if not present)
            content_rank = content_ranks.get(movie_id, len(content_recs) + 1)
            collab_rank = collab_ranks.get(movie_id, len(collab_recs) + 1)
            
            # Calculate fusion score (higher is better)
            fusion_score = 0
            if movie_id in content_ranks:
                fusion_score += self.content_weight * (1.0 / (k + content_rank))
            if movie_id in collab_ranks:
                fusion_score += self.collab_weight * (1.0 / (k + collab_rank))
            
            # Find the movie data
            movie_data = None
            content_reason = ""
            collab_reason = ""
            
            for rec in content_recs:
                if rec['movie']['movieId'] == movie_id:
                    movie_data = rec['movie']
                    content_reason = rec.get('reason', '')
                    break
            
            if movie_data is None:
                for rec in collab_recs:
                    if rec['movie']['movieId'] == movie_id:
                        movie_data = rec['movie']
                        collab_reason = rec.get('reason', '')
                        break
            
            if movie_data:
                fusion_scores[movie_id] = {
                    'movie': movie_data,
                    'score': fusion_score,
                    'sources': ['content' if movie_id in content_ranks else '', 'collab' if movie_id in collab_ranks else ''],
                    'content_reason': content_reason,
                    'collab_reason': collab_reason
                }
        
        # Convert to list and sort by fusion score
        recommendations = list(fusion_scores.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Format the final recommendations
        final_recs = []
        for rec in recommendations[:n]:
            # Generate a combined reason
            reason = self._generate_combined_reason(rec)
            
            final_rec = {
                'movie': rec['movie'],
                'score': rec['score'],
                'reason': reason
            }
            final_recs.append(final_rec)
        
        return final_recs
    
    def _generate_combined_reason(self, rec: Dict) -> str:
        """
        Generate a combined reason for recommendation.
        
        Args:
            rec: Recommendation data with sources and reasons.
            
        Returns:
            Combined reason string.
        """
        sources = rec.get('sources', [])
        
        if 'content' in sources and 'collab' in sources:
            return f"Recommended based on both similar movies you might like and your rating patterns. " + \
                   f"{rec.get('content_reason', '')}"
        elif 'content' in sources:
            return rec.get('content_reason', 'Recommended based on movie similarity.')
        elif 'collab' in sources:
            return rec.get('collab_reason', 'Recommended based on your rating patterns.')
        else:
            return "Recommended based on popularity."
    
    def save(self, path: str) -> None:
        """
        Save the hybrid recommender to a file.
        
        Args:
            path: Path to save the model.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'content_weight': self.content_weight,
            'collab_weight': self.collab_weight,
            'movies_df': self.movies_df
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Hybrid recommender saved to {path}")
    
    @classmethod
    def load(cls, path: str, content_recommender: Optional[ContentBasedRecommender] = None,
             collaborative_recommender: Optional[CollaborativeRecommender] = None) -> 'HybridRecommender':
        """
        Load the hybrid recommender from a file.
        
        Args:
            path: Path to load the model from.
            content_recommender: Content-based recommender instance.
            collaborative_recommender: Collaborative recommender instance.
            
        Returns:
            Loaded HybridRecommender instance.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(
            content_recommender=content_recommender,
            collaborative_recommender=collaborative_recommender,
            content_weight=data['content_weight']
        )
        
        recommender.movies_df = data['movies_df']
        
        logger.info(f"Hybrid recommender loaded from {path}")
        return recommender 