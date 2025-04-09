"""
Recommendation service for generating movie recommendations.
Implements collaborative filtering and content-based filtering approaches.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app
from database.db import get_db
from data.data_loader import DataLoader
import services.movie_service as movie_service

# Configure logging
logger = logging.getLogger(__name__)

# Initialize data loader
_data_loader = None
# Initialize collaborative filtering model
_collaborative_model = None
# Initialize content-based model
_content_model = None


def _get_data_loader() -> DataLoader:
    """Get or initialize the data loader."""
    return movie_service._get_data_loader()


def _build_collaborative_model():
    """
    Build the collaborative filtering model.
    Uses user-item interaction matrix and computes item-item similarities.
    """
    global _collaborative_model
    
    if _collaborative_model is None:
        try:
            # Get data loader
            data_loader = _get_data_loader()
            
            # Create user-item ratings matrix
            user_item_matrix = data_loader.ratings.pivot_table(
                index='userId', 
                columns='movieId', 
                values='rating'
            ).fillna(0)
            
            # Create item-item similarity matrix
            item_item_sim = cosine_similarity(user_item_matrix.T)
            
            # Create similarity DataFrame
            sim_df = pd.DataFrame(
                item_item_sim,
                index=user_item_matrix.columns,
                columns=user_item_matrix.columns
            )
            
            _collaborative_model = {
                'user_item_matrix': user_item_matrix,
                'item_sim_matrix': sim_df
            }
            
            logger.info(f"Built collaborative filtering model with {len(user_item_matrix.columns)} movies")
        
        except Exception as e:
            logger.error(f"Error building collaborative filtering model: {str(e)}")
            _collaborative_model = None


def _build_content_model():
    """
    Build the content-based filtering model.
    Uses movie genres to compute item-item similarities.
    """
    global _content_model
    
    if _content_model is None:
        try:
            # Get data loader
            data_loader = _get_data_loader()
            
            # Create genre feature matrix (one-hot encoded)
            movies_df = data_loader.movies.copy()
            
            # Split genres into list
            movies_df['genres_list'] = movies_df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
            
            # Get all unique genres
            all_genres = set()
            for genres in movies_df['genres_list']:
                all_genres.update(genres)
            
            # Create one-hot encoded features for genres
            for genre in all_genres:
                movies_df[f'genre_{genre}'] = movies_df['genres_list'].apply(lambda x: 1 if genre in x else 0)
            
            # Get genre feature columns
            genre_cols = [col for col in movies_df.columns if col.startswith('genre_')]
            
            if not genre_cols:
                logger.error("No genre features found for content-based model")
                return
            
            # Compute item-item similarity based on genres
            genre_features = movies_df[genre_cols].values
            movie_ids = movies_df['movieId'].values
            
            # Compute cosine similarity
            content_sim = cosine_similarity(genre_features)
            
            # Create similarity DataFrame
            sim_df = pd.DataFrame(
                content_sim,
                index=movie_ids,
                columns=movie_ids
            )
            
            _content_model = {
                'movies': movies_df,
                'genre_features': genre_features,
                'movie_ids': movie_ids,
                'item_sim_matrix': sim_df
            }
            
            logger.info(f"Built content-based model with {len(genre_cols)} genre features")
        
        except Exception as e:
            logger.error(f"Error building content-based model: {str(e)}")
            _content_model = None


def _ensure_models_built():
    """Ensure that recommendation models are built."""
    if _collaborative_model is None:
        _build_collaborative_model()
    
    if _content_model is None:
        _build_content_model()


def get_unique_genres() -> List[str]:
    """
    Get a list of unique movie genres from the dataset.
    
    Returns:
        List of unique genre names sorted alphabetically
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get movies data
        movies_df = data_loader.movies
        
        # Extract unique genres
        all_genres = set()
        for genres in movies_df['genres']:
            if isinstance(genres, str):
                all_genres.update(genres.split('|'))
        
        # Remove empty strings
        if '' in all_genres:
            all_genres.remove('')
            
        # Sort alphabetically
        return sorted(list(all_genres))
        
    except Exception as e:
        logger.error(f"Error getting unique genres: {str(e)}")
        return []


def get_similar_movies(movie_id: int, limit: int = 10, 
                       method: str = 'hybrid', weights: Dict[str, float] = None) -> List[Dict]:
    """
    Get similar movies based on specified recommendation method.
    
    Args:
        movie_id: Source movie ID
        limit: Number of recommendations to return
        method: Recommendation method ('collaborative', 'content', or 'hybrid')
        weights: Dictionary of weights for hybrid method {'collaborative': 0.7, 'content': 0.3}
    
    Returns:
        List of recommended movie dicts
    """
    try:
        # Ensure models are built
        _ensure_models_built()
        
        # Set default weights
        if weights is None:
            weights = {'collaborative': 0.7, 'content': 0.3}
        
        # Get data loader
        data_loader = _get_data_loader()
        
        # Verify movie exists
        movie = data_loader.get_movie_by_id(movie_id)
        if movie is None:
            logger.error(f"Movie with ID {movie_id} not found")
            return []
        
        # Initialize combined scores
        combined_scores = None
        
        # Get collaborative filtering recommendations
        if method in ['collaborative', 'hybrid']:
            if _collaborative_model is not None and movie_id in _collaborative_model['item_sim_matrix'].columns:
                # Get similarity scores
                cf_scores = _collaborative_model['item_sim_matrix'][movie_id]
                
                # Convert to DataFrame
                cf_scores_df = pd.DataFrame({
                    'movieId': cf_scores.index,
                    'cf_score': cf_scores.values
                })
                
                # Use as initial combined scores
                if combined_scores is None:
                    combined_scores = cf_scores_df.copy()
                    combined_scores['score'] = combined_scores['cf_score'] * weights.get('collaborative', 0.7)
                else:
                    # Merge with existing scores
                    combined_scores = pd.merge(
                        combined_scores,
                        cf_scores_df,
                        on='movieId',
                        how='outer'
                    ).fillna(0)
                    combined_scores['score'] += combined_scores['cf_score'] * weights.get('collaborative', 0.7)
        
        # Get content-based recommendations
        if method in ['content', 'hybrid']:
            if _content_model is not None and movie_id in _content_model['item_sim_matrix'].columns:
                # Get similarity scores
                cb_scores = _content_model['item_sim_matrix'][movie_id]
                
                # Convert to DataFrame
                cb_scores_df = pd.DataFrame({
                    'movieId': cb_scores.index,
                    'cb_score': cb_scores.values
                })
                
                # Use as initial combined scores
                if combined_scores is None:
                    combined_scores = cb_scores_df.copy()
                    combined_scores['score'] = combined_scores['cb_score'] * weights.get('content', 0.3)
                else:
                    # Merge with existing scores
                    combined_scores = pd.merge(
                        combined_scores,
                        cb_scores_df,
                        on='movieId',
                        how='outer'
                    ).fillna(0)
                    combined_scores['score'] += combined_scores['cb_score'] * weights.get('content', 0.3)
        
        if combined_scores is None:
            logger.error(f"Failed to generate recommendations for movie {movie_id}")
            return []
        
        # Sort by score and remove source movie
        combined_scores = combined_scores[combined_scores['movieId'] != movie_id]
        combined_scores = combined_scores.sort_values('score', ascending=False)
        
        # Get top N recommendations
        top_movie_ids = combined_scores.head(limit)['movieId'].tolist()
        
        # Get movie details
        recommended_movies = []
        for rec_id in top_movie_ids:
            movie = movie_service.get_movie_by_id(int(rec_id))
            if movie:
                # Add similarity score
                movie_score = combined_scores[combined_scores['movieId'] == rec_id]['score'].iloc[0]
                movie['similarity_score'] = round(float(movie_score), 2)
                recommended_movies.append(movie)
        
        return recommended_movies
    
    except Exception as e:
        logger.error(f"Error getting similar movies: {str(e)}")
        return []


def get_user_recommendations(user_id: int, limit: int = 10, 
                             method: str = 'collaborative') -> List[Dict]:
    """
    Get personalized movie recommendations for a user.
    
    Args:
        user_id: User ID
        limit: Number of recommendations to return
        method: Recommendation method ('collaborative', 'content', or 'hybrid')
    
    Returns:
        List of recommended movie dicts
    """
    try:
        # Ensure models are built
        _ensure_models_built()
        
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get user ratings
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            # No ratings for this user, return popular movies instead
            logger.info(f"No ratings found for user {user_id}, returning popular movies")
            return movie_service.get_popular_movies(limit)
        
        # Using collaborative filtering approach
        if method == 'collaborative' and _collaborative_model is not None:
            # Get user-item matrix
            user_item_matrix = _collaborative_model['user_item_matrix']
            
            if user_id not in user_item_matrix.index:
                # User not in training data, use content-based approach instead
                logger.info(f"User {user_id} not in collaborative model, using content-based approach")
                return get_recommendations_from_user_history(user_id, limit)
            
            # Get user's ratings
            user_ratings_vec = user_item_matrix.loc[user_id]
            
            # Get rated movies (to exclude from recommendations)
            rated_movies = user_ratings_vec[user_ratings_vec > 0].index.tolist()
            
            # Get item-item similarity matrix
            item_sim_matrix = _collaborative_model['item_sim_matrix']
            
            # Calculate scores for all movies
            scores = []
            for movie_id in item_sim_matrix.columns:
                if movie_id in rated_movies:
                    continue
                
                # Get similarities to movies the user has rated
                sims = item_sim_matrix.loc[rated_movies, movie_id]
                
                # Get user's ratings for those movies
                user_ratings_for_similar = user_ratings_vec[rated_movies]
                
                # Calculate weighted score
                score = np.sum(sims * user_ratings_for_similar) / np.sum(np.abs(sims))
                
                scores.append({
                    'movieId': movie_id,
                    'score': score
                })
            
            # Sort by score
            scores_df = pd.DataFrame(scores).sort_values('score', ascending=False)
            
            # Get top N recommendations
            top_movie_ids = scores_df.head(limit)['movieId'].tolist()
            
            # Get movie details
            recommended_movies = []
            for rec_id in top_movie_ids:
                movie = movie_service.get_movie_by_id(int(rec_id))
                if movie:
                    # Add predicted rating
                    pred_rating = scores_df[scores_df['movieId'] == rec_id]['score'].iloc[0]
                    movie['predicted_rating'] = round(float(pred_rating), 1)
                    recommended_movies.append(movie)
            
            return recommended_movies
        
        else:
            # Fall back to recommendations based on user history
            return get_recommendations_from_user_history(user_id, limit)
    
    except Exception as e:
        logger.error(f"Error getting user recommendations: {str(e)}")
        return []


def get_recommendations_from_user_history(user_id: int, limit: int = 10) -> List[Dict]:
    """
    Get recommendations based on user's rating history.
    
    Args:
        user_id: User ID
        limit: Number of recommendations to return
    
    Returns:
        List of recommended movie dicts
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get user ratings
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            # No ratings for this user, return popular movies instead
            return movie_service.get_popular_movies(limit)
        
        # Find highly rated movies
        highly_rated = user_ratings[user_ratings['rating'] >= 4.0]
        
        if len(highly_rated) == 0:
            # No highly rated movies, use all ratings
            highly_rated = user_ratings
        
        # Get similar movies for each highly rated movie
        all_recommendations = []
        
        for _, row in highly_rated.iterrows():
            similar_movies = get_similar_movies(
                int(row['movieId']),
                limit=5,  # Get 5 per movie
                method='content'  # Content-based is faster
            )
            
            all_recommendations.extend(similar_movies)
        
        # Remove duplicates and already rated movies
        rated_movie_ids = user_ratings['movieId'].tolist()
        unique_recommendations = []
        seen_ids = set()
        
        for movie in all_recommendations:
            movie_id = movie['movieId']
            if movie_id not in seen_ids and movie_id not in rated_movie_ids:
                seen_ids.add(movie_id)
                unique_recommendations.append(movie)
        
        # Sort by similarity score
        sorted_recommendations = sorted(
            unique_recommendations, 
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )
        
        return sorted_recommendations[:limit]
    
    except Exception as e:
        logger.error(f"Error getting recommendations from user history: {str(e)}")
        return []


def get_explanation(user_id: int, movie_id: int) -> Dict:
    """
    Get explanation for why a movie is recommended to a user.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
    
    Returns:
        Dictionary with explanation data
    """
    try:
        # Ensure models are built
        _ensure_models_built()
        
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get movie details
        movie = data_loader.get_movie_by_id(movie_id)
        if movie is None:
            return {"error": "Movie not found"}
        
        # Get user ratings
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id]
        
        # Initialize explanation
        explanation = {
            "movie": movie.to_dict(),
            "reasons": [],
            "similar_movies": [],
            "genre_match": []
        }
        
        # If content model is available
        if _content_model is not None and movie_id in _content_model['item_sim_matrix'].columns:
            # Get movie's genres
            movie_genres = movie['genres'].split('|') if isinstance(movie['genres'], str) else []
            
            # Find other movies with similar genres that the user has rated highly
            rated_movie_ids = user_ratings['movieId'].tolist()
            rated_movies = data_loader.movies[data_loader.movies['movieId'].isin(rated_movie_ids)]
            
            # Find genre matches
            genre_matches = []
            for _, rated_movie in rated_movies.iterrows():
                rated_genres = rated_movie['genres'].split('|') if isinstance(rated_movie['genres'], str) else []
                common_genres = set(movie_genres) & set(rated_genres)
                
                if common_genres:
                    user_rating = user_ratings[
                        user_ratings['movieId'] == rated_movie['movieId']
                    ]['rating'].iloc[0]
                    
                    genre_matches.append({
                        "movie": rated_movie.to_dict(),
                        "common_genres": list(common_genres),
                        "user_rating": float(user_rating)
                    })
            
            # Sort by user rating
            genre_matches = sorted(genre_matches, key=lambda x: x['user_rating'], reverse=True)
            
            # Add to explanation
            explanation["genre_match"] = genre_matches[:3]  # Top 3 matches
            
            if genre_matches:
                explanation["reasons"].append(
                    f"This movie shares genres with movies you've highly rated"
                )
        
        # If collaborative model is available
        if _collaborative_model is not None and movie_id in _collaborative_model['item_sim_matrix'].columns:
            # Get similar movies
            similar_movie_ids = _collaborative_model['item_sim_matrix'][movie_id].sort_values(ascending=False).index[1:6]
            
            # Find which of these similar movies the user has rated highly
            similar_rated = []
            for sim_id in similar_movie_ids:
                sim_ratings = user_ratings[user_ratings['movieId'] == sim_id]
                
                if len(sim_ratings) > 0:
                    sim_movie = data_loader.get_movie_by_id(sim_id)
                    user_rating = sim_ratings['rating'].iloc[0]
                    
                    similar_rated.append({
                        "movie": sim_movie.to_dict(),
                        "user_rating": float(user_rating),
                        "similarity": float(_collaborative_model['item_sim_matrix'].loc[movie_id, sim_id])
                    })
            
            # Sort by rating
            similar_rated = sorted(similar_rated, key=lambda x: x['user_rating'], reverse=True)
            
            # Add to explanation
            explanation["similar_movies"] = similar_rated[:3]  # Top 3 similar movies
            
            if similar_rated:
                explanation["reasons"].append(
                    f"Users who watched movies you liked also enjoyed this movie"
                )
        
        # Add general popularity as a reason if needed
        if not explanation["reasons"]:
            # Get movie popularity
            rating_count = len(data_loader.ratings[data_loader.ratings['movieId'] == movie_id])
            
            if rating_count > 100:
                explanation["reasons"].append(
                    f"This is a popular movie with {rating_count} ratings"
                )
            else:
                explanation["reasons"].append(
                    f"This movie has genres similar to your interests"
                )
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error getting recommendation explanation: {str(e)}")
        return {"error": str(e)}


def get_movie_popularity_score(movie_id: int) -> float:
    """
    Calculate popularity score for a movie.
    
    Args:
        movie_id: Movie ID
    
    Returns:
        Popularity score (0-10)
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get ratings for movie
        movie_ratings = data_loader.ratings[data_loader.ratings['movieId'] == movie_id]
        
        if len(movie_ratings) == 0:
            return 0.0
        
        # Calculate score based on number of ratings and average rating
        num_ratings = len(movie_ratings)
        avg_rating = movie_ratings['rating'].mean()
        
        # Get total number of ratings in dataset for normalization
        total_ratings = len(data_loader.ratings)
        
        # Calculate popularity score
        # Normalize by log scale to prevent extreme values
        popularity = (np.log1p(num_ratings) / np.log1p(total_ratings / len(data_loader.movies))) * 10
        
        # Adjust by average rating
        adjusted_popularity = popularity * (avg_rating / 5.0)
        
        return min(round(adjusted_popularity, 1), 10.0)
    
    except Exception as e:
        logger.error(f"Error calculating movie popularity: {str(e)}")
        return 0.0 