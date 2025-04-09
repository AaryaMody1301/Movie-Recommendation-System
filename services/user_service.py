"""
User service for managing user profiles, interaction history, and preferences.
Provides functionality for user authentication, profile management and tracking user activity.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from flask import current_app
from database.db import get_db
from data.data_loader import DataLoader
import services.movie_service as movie_service

# Configure logging
logger = logging.getLogger(__name__)

# Initialize data loader
_data_loader = None


def _get_data_loader() -> DataLoader:
    """Get or initialize the data loader."""
    return movie_service._get_data_loader()


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """
    Get user information by user ID.
    
    Args:
        user_id: The user ID
    
    Returns:
        User information as dictionary or None if not found
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Check if user exists in ratings data
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            logger.error(f"User with ID {user_id} not found in ratings data")
            return None
        
        # Get user's ratings statistics
        rating_count = len(user_ratings)
        avg_rating = round(float(user_ratings['rating'].mean()), 1) if rating_count > 0 else 0.0
        
        # Get list of rated movie IDs
        rated_movies = user_ratings['movieId'].unique().tolist()
        
        # Get favorite genres
        favorite_genres = get_user_favorite_genres(user_id)
        
        # Construct user profile
        user_profile = {
            'userId': int(user_id),
            'ratingCount': int(rating_count),
            'averageRating': avg_rating,
            'ratedMovieCount': len(rated_movies),
            'favoriteGenres': favorite_genres,
            'lastActive': datetime.now().isoformat(),
            'memberSince': "N/A"  # Would come from user database in real app
        }
        
        return user_profile
    
    except Exception as e:
        logger.error(f"Error getting user by ID: {str(e)}")
        return None


def get_user_ratings(user_id: int, limit: int = 50, 
                     sort_by: str = 'timestamp', 
                     sort_order: str = 'desc') -> List[Dict]:
    """
    Get user's movie ratings.
    
    Args:
        user_id: User ID
        limit: Maximum number of ratings to return
        sort_by: Field to sort by ('timestamp', 'rating', 'movieId')
        sort_order: Sort order ('asc' or 'desc')
        
    Returns:
        List of rating dictionaries with movie details
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get user ratings
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id].copy()
        
        if len(user_ratings) == 0:
            return []
        
        # Sort ratings
        if sort_by == 'timestamp':
            ascending = sort_order == 'asc'
            user_ratings = user_ratings.sort_values('timestamp', ascending=ascending)
        elif sort_by == 'rating':
            ascending = sort_order == 'asc'
            user_ratings = user_ratings.sort_values('rating', ascending=ascending)
        elif sort_by == 'movieId':
            ascending = sort_order == 'asc'
            user_ratings = user_ratings.sort_values('movieId', ascending=ascending)
        
        # Apply limit
        user_ratings = user_ratings.head(limit)
        
        # Prepare result with movie details
        result = []
        for _, rating in user_ratings.iterrows():
            # Get movie details
            movie = movie_service.get_movie_by_id(int(rating['movieId']))
            
            if movie:
                # Format rating data
                rating_data = {
                    'userId': int(rating['userId']),
                    'movieId': int(rating['movieId']),
                    'rating': float(rating['rating']),
                    'timestamp': int(rating['timestamp']) if 'timestamp' in rating else None,
                    'date': datetime.fromtimestamp(rating['timestamp']).isoformat() if 'timestamp' in rating else None,
                    'movie': movie
                }
                
                result.append(rating_data)
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting user ratings: {str(e)}")
        return []


def get_user_favorite_genres(user_id: int, top_n: int = 3) -> List[Dict]:
    """
    Get user's favorite genres based on rating history.
    
    Args:
        user_id: User ID
        top_n: Number of top genres to return
        
    Returns:
        List of genre dictionaries with scores
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get user ratings
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return []
        
        # Get movies with ratings
        rated_movie_ids = user_ratings['movieId'].tolist()
        movies_df = data_loader.movies[data_loader.movies['movieId'].isin(rated_movie_ids)].copy()
        
        # Merge with ratings
        movies_with_ratings = pd.merge(
            movies_df,
            user_ratings,
            on='movieId'
        )
        
        # Extract all genres
        genre_ratings = []
        
        for _, row in movies_with_ratings.iterrows():
            if not isinstance(row['genres'], str):
                continue
                
            # Split genres
            genres = row['genres'].split('|')
            rating = row['rating']
            
            # Add each genre with the rating
            for genre in genres:
                genre_ratings.append({
                    'genre': genre,
                    'rating': rating
                })
        
        # Convert to DataFrame
        if not genre_ratings:
            return []
            
        genre_df = pd.DataFrame(genre_ratings)
        
        # Calculate average rating per genre
        genre_stats = genre_df.groupby('genre').agg(
            avg_rating=('rating', 'mean'),
            count=('rating', 'count')
        ).reset_index()
        
        # Calculate a score that takes into account both rating and count
        # (weighted score to prioritize genres with more ratings)
        max_count = genre_stats['count'].max()
        genre_stats['score'] = genre_stats['avg_rating'] * (1 + np.log1p(genre_stats['count']) / np.log1p(max_count))
        
        # Sort by score
        genre_stats = genre_stats.sort_values('score', ascending=False)
        
        # Convert to list of dicts
        result = []
        for _, row in genre_stats.head(top_n).iterrows():
            result.append({
                'genre': row['genre'],
                'averageRating': round(float(row['avg_rating']), 1),
                'count': int(row['count']),
                'score': round(float(row['score']), 2)
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting user favorite genres: {str(e)}")
        return []


def add_user_rating(user_id: int, movie_id: int, rating: float) -> Dict:
    """
    Add or update a user rating for a movie.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        rating: Rating value (0-5)
        
    Returns:
        Dictionary with result status
    """
    try:
        # Validate input
        if not (0 <= rating <= 5):
            return {"success": False, "error": "Rating must be between 0 and 5"}
        
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get movie (to verify it exists)
        movie = movie_service.get_movie_by_id(movie_id)
        if movie is None:
            return {"success": False, "error": f"Movie with ID {movie_id} not found"}
        
        # In a real application, this would update a database
        # For this demo, we'll update the in-memory DataFrame
        
        # Check if rating already exists
        existing_rating = data_loader.ratings[
            (data_loader.ratings['userId'] == user_id) & 
            (data_loader.ratings['movieId'] == movie_id)
        ]
        
        # Prepare new rating data
        new_rating = {
            'userId': user_id,
            'movieId': movie_id,
            'rating': rating,
            'timestamp': int(datetime.now().timestamp())
        }
        
        # Update rating
        if len(existing_rating) > 0:
            # Update existing rating
            data_loader.ratings.loc[
                (data_loader.ratings['userId'] == user_id) & 
                (data_loader.ratings['movieId'] == movie_id),
                ['rating', 'timestamp']
            ] = [rating, new_rating['timestamp']]
            
            action = "updated"
        else:
            # Add new rating
            data_loader.ratings = pd.concat([
                data_loader.ratings, 
                pd.DataFrame([new_rating])
            ], ignore_index=True)
            
            action = "added"
        
        # Create response with movie details
        response = {
            "success": True,
            "action": action,
            "rating": {
                "userId": user_id,
                "movieId": movie_id,
                "rating": rating,
                "timestamp": new_rating['timestamp'],
                "date": datetime.fromtimestamp(new_rating['timestamp']).isoformat()
            },
            "movie": movie
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error adding user rating: {str(e)}")
        return {"success": False, "error": str(e)}


def delete_user_rating(user_id: int, movie_id: int) -> Dict:
    """
    Delete a user rating for a movie.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        Dictionary with result status
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Check if rating exists
        existing_rating = data_loader.ratings[
            (data_loader.ratings['userId'] == user_id) & 
            (data_loader.ratings['movieId'] == movie_id)
        ]
        
        if len(existing_rating) == 0:
            return {"success": False, "error": "Rating not found"}
        
        # Delete rating
        data_loader.ratings = data_loader.ratings[
            ~((data_loader.ratings['userId'] == user_id) & 
              (data_loader.ratings['movieId'] == movie_id))
        ]
        
        return {
            "success": True,
            "message": f"Rating for movie {movie_id} by user {user_id} deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting user rating: {str(e)}")
        return {"success": False, "error": str(e)}


def get_user_activity(user_id: int, limit: int = 20) -> List[Dict]:
    """
    Get user activity history.
    This is a placeholder implementation that returns rating activity.
    In a real app, this would track views, searches, etc.
    
    Args:
        user_id: User ID
        limit: Maximum number of activities to return
        
    Returns:
        List of activity dictionaries
    """
    try:
        # Get user ratings as activity
        ratings = get_user_ratings(
            user_id, 
            limit=limit, 
            sort_by='timestamp', 
            sort_order='desc'
        )
        
        # Convert to activity format
        activities = []
        
        for rating in ratings:
            activity = {
                "type": "rating",
                "userId": rating['userId'],
                "timestamp": rating['timestamp'],
                "date": rating['date'],
                "details": {
                    "movieId": rating['movieId'],
                    "rating": rating['rating'],
                    "movieTitle": rating['movie']['title']
                }
            }
            
            activities.append(activity)
        
        return activities
    
    except Exception as e:
        logger.error(f"Error getting user activity: {str(e)}")
        return []


def get_user_statistics(user_id: int) -> Dict:
    """
    Get user statistics.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary with user statistics
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get user ratings
        user_ratings = data_loader.ratings[data_loader.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return {"error": "User not found or has no ratings"}
        
        # Calculate rating distribution
        rating_counts = user_ratings['rating'].value_counts().sort_index()
        
        # Create rating distribution with all possible ratings
        distribution = {}
        for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            distribution[str(r)] = int(rating_counts.get(r, 0))
        
        # Get rated genres
        genres_data = get_user_favorite_genres(user_id, top_n=10)
        
        # Calculate activity over time (by month)
        # Convert timestamp to datetime
        user_ratings['date'] = pd.to_datetime(user_ratings['timestamp'], unit='s')
        user_ratings['month'] = user_ratings['date'].dt.strftime('%Y-%m')
        
        # Group by month
        monthly_activity = user_ratings.groupby('month').size().reset_index()
        monthly_activity.columns = ['month', 'count']
        
        # Convert to list of dicts
        activity_by_month = []
        for _, row in monthly_activity.iterrows():
            activity_by_month.append({
                'month': row['month'],
                'count': int(row['count'])
            })
        
        # Prepare statistics
        statistics = {
            'userId': int(user_id),
            'totalRatings': int(len(user_ratings)),
            'averageRating': round(float(user_ratings['rating'].mean()), 2),
            'ratingDistribution': distribution,
            'favoriteGenres': genres_data,
            'activityByMonth': activity_by_month
        }
        
        return statistics
    
    except Exception as e:
        logger.error(f"Error getting user statistics: {str(e)}")
        return {"error": str(e)}


def get_user_watchlist(user_id: int) -> List[Dict]:
    """
    Get user's watchlist.
    This is a placeholder implementation.
    In a real app, this would get data from a database.
    
    Args:
        user_id: User ID
        
    Returns:
        List of movie dictionaries in the watchlist
    """
    # This would be implemented with a database in a real application
    # For now, return an empty list
    return []


def add_to_watchlist(user_id: int, movie_id: int) -> Dict:
    """
    Add a movie to user's watchlist.
    This is a placeholder implementation.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        Dictionary with result status
    """
    # This would be implemented with a database in a real application
    return {
        "success": True,
        "message": f"Movie {movie_id} added to watchlist for user {user_id}"
    }


def remove_from_watchlist(user_id: int, movie_id: int) -> Dict:
    """
    Remove a movie from user's watchlist.
    This is a placeholder implementation.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        Dictionary with result status
    """
    # This would be implemented with a database in a real application
    return {
        "success": True,
        "message": f"Movie {movie_id} removed from watchlist for user {user_id}"
    }


def calculate_user_similarity(user_id1: int, user_id2: int) -> float:
    """
    Calculate similarity between two users based on ratings.
    
    Args:
        user_id1: First user ID
        user_id2: Second user ID
        
    Returns:
        Similarity score (0-1)
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get ratings for both users
        ratings1 = data_loader.ratings[data_loader.ratings['userId'] == user_id1]
        ratings2 = data_loader.ratings[data_loader.ratings['userId'] == user_id2]
        
        if len(ratings1) == 0 or len(ratings2) == 0:
            return 0.0
        
        # Find common movies
        movies1 = set(ratings1['movieId'])
        movies2 = set(ratings2['movieId'])
        common_movies = movies1.intersection(movies2)
        
        if len(common_movies) < 3:  # Need at least 3 common movies for meaningful similarity
            return 0.0
        
        # Create rating vectors for common movies
        common_ratings1 = []
        common_ratings2 = []
        
        for movie_id in common_movies:
            rating1 = ratings1[ratings1['movieId'] == movie_id]['rating'].iloc[0]
            rating2 = ratings2[ratings2['movieId'] == movie_id]['rating'].iloc[0]
            
            common_ratings1.append(rating1)
            common_ratings2.append(rating2)
        
        # Calculate cosine similarity
        dot_product = sum(r1 * r2 for r1, r2 in zip(common_ratings1, common_ratings2))
        magnitude1 = sum(r ** 2 for r in common_ratings1) ** 0.5
        magnitude2 = sum(r ** 2 for r in common_ratings2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return min(max(0.0, float(similarity)), 1.0)
    
    except Exception as e:
        logger.error(f"Error calculating user similarity: {str(e)}")
        return 0.0


def find_similar_users(user_id: int, limit: int = 10) -> List[Dict]:
    """
    Find users with similar taste.
    
    Args:
        user_id: User ID
        limit: Maximum number of similar users to return
        
    Returns:
        List of similar user dictionaries
    """
    try:
        # Get data loader
        data_loader = _get_data_loader()
        
        # Get all unique user IDs
        all_users = data_loader.ratings['userId'].unique()
        
        # Calculate similarity for all users
        similarities = []
        
        for other_id in all_users:
            if other_id == user_id:
                continue
                
            similarity = calculate_user_similarity(user_id, other_id)
            
            if similarity > 0:
                similarities.append({
                    'userId': int(other_id),
                    'similarity': round(similarity, 3)
                })
        
        # Sort by similarity (descending)
        similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        
        # Return top similar users
        return similarities[:limit]
    
    except Exception as e:
        logger.error(f"Error finding similar users: {str(e)}")
        return [] 