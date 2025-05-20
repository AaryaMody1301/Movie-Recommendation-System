"""
Movie service for handling movie operations.

This module provides functions to fetch, search, and filter movies from the dataset.
It also integrates with TMDb API for enhanced movie data.
"""
import logging
import os
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Any, Callable
from flask import current_app, g
# from recommendation import ContentBasedRecommender # Old import
from models.content_based import ContentBasedRecommender # Import the SentenceTransformer recommender
from services.tmdb_service import (
    find_tmdb_id_for_movie, 
    get_movie_details, 
    get_watch_providers,
    get_similar_movies as get_tmdb_similar_movies_api # Renamed to avoid conflict
)

# Setup logger
logger = logging.getLogger(__name__)

# Import cache object from app
cache = None

def set_cache(cache_obj):
    """Set the cache object from the Flask app."""
    global cache
    cache = cache_obj

# Define a cache decorator fallback for when cache is None
def memoize_or_pass_through(timeout=300):
    """
    Decorator that will use cache.memoize if available, otherwise just return the original function.
    This handles the case when cache is None.
    """
    def decorator(func):
        if cache is not None:
            return cache.memoize(timeout=timeout)(func)
        else:
            return func
    return decorator

# DataLoader singleton for fast access
_DATA_LOADER = None

def get_data_loader():
    global _DATA_LOADER
    if _DATA_LOADER is None:
        from data.data_loader import DataLoader
        _DATA_LOADER = DataLoader(
            movies_path=os.environ.get('MOVIES_CSV', 'data/movies.csv'),
            ratings_path=os.environ.get('RATINGS_CSV', 'data/ratings.csv')
        )
    return _DATA_LOADER

# Global instance for the recommender
_RECOMMENDER = None

# Dictionary to cache TMDb IDs (movieId -> tmdb_id)
_TMDB_ID_CACHE = {}

def _extract_year_from_title(title):
    """Extract the year from a movie title if present in parentheses."""
    year_match = re.search(r'\((\d{4})\)$', title)
    if year_match:
        return int(year_match.group(1))
    return None

def _get_data_loader():
    return get_data_loader()

def _get_recommender() -> Optional[ContentBasedRecommender]:
    """
    Get the ContentBasedRecommender instance from Flask's g object.
    It should be initialized in create_app.
    """
    if 'recommender' not in g:
        logger.error("ContentBasedRecommender not found in g. It should be initialized in create_app.")
        return None
    return g.recommender

def get_all_movies(page: int = 1, per_page: int = 24, 
                   sort_by: str = 'title', sort_order: str = 'asc') -> Tuple[List[Dict], int]:
    """
    Get all movies with pagination and sorting.
    
    Args:
        page: Page number (1-indexed)
        per_page: Number of items per page
        sort_by: Column to sort by ('title', 'year', 'rating')
        sort_order: Sort order ('asc' or 'desc')
        
    Returns:
        Tuple of (list of movie dictionaries, total count)
    """
    try:
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Get movies DataFrame
        movies_df = data_loader.get_movies()
        
        # Apply sorting
        sort_ascending = sort_order.lower() != 'desc'
        
        if sort_by == 'year' and 'year' in movies_df.columns:
            movies_df = movies_df.sort_values('year', ascending=sort_ascending)
        elif sort_by == 'rating' and hasattr(data_loader, 'get_high_rated_movies'):
            # This sorts by average rating - might need adjustment based on your DataLoader
            movies_df = data_loader.get_high_rated_movies(n=len(movies_df))
            if not sort_ascending:
                movies_df = movies_df.iloc[::-1]
        else:  # Default to title
            movies_df = movies_df.sort_values('title', ascending=sort_ascending)
        
        # Get total count
        total = len(movies_df)
        
        # Apply pagination
        offset = (page - 1) * per_page
        movies_df = movies_df.iloc[offset:offset + per_page]
        
        # Convert to list of dictionaries
        movies_list = movies_df.to_dict('records')
        
        return movies_list, total
        
    except Exception as e:
        logger.error(f"Error getting all movies: {str(e)}")
        return [], 0


def get_movie_by_id(movie_id: int, with_tmdb: bool = True) -> Optional[Dict]:
    """
    Get a movie by its ID, optionally enriched with TMDb data.
    
    Args:
        movie_id: Movie ID
        with_tmdb: Whether to include TMDb data
        
    Returns:
        Movie dictionary or None if not found
    """
    try:
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Get movie by ID
        movie = data_loader.get_movie_by_id(movie_id)
        
        # Return as dictionary if found
        if movie is not None:
            movie_dict = movie.to_dict()
            
            # Enrich with TMDb data if requested
            if with_tmdb:
                movie_dict = enrich_movie_with_tmdb(movie_dict)
            
            return movie_dict
        else:
            logger.warning(f"Movie with ID {movie_id} not found")
            return None
            
    except Exception as e:
        logger.error(f"Error getting movie {movie_id}: {str(e)}")
        return None


def search_movies(query: str, page: int = 1, per_page: int = 24) -> Tuple[List[Dict], int]:
    """
    Search movies by title or other fields.
    
    Args:
        query: Search query
        page: Page number (1-indexed)
        per_page: Number of items per page
        
    Returns:
        Tuple of (list of movie dictionaries, total count)
    """
    try:
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Search movies
        search_results = data_loader.search_movies(query)
        
        # Get total count
        total = len(search_results)
        
        # Apply pagination
        offset = (page - 1) * per_page
        search_results = search_results.iloc[offset:offset + per_page]
        
        # Convert to list of dictionaries
        movies_list = search_results.to_dict('records')
        
        return movies_list, total
        
    except Exception as e:
        logger.error(f"Error searching movies for '{query}': {str(e)}")
        return [], 0


def get_movies_by_genre(genre: str, page: int = 1, per_page: int = 24,
                        sort_by: str = 'title', sort_order: str = 'asc') -> Tuple[List[Dict], int]:
    """
    Get movies by genre with pagination and sorting.
    
    Args:
        genre: Genre to filter by
        page: Page number (1-indexed)
        per_page: Number of items per page
        sort_by: Column to sort by ('title', 'year', 'rating')
        sort_order: Sort order ('asc' or 'desc')
        
    Returns:
        Tuple of (list of movie dictionaries, total count)
    """
    try:
        data_loader = _get_data_loader()
        
        # Get movies by genre using DataLoader method
        genre_movies_df = data_loader.get_movies_by_genre(genre)
        
        # Apply sorting (handle potential missing columns)
        sort_ascending = sort_order.lower() != 'desc'
        
        # Determine the actual column name for sorting rating (could be avg_rating etc.)
        rating_col = None
        if sort_by == 'rating':
            if 'average_rating' in genre_movies_df.columns:
                rating_col = 'average_rating'
            elif 'avg_rating' in genre_movies_df.columns:
                rating_col = 'avg_rating'
            # Add more potential rating column names if needed

        if sort_by == 'year' and 'year' in genre_movies_df.columns:
            genre_movies_df = genre_movies_df.sort_values('year', ascending=sort_ascending, na_position='last')
        elif rating_col:
            genre_movies_df = genre_movies_df.sort_values(rating_col, ascending=sort_ascending, na_position='last')
        else:  # Default to title
            # Ensure title column exists before sorting
            if 'title' in genre_movies_df.columns:
                 genre_movies_df = genre_movies_df.sort_values('title', ascending=sort_ascending, na_position='last')
            else:
                 logger.warning("Cannot sort by title: 'title' column missing.")

        # Get total count before pagination
        total = len(genre_movies_df)

        # Apply pagination
        offset = (page - 1) * per_page
        genre_movies_df = genre_movies_df.iloc[offset:offset + per_page]
        
        # Convert to list of dictionaries
        movies_list = genre_movies_df.to_dict('records')
        
        return movies_list, total
        
    except Exception as e:
        logger.error(f"Error getting movies for genre '{genre}': {str(e)}")
        return [], 0


@memoize_or_pass_through(timeout=3600) # Cache popular movies for 1 hour
def get_popular_movies(limit: int = 10) -> List[Dict]:
    """
    Get popular movies based on number of ratings or other popularity metrics.
    
    Args:
        limit: Maximum number of movies to return
        
    Returns:
        List of movie dictionaries
    """
    try:
        data_loader = _get_data_loader()
        popular_movies_df = data_loader.get_popular_movies(n=limit)
        return popular_movies_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting popular movies: {str(e)}")
        return []


@memoize_or_pass_through(timeout=3600) # Cache high-rated movies for 1 hour
def get_high_rated_movies(limit: int = 10, min_ratings: int = 10) -> List[Dict]:
    """
    Get high-rated movies based on average rating and minimum rating count.
    
    Args:
        limit: Maximum number of movies to return
        min_ratings: Minimum number of ratings required
        
    Returns:
        List of movie dictionaries
    """
    try:
        data_loader = _get_data_loader()
        high_rated_movies_df = data_loader.get_high_rated_movies(min_ratings=min_ratings, n=limit)
        return high_rated_movies_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting high-rated movies: {str(e)}")
        return []


@memoize_or_pass_through(timeout=600) # Cache recommendations for 10 minutes
def get_content_recommendations(movie_id: int, top_n: int = 10) -> List[Dict]:
    """
    Get content-based movie recommendations using the fitted recommender.
    If no recommendations are found, return fallback popular movies.
    """
    recommender = _get_recommender()
    if not recommender:
        logger.error("ContentBasedRecommender is not available.")
        return get_popular_movies(top_n)

    try:
        recommendations = recommender.get_recommendations(movie_id, top_n=top_n)
        if not recommendations:
            logger.info(f"No content-based recommendations for movie {movie_id}, returning fallback popular movies.")
            return get_popular_movies(top_n)
        return recommendations
    except ValueError as ve:
        logger.warning(f"Could not get recommendations for movie ID {movie_id}: {ve}")
        return get_popular_movies(top_n)
    except Exception as e:
        logger.error(f"Error getting content recommendations for movie ID {movie_id}: {str(e)}", exc_info=True)
        return get_popular_movies(top_n)


def update_movie_metadata(movie_id: int, metadata: Dict[str, Any]) -> bool:
    """
    Update a movie's metadata in the database.
    
    Args:
        movie_id: Movie ID
        metadata: Dictionary with metadata to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Get the movie
        movie = data_loader.get_movie_by_id(movie_id)
        if movie is None:
            logger.warning(f"Movie with ID {movie_id} not found")
            return False
        
        # Update movie in the movies DataFrame
        # Note: This only updates the in-memory DataFrame, not the CSV file
        # In a real application, you would save to database or CSV
        movies_df = data_loader.get_movies()
        idx = movies_df.index[movies_df['movieId'] == movie_id].tolist()
        
        if not idx:
            logger.warning(f"Movie with ID {movie_id} not found in DataFrame")
            return False
        
        # Update only allowed fields
        allowed_fields = ['title', 'genres', 'release_year', 'director', 
                         'plot', 'poster_path', 'imdb_id', 'runtime']
        
        for field, value in metadata.items():
            if field in allowed_fields and field in movies_df.columns:
                movies_df.at[idx[0], field] = value
        
        logger.info(f"Movie {movie_id} metadata updated")
        return True
        
    except Exception as e:
        logger.error(f"Error updating movie {movie_id} metadata: {str(e)}")
        return False 

def get_tmdb_id_for_movie(movie: Dict) -> Optional[int]:
    """
    Get TMDb ID for a movie, either from cache or by searching the TMDb API.
    
    Args:
        movie: Movie dictionary with at least 'movieId' and 'title'
        
    Returns:
        TMDb ID or None if not found
    """
    try:
        movie_id = movie.get('movieId')
        
        # Check cache first
        if movie_id in _TMDB_ID_CACHE:
            return _TMDB_ID_CACHE[movie_id]
        
        # Extract year from title if available
        title = movie.get('title', '')
        year = _extract_year_from_title(title)
        
        # Clean title (remove year in parentheses)
        clean_title = re.sub(r'\s*\(\d{4}\)$', '', title)
        
        # Search TMDb
        tmdb_id = find_tmdb_id_for_movie(clean_title, year)
        
        if tmdb_id:
            # Cache the result
            _TMDB_ID_CACHE[movie_id] = tmdb_id
            return tmdb_id
        
        return None
    except Exception as e:
        # Only log movie ID to avoid Unicode issues
        logger.error(f"Error getting TMDb ID for movie ID {movie.get('movieId', 'unknown')}: {str(e)}")
        return None

def enrich_movie_with_tmdb(movie: Dict) -> Dict:
    """
    Enrich movie data with information from TMDb.
    
    Args:
        movie: Movie dictionary with at least 'movieId' and 'title'
        
    Returns:
        Enriched movie dictionary
    """
    try:
        # Make a copy of the original movie to avoid modifying the input
        enriched_movie = movie.copy()
        
        # Initialize TMDb fields with default values
        enriched_movie['tmdb_id'] = None
        enriched_movie['tmdb_poster_url'] = None
        enriched_movie['tmdb_backdrop_url'] = None
        
        # Get TMDb ID
        tmdb_id = get_tmdb_id_for_movie(movie)
        
        if not tmdb_id:
            # No TMDb match found, return movie with default empty fields
            # Use debug instead of warning to reduce console noise
            logger.debug(f"No TMDb match found for movie {movie.get('movieId')}")
            return enriched_movie
        
        # Store the TMDb ID in the movie dictionary
        enriched_movie['tmdb_id'] = tmdb_id
        
        # Get movie details from TMDb
        tmdb_details = get_movie_details(tmdb_id)
        
        if not tmdb_details:
            # No details found, return movie with just the ID
            return enriched_movie
        
        # Get watch providers
        watch_providers = get_watch_providers(tmdb_id)
        
        # Add TMDb data to the enriched movie dictionary
        tmdb_fields = {
            'tmdb_poster_url': tmdb_details.get('poster_url'),
            'tmdb_backdrop_url': tmdb_details.get('backdrop_url'),
            'overview': tmdb_details.get('overview'),
            'release_date': tmdb_details.get('release_date'),
            'runtime': tmdb_details.get('runtime'),
            'vote_average': tmdb_details.get('vote_average'),
            'vote_count': tmdb_details.get('vote_count'),
            'popularity': tmdb_details.get('popularity'),
            'tmdb_genres': tmdb_details.get('genres', []),
            'director': tmdb_details.get('director'),
            'cast': tmdb_details.get('cast', []),
            'trailers': tmdb_details.get('trailers', []),
            'keywords': tmdb_details.get('keywords', []),
            'production_companies': tmdb_details.get('production_companies', []),
            'production_countries': tmdb_details.get('production_countries', []),
            'watch_providers': watch_providers
        }
        
        # Update the enriched movie with TMDb data (only non-None values)
        for key, value in tmdb_fields.items():
            if value is not None:
                enriched_movie[key] = value
        
        return enriched_movie
    except Exception as e:
        # Only log movie ID to avoid Unicode issues
        logger.error(f"Error enriching movie ID {movie.get('movieId', 'unknown')}: {str(e)}")
        # Return the original movie data on error, don't return None
        return movie

def get_tmdb_similar_movies(movie_id: int, limit: int = 10) -> List[Dict]:
    """
    Get similar movies from TMDb API.
    
    Args:
        movie_id: The *TMDb* ID of the movie (Note: Not the local movieId).
        limit: Number of similar movies to return.
        
    Returns:
        List of similar movie dictionaries from TMDb.
    """
    try:
        # Directly call the renamed TMDb API function with the correct parameter name (max_results)
        similar_movies_raw = get_tmdb_similar_movies_api(movie_id, max_results=limit)
        
        # Optional: Process/format the raw TMDb results if needed
        # For now, return as is
        return similar_movies_raw
        
    except Exception as e:
        logger.error(f"Error getting TMDb similar movies for TMDb ID {movie_id}: {str(e)}")
        return []

def enrich_movies_list(movies: List[Dict], with_tmdb: bool = True) -> List[Dict]:
    """
    Enrich a list of movie dictionaries, optionally with TMDb data.
    
    Args:
        movies: List of movie dictionaries
        with_tmdb: Whether to include TMDb data
        
    Returns:
        List of enriched movie dictionaries
    """
    # Handle empty list or None
    if not movies:
        return []
        
    if not with_tmdb:
        return movies
    
    # Local cache for this function call to avoid enriching the same movie twice
    # (sometimes movies can appear multiple times in a list)
    local_cache = {}
    
    enriched_movies = []
    for movie in movies:
        try:
            # Skip None or invalid movies
            if not movie or not isinstance(movie, dict):
                logger.warning("Skipping invalid movie object in enrich_movies_list")
                continue

            # Use movieId as cache key
            movie_id = movie.get('movieId')
            if not movie_id:
                enriched_movies.append(movie)
                continue

            if movie_id in local_cache:
                enriched_movies.append(local_cache[movie_id])
                continue

            # Enrich movie
            enriched_movie = enrich_movie_with_tmdb(movie)

            # Ensure a valid poster URL is always set
            if not enriched_movie.get('tmdb_poster_url') or not enriched_movie['tmdb_poster_url'].startswith('http'):
                # Use local static placeholder if available
                enriched_movie['tmdb_poster_url'] = '/static/img/movie-placeholder.jpg'

            # Cache the result
            local_cache[movie_id] = enriched_movie
            enriched_movies.append(enriched_movie)
        except Exception as e:
            logger.error(f"Error enriching movie ID {movie.get('movieId', 'unknown')}: {str(e)}")
            if movie:
                enriched_movies.append(movie)
    return enriched_movies

# Make unique genre list cachable
@memoize_or_pass_through(timeout=86400) # Cache genres for 1 day
def get_unique_genres() -> List[str]:
    """
    Get a list of all unique genres in the dataset.
    
    Returns:
        List of unique genre strings
    """
    try:
        data_loader = _get_data_loader()
        return data_loader.get_unique_genres()
    except Exception as e:
        logger.error(f"Error getting unique genres: {str(e)}")
        return [] 

def find_local_id_from_tmdb_id(tmdb_id: int) -> Optional[int]:
    """
    Find the local movie ID based on a TMDb ID.
    
    Args:
        tmdb_id: The TMDb ID to look up
        
    Returns:
        The local movieId if found, None otherwise
    """
    try:
        data_loader = _get_data_loader()
        movies_df = data_loader.get_movies()
        
        # Check if 'tmdb_id' column exists - if using our enriched dataframe
        if 'tmdb_id' in movies_df.columns:
            movie = movies_df[movies_df['tmdb_id'] == tmdb_id]
            if not movie.empty:
                return int(movie.iloc[0]['movieId'])
        
        # Otherwise try to find the movie by searching through enriched movies
        # This is more expensive as it has to enrich all movies
        # We'll limit to a sample of movies to make it more efficient
        sample_size = min(1000, len(movies_df))
        sample_df = movies_df.sample(sample_size)
        
        for _, row in sample_df.iterrows():
            movie_dict = row.to_dict()
            enriched = enrich_movie_with_tmdb(movie_dict)
            if enriched.get('tmdb_id') == tmdb_id:
                return int(enriched['movieId'])
        
        return None
    except Exception as e:
        logger.error(f"Error finding local ID for TMDb ID {tmdb_id}: {str(e)}")
        return None