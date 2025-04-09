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
from typing import List, Dict, Optional, Tuple, Any
from flask import current_app
from data.data_loader import DataLoader
from services.tmdb_service import (
    find_tmdb_id_for_movie, 
    get_movie_details, 
    get_watch_providers,
    get_similar_movies
)

# Setup logger
logger = logging.getLogger(__name__)

# Dictionary to cache TMDb IDs (movieId -> tmdb_id)
_TMDB_ID_CACHE = {}

def _extract_year_from_title(title):
    """Extract the year from a movie title if present in parentheses."""
    year_match = re.search(r'\((\d{4})\)$', title)
    if year_match:
        return int(year_match.group(1))
    return None

def _get_data_loader() -> DataLoader:
    """
    Get or create a DataLoader instance.
    
    Returns:
        DataLoader instance
    """
    try:
        # First, try to use the global cached loader if available
        from app import get_cached_data_loader
        return get_cached_data_loader()
    except (ImportError, AttributeError):
        try:
            # Fallback to creating a new one based on current_app config
            # Get paths from config
            movies_path = current_app.config.get('MOVIES_CSV', 'data/movies.csv')
            ratings_path = current_app.config.get('RATINGS_CSV', 'data/ratings.csv')
            
            # Create DataLoader
            data_loader = DataLoader(
                movies_path=movies_path,
                ratings_path=ratings_path,
                test_size=current_app.config.get('TEST_SIZE', 0.2),
                random_state=current_app.config.get('RANDOM_STATE', 42)
            )
            
            logger.info(f"DataLoader created with movies from {movies_path}")
            return data_loader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {str(e)}")
            # Return a basic DataLoader with default paths as fallback
            return DataLoader(
                movies_path='data/movies.csv',
                ratings_path='data/ratings.csv'
            )


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
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Get movies by genre
        genre_movies = data_loader.get_movies_by_genre(genre)
        
        # Apply sorting
        sort_ascending = sort_order.lower() != 'desc'
        
        if sort_by == 'year' and 'year' in genre_movies.columns:
            genre_movies = genre_movies.sort_values('year', ascending=sort_ascending)
        elif sort_by == 'rating' and 'vote_average' in genre_movies.columns:
            genre_movies = genre_movies.sort_values('vote_average', ascending=sort_ascending)
        else:  # Default to title
            genre_movies = genre_movies.sort_values('title', ascending=sort_ascending)
        
        # Get total count
        total = len(genre_movies)
        
        # Apply pagination
        offset = (page - 1) * per_page
        genre_movies = genre_movies.iloc[offset:offset + per_page]
        
        # Convert to list of dictionaries
        movies_list = genre_movies.to_dict('records')
        
        return movies_list, total
        
    except Exception as e:
        logger.error(f"Error getting movies for genre '{genre}': {str(e)}")
        return [], 0


def get_popular_movies(limit: int = 10) -> List[Dict]:
    """
    Get popular movies based on number of ratings or other popularity metrics.
    
    Args:
        limit: Maximum number of movies to return
        
    Returns:
        List of movie dictionaries
    """
    try:
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Get popular movies
        popular_movies = data_loader.get_popular_movies(n=limit)
        
        # Convert to list of dictionaries
        movies_list = popular_movies.to_dict('records')
        
        return movies_list
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {str(e)}")
        return []


def get_high_rated_movies(limit: int = 10, min_ratings: int = 10) -> List[Dict]:
    """
    Get highly rated movies with a minimum number of ratings.
    
    Args:
        limit: Maximum number of movies to return
        min_ratings: Minimum number of ratings required
        
    Returns:
        List of movie dictionaries
    """
    try:
        # Get DataLoader
        data_loader = _get_data_loader()
        
        # Get high rated movies
        high_rated_movies = data_loader.get_high_rated_movies(min_ratings=min_ratings, n=limit)
        
        # Convert to list of dictionaries
        movies_list = high_rated_movies.to_dict('records')
        
        return movies_list
        
    except Exception as e:
        logger.error(f"Error getting high rated movies: {str(e)}")
        return []


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
        movie_id: Movie ID
        limit: Maximum number of similar movies to return
        
    Returns:
        List of similar movies
    """
    try:
        # Get movie first
        movie = get_movie_by_id(movie_id, with_tmdb=False)
        
        if not movie:
            return []
        
        # Get TMDb ID
        tmdb_id = get_tmdb_id_for_movie(movie)
        
        if not tmdb_id:
            return []
        
        # Get similar movies from TMDb
        similar_movies = get_similar_movies(tmdb_id, max_results=limit)
        
        return similar_movies
    except Exception as e:
        logger.error(f"Error getting TMDb similar movies for {movie_id}: {str(e)}")
        return []

def enrich_movies_list(movies: List[Dict], with_tmdb: bool = True) -> List[Dict]:
    """
    Enrich a list of movies with TMDb data.
    
    Args:
        movies: List of movie dictionaries
        with_tmdb: Whether to include TMDb data
        
    Returns:
        List of enriched movie dictionaries
    """
    if not with_tmdb:
        return movies
    
    # Local cache for this function call to avoid enriching the same movie twice
    # (sometimes movies can appear multiple times in a list)
    local_cache = {}
    
    enriched_movies = []
    for movie in movies:
        try:
            # Use movieId as cache key
            movie_id = movie.get('movieId')
            if movie_id in local_cache:
                # Use cached result
                enriched_movies.append(local_cache[movie_id])
                continue
                
            # Enrich movie
            enriched_movie = enrich_movie_with_tmdb(movie)
            
            # Cache the result
            local_cache[movie_id] = enriched_movie
            
            # Add to result list
            enriched_movies.append(enriched_movie)
        except Exception as e:
            # Only log movie ID to avoid Unicode issues
            logger.error(f"Error enriching movie ID {movie.get('movieId', 'unknown')}: {str(e)}")
            # Add the original movie to the list on error
            enriched_movies.append(movie)
    
    return enriched_movies 