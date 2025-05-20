"""
TMDb API Service Module.

This module provides interfaces to interact with The Movie Database (TMDb) API.
It includes functions for searching movies, fetching movie details, and retrieving
related information like cast, trailers, and watch providers.
"""
import os
import json
import time
import requests
from functools import lru_cache
import logging
import random

# TMDb configuration (should be loaded from environment/config)
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"

# Flag to indicate if we're using dummy data due to missing API key
USING_DUMMY_DATA = False

# Flag to control whether we've already logged the warning about missing API key
LOGGED_API_KEY_WARNING = False

# Simple in-memory cache as fallback if Flask-Caching is not used/available
API_CACHE = {}
CACHE_EXPIRY_TIME = 7 * 24 * 60 * 60  # One week cache

# Setup logger
logger = logging.getLogger(__name__)

def _get_tmdb_api_key():
    """Helper function to get the TMDb API key from Flask config or fallback."""
    global TMDB_API_KEY, USING_DUMMY_DATA, LOGGED_API_KEY_WARNING
    if TMDB_API_KEY is None:
        TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
        if not TMDB_API_KEY:
            if not LOGGED_API_KEY_WARNING:
                logger.warning("TMDB_API_KEY is not set. Using dummy data for TMDb features.")
                LOGGED_API_KEY_WARNING = True
            USING_DUMMY_DATA = True
            return None
    return TMDB_API_KEY

# Function to generate dummy data for movies when API key is missing
def _generate_dummy_movie_data(movie_id=None, title=None):
    """Generate dummy movie data to use when TMDb API is not available."""
    if movie_id is None:
        movie_id = random.randint(1000, 9999)
    
    if title is None:
        dummy_titles = [
            "Imaginary Adventure", "Ghost of Tomorrow", "Virtual Reality Dream",
            "Digital Odyssey", "Simulation Game", "Holographic Memory",
            "Quantum Paradox", "Synthetic Journey", "Binary Dreams", "Placeholder Movie"
        ]
        title = f"{random.choice(dummy_titles)} {random.randint(1, 5)}"
    
    # Generate placeholders with movie ID to make them unique
    poster_path = f"/placeholder_poster_{movie_id}.jpg"
    backdrop_path = f"/placeholder_backdrop_{movie_id}.jpg"
    
    return {
        "id": movie_id,
        "title": title,
        "overview": "This is a placeholder description for a movie. TMDb API is not available or configured.",
        "release_date": f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "vote_average": round(random.uniform(6.0, 9.0), 1),
        "vote_count": random.randint(100, 5000),
        "popularity": round(random.uniform(10.0, 500.0), 1),
        "poster_path": poster_path,
        "backdrop_path": backdrop_path,
        "poster_url": f"https://via.placeholder.com/500x750/3498db/ffffff?text={title.replace(' ', '+')}",
        "backdrop_url": f"https://via.placeholder.com/1280x720/2c3e50/ffffff?text={title.replace(' ', '+')}",
        "genres": [{"id": 1, "name": "Placeholder"}],
        "runtime": random.randint(90, 150),
        "dummy_data": True  # Flag to indicate this is dummy data
    }

# API Read Access Token (for v4 API if needed)
TMDB_ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NmFhYzI3MWZiZjVlYzI0ZDdjNTI3MzY0MmY0YTc0YSIsIm5iZiI6MTc0NDIwNDQ2NS43LCJzdWIiOiI2N2Y2NzJiMTdiNDNiZGNlMjBhZGQzMGYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.pbxzkkd1LO1xzcTlNbZDkOTma-PI4HlHoZokN7JXonw'

# Cache dictionary for API responses to avoid redundant requests
# Key: endpoint_path, Value: (response_data, timestamp)
API_CACHE = {}
CACHE_EXPIRY_TIME = 7 * 24 * 60 * 60  # One week cache instead of one day

def _make_request(endpoint, params=None):
    """
    Make a request to the TMDb API using Flask-Caching if available.
    
    Args:
        endpoint (str): The API endpoint path
        params (dict, optional): Additional query parameters
        
    Returns:
        dict or None: The API response as JSON or None on error.
    """
    api_key = _get_tmdb_api_key()
    if not api_key:
        # Return appropriate dummy data based on endpoint
        if endpoint.startswith("/movie/") and "similar" in endpoint:
            # Similar movies request
            movie_id = endpoint.split("/")[2]
            return {"results": [_generate_dummy_movie_data() for _ in range(10)]}
        elif endpoint.startswith("/movie/") and "videos" in endpoint:
            # Videos request
            return {"results": []}
        elif endpoint.startswith("/movie/") and "keywords" in endpoint:
            # Keywords request
            return {"keywords": [{"id": i, "name": f"keyword{i}"} for i in range(1, 6)]}
        elif endpoint.startswith("/movie/") and "credits" in endpoint:
            # Credits request
            return {
                "cast": [{"id": i, "name": f"Actor {i}", "character": f"Character {i}"} for i in range(1, 6)],
                "crew": [{"id": 100, "name": "Director", "job": "Director"}]
            }
        elif endpoint.startswith("/movie/") and "watch/providers" in endpoint:
            # Watch providers request
            return {"results": {}}
        elif endpoint.startswith("/movie/"):
            # Movie details request
            movie_id = endpoint.split("/")[2]
            return _generate_dummy_movie_data(movie_id=int(movie_id))
        elif endpoint == "/search/movie":
            # Search request
            title = params.get("query", "Unknown")
            return {"results": [_generate_dummy_movie_data(title=title) for _ in range(3)]}
        else:
            # Generic dummy response
            return {"results": []}
    
    if params is None:
        params = {}
        
    # Convert params to JSON string for hashing
    params_json = json.dumps(params, sort_keys=True)
    cache_key = f"{endpoint}:{params_json}"
    
    # Check cache first
    if cache_key in API_CACHE:
        cached_data, timestamp = API_CACHE[cache_key]
        if time.time() - timestamp < CACHE_EXPIRY_TIME:
            return cached_data

    url = f"{TMDB_BASE_URL}{endpoint}"
    try:
        params['api_key'] = api_key # Use fetched API key
        response = requests.get(url, params=params)
        response.raise_for_status()  
        data = response.json()
        API_CACHE[cache_key] = (data, time.time())
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to TMDb API ({endpoint}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        return None # Return None on error
    except Exception as e:
         logger.error(f"Unexpected error in _make_request ({endpoint}): {e}", exc_info=True)
         return None

def get_poster_url(poster_path, size="w500"):
    """
    Get the full URL for a movie poster image.
    
    Args:
        poster_path (str): The poster path from TMDb
        size (str): The size of the image (w92, w154, w185, w342, w500, w780, original)
        
    Returns:
        str: The full URL to the poster image or a placeholder
    """
    if not poster_path:
        # Return a placeholder image URL rather than None
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    
    return f"{TMDB_IMAGE_BASE_URL}{size}{poster_path}"

def get_backdrop_url(backdrop_path, size="w1280"):
    """
    Get the full URL for a movie backdrop image.
    
    Args:
        backdrop_path (str): The backdrop path from TMDb
        size (str): The size of the image (w300, w780, w1280, original)
        
    Returns:
        str: The full URL to the backdrop image or a placeholder
    """
    if not backdrop_path:
        # Return a placeholder image URL rather than None
        return "https://via.placeholder.com/1280x720?text=No+Backdrop+Available"
    
    return f"{TMDB_IMAGE_BASE_URL}{size}{backdrop_path}"

@lru_cache(maxsize=1000)
def search_movie_by_title(title, year=None):
    """
    Search for movies by title.
    
    Args:
        title (str): The movie title to search for
        year (int, optional): Release year to narrow search
        
    Returns:
        list: List of movie matches
    """
    params = {'query': title}
    if year:
        params['year'] = year
    
    try:
        response = _make_request("/search/movie", params)
        return response.get('results', [])
    except Exception as e:
        print(f"Error searching movie by title: {e}")
        return []

@lru_cache(maxsize=1000)
def get_movie_details(movie_id):
    """
    Get comprehensive details for a movie by its TMDb ID.
    
    Args:
        movie_id (int): The TMDb movie ID
        
    Returns:
        dict: Movie details including title, overview, release date, etc.
    """
    try:
        # Get basic movie details
        movie_details = _make_request(f"/movie/{movie_id}", {
            'append_to_response': 'keywords,videos,credits'
        })
        
        # Extract director from credits
        director = None
        if 'credits' in movie_details and 'crew' in movie_details['credits']:
            directors = [
                crew for crew in movie_details['credits']['crew']
                if crew.get('job') == 'Director'
            ]
            if directors:
                director = directors[0]
        
        # Extract cast
        cast = []
        if 'credits' in movie_details and 'cast' in movie_details['credits']:
            cast = movie_details['credits']['cast'][:10]  # Get top 10 cast members
        
        # Extract trailers
        trailers = []
        if 'videos' in movie_details and 'results' in movie_details['videos']:
            trailers = [
                video for video in movie_details['videos']['results']
                if video.get('type') == 'Trailer' and video.get('site') == 'YouTube'
            ]
        
        # Extract keywords
        keywords = []
        if 'keywords' in movie_details and 'keywords' in movie_details['keywords']:
            keywords = movie_details['keywords']['keywords']
        
        # Create a structured response
        result = {
            'id': movie_details.get('id'),
            'title': movie_details.get('title'),
            'original_title': movie_details.get('original_title'),
            'overview': movie_details.get('overview'),
            'release_date': movie_details.get('release_date'),
            'runtime': movie_details.get('runtime'),
            'genres': movie_details.get('genres', []),
            'vote_average': movie_details.get('vote_average'),
            'vote_count': movie_details.get('vote_count'),
            'popularity': movie_details.get('popularity'),
            'poster_path': movie_details.get('poster_path'),
            'backdrop_path': movie_details.get('backdrop_path'),
            'director': director,
            'cast': cast,
            'trailers': trailers,
            'keywords': keywords,
            'production_companies': movie_details.get('production_companies', []),
            'production_countries': movie_details.get('production_countries', []),
            'poster_url': get_poster_url(movie_details.get('poster_path')),
            'backdrop_url': get_backdrop_url(movie_details.get('backdrop_path'))
        }
        
        return result
    except Exception as e:
        print(f"Error getting movie details: {e}")
        return None

def get_watch_providers(movie_id, region='IN'):
    """
    Get watch providers (streaming services) for a movie in a specific region.
    
    Args:
        movie_id (int): The TMDb movie ID
        region (str): The region code (e.g., 'IN' for India)
        
    Returns:
        dict: Watch providers information
    """
    try:
        response = _make_request(f"/movie/{movie_id}/watch/providers")
        results = response.get('results', {})
        providers = results.get(region, {})
        
        return {
            'flatrate': providers.get('flatrate', []),  # Streaming
            'rent': providers.get('rent', []),  # Rental
            'buy': providers.get('buy', [])  # Purchase
        }
    except Exception as e:
        print(f"Error getting watch providers: {e}")
        return {'flatrate': [], 'rent': [], 'buy': []}

def get_similar_movies(movie_id, max_results=10):
    """
    Get similar movies as recommended by TMDb.
    
    Args:
        movie_id (int): The TMDb movie ID
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of similar movies
    """
    try:
        response = _make_request(f"/movie/{movie_id}/similar")
        similar_movies = response.get('results', [])
        
        # Limit the number of results and add poster URLs
        result = []
        for movie in similar_movies[:max_results]:
            movie['poster_url'] = get_poster_url(movie.get('poster_path'))
            result.append(movie)
        
        return result
    except Exception as e:
        print(f"Error getting similar movies: {e}")
        return []

def find_tmdb_id_for_movie(title, year=None):
    """
    Find the TMDb ID for a movie using its title and optional year.
    
    Args:
        title (str): The movie title
        year (int, optional): The release year
        
    Returns:
        int or None: The TMDb ID if found, None otherwise
    """
    try:
        # First search for the movie
        search_results = search_movie_by_title(title, year)
        
        if not search_results:
            return None
        
        # If multiple matches, find the closest one
        if year:
            # Try to match the year exactly
            for result in search_results:
                release_date = result.get('release_date', '')
                if release_date.startswith(str(year)):
                    return result['id']
        
        # If no exact year match or no year provided, return the first result
        return search_results[0]['id']
    except Exception as e:
        print(f"Error finding TMDb ID for movie: {e}")
        return None

def get_movie_keywords(movie_id):
    """
    Get keywords associated with a movie.
    
    Args:
        movie_id (int): The TMDb movie ID
        
    Returns:
        list: List of keyword objects
    """
    try:
        response = _make_request(f"/movie/{movie_id}/keywords")
        return response.get('keywords', [])
    except Exception as e:
        print(f"Error getting movie keywords: {e}")
        return []

def get_movie_credits(movie_id):
    """
    Get cast and crew for a movie.
    
    Args:
        movie_id (int): The TMDb movie ID
        
    Returns:
        dict: Cast and crew information
    """
    try:
        response = _make_request(f"/movie/{movie_id}/credits")
        return {
            'cast': response.get('cast', []),
            'crew': response.get('crew', [])
        }
    except Exception as e:
        print(f"Error getting movie credits: {e}")
        return {'cast': [], 'crew': []}

def get_movie_videos(movie_id):
    """
    Get videos (trailers, teasers, etc.) for a movie.
    
    Args:
        movie_id (int): The TMDb movie ID
        
    Returns:
        list: List of video objects
    """
    try:
        response = _make_request(f"/movie/{movie_id}/videos")
        return response.get('results', [])
    except Exception as e:
        print(f"Error getting movie videos: {e}")
        return []