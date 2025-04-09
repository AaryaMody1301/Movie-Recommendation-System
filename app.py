"""
Movie Recommendation System Flask Application.

This is the main application file that initializes the Flask app,
registers blueprints, and sets up the necessary configurations.
"""
import os
import logging
import dotenv
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager
from services.auth_service import UserAuth, get_user_by_id
from database.db import db, init_db
from services.movie_service import get_movie_by_id, get_tmdb_similar_movies, get_all_movies, get_high_rated_movies, enrich_movies_list, search_movies, get_movies_by_genre
from typing import List, Dict, Tuple

# Import simple app functionality
from simple_app import load_movies, build_tfidf_matrix, get_movie_recommendations, get_unique_genres

# Create a global cache for DataLoader to avoid repeated instantiation
from data.data_loader import DataLoader
_DATA_LOADER_CACHE = None
def get_cached_data_loader():
    """Get or create a cached DataLoader instance."""
    global _DATA_LOADER_CACHE
    if _DATA_LOADER_CACHE is None:
        _DATA_LOADER_CACHE = DataLoader(
            movies_path='data/movies.csv',
            ratings_path='data/ratings.csv'
        )
    return _DATA_LOADER_CACHE

# Load environment variables from .env file
dotenv.load_dotenv()

def create_app(test_config=None):
    """
    Create and configure the Flask application.
    
    Args:
        test_config: Test configuration dictionary
        
    Returns:
        Configured Flask application
    """
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_change_in_production'),
        DATABASE=os.path.join(app.instance_path, 'movie_recommendation.sqlite'),
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{os.path.join(app.instance_path, "movie_recommendation.sqlite")}',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'uploads'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload
        TEMPLATES_AUTO_RELOAD=True
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError:
        pass

    # Initialize database
    init_db(app)
    
    # Initialize stub SQLAlchemy
    db.init_app(app)

    # Load movies data
    movies_df = load_movies()
    if len(movies_df) > 0:
        build_tfidf_matrix()
    
    # Setup login manager
    login_manager = LoginManager()
    login_manager.login_view = 'login'
    login_manager.login_message_category = 'info'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        # Convert to integer since Flask-Login uses string IDs
        user_data = get_user_by_id(int(user_id))
        if user_data:
            return UserAuth(user_data)
        return None

    # Basic routes
    @app.route('/')
    def index():
        """Render the homepage."""
        # Initialize with empty lists in case of errors
        popular_movies = []
        top_rated_movies = []
        
        # Temporarily raise the logging level for movie_service to suppress warnings
        movie_service_logger = logging.getLogger('services.movie_service')
        original_level = movie_service_logger.level
        movie_service_logger.setLevel(logging.ERROR)
        
        try:
            # Get data loader once
            data_loader = get_cached_data_loader()
            
            # Get movies directly without pagination first (faster than service calls)
            all_movies_df = data_loader.get_movies()
            
            # For popular movies, sort by title and take first batch
            # (in a real app, you would sort by popularity metrics)
            popular_df = all_movies_df.sort_values('title').head(24)
            popular_movies = popular_df.to_dict('records')
            
            # For top rated, just take a different batch 
            # (in a real app, you would sort by actual ratings)
            top_rated_df = all_movies_df.sort_values('title', ascending=False).head(24)
            top_rated_movies = top_rated_df.to_dict('records')
            
            # Enrich with TMDb data - do this once for all movies to leverage caching
            all_movies_to_enrich = popular_movies + top_rated_movies
            all_enriched = enrich_movies_list(all_movies_to_enrich)
            
            # Split the results back
            popular_movies = all_enriched[:len(popular_movies)]
            top_rated_movies = all_enriched[len(popular_movies):]
            
            # Filter for movies with poster URLs and limit to 8
            popular_movies = [movie for movie in popular_movies if movie.get('tmdb_poster_url')][:8]
            top_rated_movies = [movie for movie in top_rated_movies if movie.get('tmdb_poster_url')][:8]
            
            # If we still don't have enough, try another approach
            if len(popular_movies) < 8 or len(top_rated_movies) < 8:
                # Get more movies randomly
                random_movies = all_movies_df.sample(min(50, len(all_movies_df))).to_dict('records')
                random_enriched = enrich_movies_list(random_movies)
                
                # Fill missing popular movies
                if len(popular_movies) < 8:
                    more_movies = [m for m in random_enriched if m.get('tmdb_poster_url') and m not in popular_movies]
                    popular_movies.extend(more_movies[:8-len(popular_movies)])
                
                # Fill missing top rated movies
                if len(top_rated_movies) < 8:
                    more_movies = [m for m in random_enriched if m.get('tmdb_poster_url') and m not in top_rated_movies and m not in popular_movies]
                    top_rated_movies.extend(more_movies[:8-len(top_rated_movies)])
        except Exception as e:
            print(f"Error getting movies for homepage: {e}")
            # Fallback to simple approach with minimal operations
            try:
                movies_df = load_movies()
                fallback_movies = movies_df.head(16).to_dict('records')
                fallback_enriched = enrich_movies_list(fallback_movies)
                
                # Split between popular and top rated
                mid = len(fallback_enriched) // 2
                popular_movies = fallback_enriched[:mid]
                top_rated_movies = fallback_enriched[mid:] 
            except Exception as e:
                print(f"Error with fallback movies: {e}")
        
        # Restore original logging level
        movie_service_logger.setLevel(original_level)
        
        # Get all genres
        genres = get_unique_genres()
        
        return render_template(
            'index.html',
            popular_movies=popular_movies,
            top_rated_movies=top_rated_movies,
            genres=genres
        )

    @app.route('/search')
    def search():
        """Search for movies."""
        query = request.args.get('query', '')
        
        if not query:
            return render_template('search.html', movies=[], query='', genres=get_unique_genres())
        
        # Temporarily raise the logging level
        movie_service_logger = logging.getLogger('services.movie_service')
        original_level = movie_service_logger.level
        movie_service_logger.setLevel(logging.ERROR)
        
        try:
            # Use cached data loader directly for better performance
            data_loader = get_cached_data_loader()
            
            # Search movies directly
            search_results = data_loader.search_movies(query, limit=20)
            
            # Convert to list of dictionaries
            matching_movies = search_results.to_dict('records')
            
            # Enrich with TMDb data in one batch operation
            movies = enrich_movies_list(matching_movies)
            
            # Restore original logging level
            movie_service_logger.setLevel(original_level)
            
            return render_template('search.html', movies=movies, query=query, genres=get_unique_genres())
        except Exception as e:
            # Restore original logging level in case of error
            movie_service_logger.setLevel(original_level)
            print(f"Error searching movies: {e}")
            
            # Fallback to simple search with DataFrame operations
            try:
                # Load movies directly
                movies_df = load_movies()
                matching_movies = movies_df[movies_df['title'].str.contains(query, case=False)]
                movies = matching_movies.head(20).to_dict('records')
                movies = enrich_movies_list(movies)
                return render_template('search.html', movies=movies, query=query, genres=get_unique_genres())
            except Exception as e:
                print(f"Error with search fallback: {e}")
                return render_template('search.html', movies=[], query=query, genres=get_unique_genres())

    @app.route('/movie/<int:movie_id>')
    def movie_detail(movie_id):
        """Show movie details with TMDb enrichment."""
        # Temporarily raise the logging level 
        movie_service_logger = logging.getLogger('services.movie_service')
        original_level = movie_service_logger.level
        movie_service_logger.setLevel(logging.ERROR)
        
        try:
            # Get the movie with TMDb data
            movie_data = get_movie_by_id(movie_id, with_tmdb=True)
            
            if not movie_data:
                movie_service_logger.setLevel(original_level)
                return render_template('404.html', genres=get_unique_genres()), 404
            
            # Get content-based recommendations
            similar_movies = []
            try:
                similar_movies = get_movie_recommendations(movie_data['title'], n=6)
                
                # Get all the movie IDs we need to enrich at once
                movie_ids_to_enrich = [rec['movie']['movieId'] for rec in similar_movies if 'movie' in rec]
                
                # Pre-fetch all movies at once rather than one at a time
                movies_to_enrich = {}
                if movie_ids_to_enrich:
                    data_loader = get_cached_data_loader()
                    for movie_id in movie_ids_to_enrich:
                        movie = data_loader.get_movie_by_id(movie_id)
                        if movie is not None:
                            movie_dict = movie.to_dict()
                            movies_to_enrich[movie_id] = movie_dict
                    
                    # Enrich all movies at once
                    if movies_to_enrich:
                        enriched_movies = enrich_movies_list(list(movies_to_enrich.values()))
                        # Update the dictionary with enriched movies
                        for enriched_movie in enriched_movies:
                            movie_id = enriched_movie.get('movieId')
                            if movie_id in movies_to_enrich:
                                movies_to_enrich[movie_id] = enriched_movie
                
                # Now update the recommendations with the enriched data
                for recommendation in similar_movies:
                    if 'movie' in recommendation:
                        movie_obj = recommendation['movie']
                        movie_id = movie_obj.get('movieId')
                        if movie_id in movies_to_enrich:
                            enriched_movie = movies_to_enrich[movie_id]
                            # Add TMDb poster URL and other fields to the movie object
                            movie_obj['tmdb_poster_url'] = enriched_movie.get('tmdb_poster_url')
                            movie_obj['tmdb_backdrop_url'] = enriched_movie.get('tmdb_backdrop_url')
                            movie_obj['vote_average'] = enriched_movie.get('vote_average')
            except Exception as e:
                print(f"Error getting content-based recommendations: {e}")
            
            # Get TMDb similar movies if TMDb ID is available
            tmdb_similar_movies = []
            if 'tmdb_id' in movie_data:
                try:
                    tmdb_similar_movies = get_tmdb_similar_movies(movie_id, limit=6)
                except Exception as e:
                    print(f"Error getting TMDb similar movies: {e}")
            
            # Decide which recommendation method to show based on URL parameter
            rec_method = request.args.get('rec', 'content')
            if rec_method == 'tmdb' and tmdb_similar_movies:
                similar_movies_to_display = tmdb_similar_movies
            else:
                similar_movies_to_display = similar_movies
            
            # If no recommendations are found, fallback to popular movies
            if not similar_movies_to_display:
                try:
                    # Use a more direct approach instead of service calls
                    data_loader = get_cached_data_loader()
                    fallback_df = data_loader.get_movies().head(6)
                    fallback_movies = fallback_df.to_dict('records')
                    fallback_movies = enrich_movies_list(fallback_movies)
                    
                    # Format popular movies to match recommendations format
                    similar_movies_to_display = []
                    for movie in fallback_movies:
                        similar_movies_to_display.append({
                            'movie': movie,
                            'score': 0.5  # Default similarity score
                        })
                except Exception as e:
                    print(f"Error getting fallback popular movies: {e}")
            
            # Restore original logging level
            movie_service_logger.setLevel(original_level)
            
            return render_template(
                'movie.html',
                movie=movie_data,
                similar_movies=similar_movies_to_display,
                rec_method=rec_method,
                has_tmdb_similar=len(tmdb_similar_movies) > 0,
                genres=get_unique_genres()
            )
        except Exception as e:
            # Restore original logging level in case of errors
            movie_service_logger.setLevel(original_level)
            print(f"Error in movie_detail: {e}")
            return render_template('500.html', genres=get_unique_genres()), 500

    @app.route('/movie/tmdb/<int:tmdb_id>')
    def movie_detail_by_tmdb(tmdb_id):
        """Show movie details directly from TMDb ID."""
        from services.tmdb_service import get_movie_details, get_watch_providers, get_similar_movies
        
        # Temporarily raise the logging level
        movie_service_logger = logging.getLogger('services.movie_service')
        tmdb_service_logger = logging.getLogger('services.tmdb_service')
        original_movie_level = movie_service_logger.level
        original_tmdb_level = tmdb_service_logger.level
        movie_service_logger.setLevel(logging.ERROR)
        tmdb_service_logger.setLevel(logging.ERROR)
        
        try:
            # Get movie details from TMDb - this already includes all the data we need
            movie_data = get_movie_details(tmdb_id)
            
            if not movie_data:
                # Restore logging levels
                movie_service_logger.setLevel(original_movie_level)
                tmdb_service_logger.setLevel(original_tmdb_level)
                return render_template('404.html', genres=get_unique_genres()), 404
            
            # Get watch providers (add to the existing data)
            watch_providers = get_watch_providers(tmdb_id)
            movie_data['watch_providers'] = watch_providers
            
            # Get similar movies from TMDb in a single API call
            similar_movies = get_similar_movies(tmdb_id, max_results=8)
            
            # If no similar movies are found, use a simplified fallback
            if not similar_movies:
                try:
                    # Use a more direct approach
                    data_loader = get_cached_data_loader()
                    fallback_df = data_loader.get_movies().head(6)
                    fallback_movies = fallback_df.to_dict('records')
                    enriched_movies = enrich_movies_list(fallback_movies)
                    
                    # Convert to the TMDb similar movies format
                    similar_movies = []
                    for movie in enriched_movies:
                        if 'tmdb_id' in movie:
                            similar_movies.append({
                                'id': movie.get('tmdb_id'),
                                'title': movie.get('title', ''),
                                'poster_url': movie.get('tmdb_poster_url', ''),
                                'vote_average': movie.get('vote_average'),
                                'release_date': movie.get('release_date', '')
                            })
                except Exception as e:
                    print(f"Error getting fallback popular movies for TMDb view: {e}")
            
            # Restore logging levels
            movie_service_logger.setLevel(original_movie_level)
            tmdb_service_logger.setLevel(original_tmdb_level)
            
            return render_template(
                'movie.html',
                movie=movie_data,
                similar_movies=similar_movies,
                rec_method='tmdb',
                has_tmdb_similar=True,
                genres=get_unique_genres()
            )
        except Exception as e:
            # Restore logging levels in case of error
            movie_service_logger.setLevel(original_movie_level)
            tmdb_service_logger.setLevel(original_tmdb_level)
            print(f"Error in movie_detail_by_tmdb: {e}")
            return render_template('500.html', genres=get_unique_genres()), 500

    @app.route('/genre/<genre>')
    def genre(genre):
        """Show movies in a genre."""
        # Temporarily raise the logging level
        movie_service_logger = logging.getLogger('services.movie_service')
        original_level = movie_service_logger.level
        movie_service_logger.setLevel(logging.ERROR)
        
        try:
            # Use the cached data loader for improved performance
            data_loader = get_cached_data_loader()
            
            # Get movies by genre directly rather than through service
            genre_movies_df = data_loader.get_movies_by_genre(genre)
            genre_movies_df = genre_movies_df.head(50)  # Limit to 50 movies
            
            # Convert to list of dictionaries and enrich all at once
            genre_movies = genre_movies_df.to_dict('records')
            movies = enrich_movies_list(genre_movies)
            
            # Restore original logging level
            movie_service_logger.setLevel(original_level)
            
            return render_template('genre.html', genres=get_unique_genres(), genre=genre, movies=movies)
        except Exception as e:
            # Restore original logging level in case of error
            movie_service_logger.setLevel(original_level)
            print(f"Error getting movies by genre: {e}")
            
            # Simple fallback using direct DataFrame operations
            try:
                movies_df = load_movies()
                # Use string contains for faster filtering
                genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False)].head(50).to_dict('records')
                movies = enrich_movies_list(genre_movies)
                return render_template('genre.html', genres=get_unique_genres(), genre=genre, movies=movies)
            except Exception as e:
                print(f"Error with genre fallback: {e}")
                return render_template('genre.html', genres=get_unique_genres(), genre=genre, movies=[])

    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html', genres=get_unique_genres()), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html', genres=get_unique_genres()), 500

    # Setup logging
    if not app.debug:
        # Configure logging with proper encoding for all handlers
        import sys
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
            handlers=[
                logging.FileHandler(os.path.join(app.instance_path, 'app.log'), encoding='utf-8'),
                logging.StreamHandler(stream=sys.stdout)  # Use stdout which handles Unicode better
            ]
        )
    else:
        # In debug mode, configure with a higher threshold for warnings to reduce noise
        import sys
        logging.basicConfig(
            level=logging.WARNING,  # Show only WARNING and above in debug mode
            format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
            handlers=[
                logging.FileHandler(os.path.join(app.instance_path, 'app.log'), encoding='utf-8'),
                logging.StreamHandler(stream=sys.stdout)
            ]
        )
        # Set higher threshold for specific loggers that are too verbose
        logging.getLogger('services.movie_service').setLevel(logging.ERROR)
        logging.getLogger('services.tmdb_service').setLevel(logging.ERROR)

    return app

def get_high_rated_movies_for_home(limit: int = 10, min_ratings: int = 10) -> Tuple[List[Dict], int]:
    """
    Wrapper for get_high_rated_movies that ensures consistent return format with get_all_movies.
    
    Args:
        limit: Maximum number of movies to return
        min_ratings: Minimum number of ratings required
        
    Returns:
        Tuple of (list of movie dictionaries, total count)
    """
    try:
        from services.movie_service import get_high_rated_movies
        
        # Get high rated movies
        movies_list = get_high_rated_movies(limit=limit, min_ratings=min_ratings)
        
        # Return in format consistent with get_all_movies
        return movies_list, len(movies_list)
        
    except Exception as e:
        print(f"Error in get_high_rated_movies_for_home: {e}")
        return [], 0

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 