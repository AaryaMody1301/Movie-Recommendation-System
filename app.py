"""
Movie Recommendation System Flask Application.

This is the main application file that initializes the Flask app,
registers blueprints, and sets up the necessary configurations.
"""
import os
import logging
import dotenv
import argparse
from flask import Flask, render_template, redirect, url_for, request, g, flash
from flask_login import LoginManager
from services.auth_service import UserAuth, get_user_by_id
from database.db import db, init_db
import services.movie_service as movie_service # Import the service module
from models.content_based import ContentBasedRecommender # Import the new recommender
from data.data_loader import DataLoader
from flask_caching import Cache # Import Cache
from typing import List, Dict, Tuple
from config import get_config 

# Load environment variables from .env file
dotenv.load_dotenv()

# Import configuration function
from config import get_config 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Cache extension
cache = Cache()

# After initializing cache, set it in movie_service
import services.movie_service as movie_service
movie_service.set_cache(cache)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Movie Recommendation System Server')
    parser.add_argument('--rebuild-embeddings', action='store_true', help='Force rebuild of embeddings instead of loading from cache')
    parser.add_argument('--max-movies', type=int, default=1000, help='Maximum number of movies to process for embeddings (default: 1000)')
    return parser.parse_args()

def create_app(test_config=None, embedding_args=None):
    """
    Create and configure the Flask application.
    
    Args:
        test_config: Test configuration dictionary
        embedding_args: Dictionary with embedding parameters
        
    Returns:
        Configured Flask application
    """
    # Parse command line arguments or use provided embedding_args
    if embedding_args is None:
        args = parse_args()
    else:
        # Create an object from dict for consistent access
        class Args:
            def __init__(self, args_dict):
                for key, value in args_dict.items():
                    setattr(self, key, value)
        args = Args(embedding_args)
        
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Load configuration from config.py based on FLASK_ENV
    config_obj = get_config()
    app.config.from_object(config_obj)

    # Apply test config overrides if provided
    if test_config:
        app.config.from_mapping(test_config)

    # Load instance/config.py (if exists, overrides defaults/env config)
    # Note: app.config.from_pyfile is loaded AFTER from_object, so instance config takes precedence
    app.config.from_pyfile('config.py', silent=True)

    # Load TMDb API key from environment if not in config
    if not app.config.get('TMDB_API_KEY'):
        # Try to load directly from environment variable as fallback
        tmdb_api_key = os.environ.get('TMDB_API_KEY')
        if tmdb_api_key:
            app.config['TMDB_API_KEY'] = tmdb_api_key
            logger.info("Loaded TMDB_API_KEY from environment variable.")
        else:
            logger.warning("TMDB_API_KEY is not set in the configuration. TMDb features will not work.")

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating instance or upload folders: {e}")
        # Depending on severity, you might want to exit or handle differently
        pass

    # Initialize database
    try:
        init_db(app)
        db.init_app(app)
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        # Handle DB initialization error

    # Initialize Flask extensions
    cache.init_app(app)
    logger.info(f"Cache initialized with type: {app.config.get('CACHE_TYPE', 'Not Set')}") # Use get() for safety

    # Defensive: Ensure max_movies is a valid positive integer
    if not hasattr(args, 'max_movies') or args.max_movies is None or not isinstance(args.max_movies, int) or args.max_movies <= 0:
        logger.warning(f"Invalid max_movies value: {getattr(args, 'max_movies', None)}. Using default of 1000.")
        args.max_movies = 1000

    # Initialize DataLoader and Recommender ONCE at app startup, store on app object
    try:
        logger.info("Initializing DataLoader (singleton)...")
        app.data_loader = DataLoader(
            movies_path=app.config['MOVIES_CSV'],
            ratings_path=app.config.get('RATINGS_CSV')
        )
        logger.info("DataLoader initialized.")

        logger.info("Initializing ContentBasedRecommender (singleton)...")
        app.recommender = ContentBasedRecommender()
        movies_df = app.data_loader.get_movies()
        if all(col in movies_df.columns for col in ['movieId', 'title', 'genres', 'clean_title', 'overview']):
            app.recommender.fit(
                movies_df,
                max_items=args.max_movies,
                force_rebuild=args.rebuild_embeddings
            )
            logger.info("ContentBasedRecommender initialized and fitted.")
        else:
            logger.error("Movies DataFrame missing required columns for ContentBasedRecommender. Cannot fit.")
            app.recommender = None
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}. Please ensure '{app.config['MOVIES_CSV']}' exists.")
        app.data_loader = None
        app.recommender = None
    except Exception as e:
        logger.error(f"Error initializing DataLoader or Recommender: {e}")
        app.data_loader = None
        app.recommender = None

    # Setup login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login' # Assuming login route is in an 'auth' blueprint
    login_manager.login_message_category = 'info'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        # Convert to integer since Flask-Login uses string IDs
        try:
            user_data = get_user_by_id(int(user_id))
            if user_data:
                return UserAuth(user_data)
        except ValueError:
            logger.warning(f"Invalid user_id format: {user_id}")
        except Exception as e:
            logger.error(f"Error loading user {user_id}: {e}")
        return None

    # Request setup/teardown
    @app.before_request
    def before_request():
        # Assign app-level singletons to g for request context
        g.data_loader = getattr(app, 'data_loader', None)
        g.recommender = getattr(app, 'recommender', None)

    # Basic routes
    @app.route('/')
    def index():
        """Render the homepage using data from movie_service."""
        popular_movies = []
        top_rated_movies = []
        genres = []

        try:
            # Get popular and top-rated movies from the service
            # Using the actual implemented logic now
            popular_movies_raw = movie_service.get_popular_movies(limit=24)
            top_rated_movies_raw = movie_service.get_high_rated_movies(limit=24, min_ratings=5) # Example min_ratings

            # Enrich with TMDb data in batches
            all_movies_to_enrich = popular_movies_raw + top_rated_movies_raw
            # Deduplicate based on movieId if necessary before enrichment
            seen_ids = set()
            unique_movies_to_enrich = []
            for movie in all_movies_to_enrich:
                if movie['movieId'] not in seen_ids:
                    unique_movies_to_enrich.append(movie)
                    seen_ids.add(movie['movieId'])

            if unique_movies_to_enrich:
                all_enriched = movie_service.enrich_movies_list(unique_movies_to_enrich)
            else:
                all_enriched = []

            # Create lookup for enriched data
            enriched_lookup = {movie['movieId']: movie for movie in all_enriched}

            # Reconstruct lists with enriched data, filter for posters, limit to 8
            popular_movies = [
                enriched_lookup[movie['movieId']]
                for movie in popular_movies_raw
                if movie['movieId'] in enriched_lookup and enriched_lookup[movie['movieId']].get('tmdb_poster_url')
            ][:8]

            top_rated_movies = [
                enriched_lookup[movie['movieId']]
                for movie in top_rated_movies_raw
                if movie['movieId'] in enriched_lookup and enriched_lookup[movie['movieId']].get('tmdb_poster_url')
            ][:8]

            # Get unique genres
            genres = movie_service.get_unique_genres()

        except Exception as e:
            # Use app logger instead of print
            logger.error(f"Error fetching data for homepage: {e}", exc_info=True)
            # Optionally, set a flash message for the user
            # flash("Could not load movie data for the homepage. Please try again later.", "error")

        return render_template(
            'index.html',
            popular_movies=popular_movies,
            top_rated_movies=top_rated_movies,
            genres=genres
        )

    @app.route('/search')
    def search():
        """Search for movies using movie_service."""
        query = request.args.get('query', '')
        page = request.args.get('page', 1, type=int)
        per_page = 20 # Define items per page for search results

        movies = []
        total = 0
        genres = movie_service.get_unique_genres() # Get genres for layout

        if not query:
            # Render empty search page if no query
            return render_template('search.html', movies=movies, query=query, genres=genres, pagination=None)

        try:
            # Use movie_service to search (handles pagination internally)
            search_results, total = movie_service.search_movies(query, page=page, per_page=per_page)

            # Enrich search results with TMDb data
            if search_results:
                movies = movie_service.enrich_movies_list(search_results)
            else:
                movies = []
                flash(f'No movies found matching "{query}". Try another search term.', 'info')

            # Basic pagination object (can be replaced with Flask-SQLAlchemy pagination or similar)
            pagination = {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }

        except Exception as e:
            logger.error(f"Error searching movies for '{query}': {e}", exc_info=True)
            flash(f"Error searching for movies: {e}", "danger")
            pagination = None # No pagination if error

        return render_template('search.html', movies=movies, query=query, genres=genres, pagination=pagination)

    @app.route('/movie/<int:movie_id>')
    def movie_detail(movie_id):
        """Show movie details and recommendations."""
        movie_data = None
        content_recommendations = []
        tmdb_similar = []
        genres = movie_service.get_unique_genres() # For layout

        try:
            # Get the movie with TMDb data using the service
            movie_data = movie_service.get_movie_by_id(movie_id, with_tmdb=True)

            if not movie_data:
                logger.warning(f"Movie with ID {movie_id} not found.")
                return render_template('404.html', genres=genres), 404

            # Get content-based recommendations using the service
            try:
                # Pass the movie_id instead of title
                content_recommendations_raw = movie_service.get_content_recommendations(movie_id, top_n=6)
                # Enrich content recommendations
                if content_recommendations_raw:
                     # Ensure recommendations are dictionaries before enrichment
                     valid_recommendations = [rec for rec in content_recommendations_raw if isinstance(rec, dict) and 'movie' in rec]
                     if valid_recommendations:
                        # Assuming the recommendation dict structure includes 'movie' key with details
                        movies_to_enrich = [rec['movie'] for rec in valid_recommendations]
                        enriched_movies = movie_service.enrich_movies_list(movies_to_enrich)
                        # Map enriched data back (this part needs careful implementation based on structure)
                        enriched_map = {m['movieId']: m for m in enriched_movies}
                        content_recommendations = []
                        for rec in valid_recommendations:
                            movie_rec_id = rec['movie'].get('movieId')
                            if movie_rec_id in enriched_map:
                                rec['movie'] = enriched_map[movie_rec_id]
                                content_recommendations.append(rec)
                            else: # Append original if enrichment failed/missing ID
                                content_recommendations.append(rec)
                     else:
                        content_recommendations = []
                     # If there are recommendations without 'movie' key, log and skip
                     skipped = [rec for rec in content_recommendations_raw if isinstance(rec, dict) and 'movie' not in rec]
                     if skipped:
                         logger.warning(f"Some recommendations for movie {movie_id} lacked 'movie' key and were skipped: {skipped}")
            except Exception as rec_err:
                 logger.error(f"Error getting content recommendations for movie {movie_id}: {rec_err}", exc_info=True)

            # Get similar movies from TMDb API via service, if tmdb_id is available
            tmdb_id = movie_data.get('tmdb_id')
            if tmdb_id:
                try:
                    tmdb_similar_raw = movie_service.get_tmdb_similar_movies(tmdb_id, limit=6)
                     # Enrich TMDb similar movies (optional, they might already be rich)
                    if tmdb_similar_raw:
                         # We need to convert TMDb results (which might not have our movieId) to our format
                         # This might involve searching our DB for these TMDb IDs or just displaying TMb info
                         # For simplicity, let's assume enrichment adds necessary fields like tmdb_poster_url if missing
                         # You might need a more robust way to handle/display movies only known via TMDb
                         tmdb_similar = movie_service.enrich_movies_list(tmdb_similar_raw, with_tmdb=False) # Enrich basic fields if possible
                    else:
                         tmdb_similar = []
                except Exception as tmdb_err:
                    logger.error(f"Error getting TMDb similar movies for movie {movie_id} (TMDb ID: {tmdb_id}): {tmdb_err}", exc_info=True)
            else:
                logger.warning(f"Cannot fetch TMDb similar movies for movie {movie_id}: Missing tmdb_id.")

        except Exception as e:
            logger.error(f"Error fetching details for movie {movie_id}: {e}", exc_info=True)
            # Render a generic error page or redirect
            return render_template('500.html', genres=genres), 500

        return render_template(
            'movie_detail.html',
            movie=movie_data,
            similar_movies=content_recommendations, # Renamed variable for clarity
            tmdb_similar_movies=tmdb_similar, # Added TMDb similar
            genres=genres
        )

    @app.route('/movie/tmdb/<int:tmdb_id>')
    def movie_detail_by_tmdb(tmdb_id):
        """Show movie details based on TMDb ID."""
        movie_data = None
        genres = movie_service.get_unique_genres()
        try:
            # We need a way to find our internal movie_id from a tmdb_id, or fetch directly
            # Let's assume tmdb_service has a function or we adapt movie_service
            # Option 1: Fetch directly from TMDb service (might lack local data like ratings)
            from services.tmdb_service import get_movie_details as get_tmdb_details_direct
            movie_data_raw = get_tmdb_details_direct(tmdb_id)

            if not movie_data_raw:
                 logger.warning(f"Movie with TMDb ID {tmdb_id} not found via TMDb API.")
                 return render_template('404.html', genres=genres), 404

            # Try to find the corresponding local movie ID
            # Use the new function from movie_service
            local_movie_id = movie_service.find_local_id_from_tmdb_id(tmdb_id)
            if local_movie_id:
                # If we found a local ID, use that to get the full movie details
                logger.info(f"Found local movie ID {local_movie_id} for TMDb ID {tmdb_id}")
                # Fetch local movie data with TMDb enrichment
                movie_data = movie_service.get_movie_by_id(local_movie_id, with_tmdb=True)
            else:
                # Otherwise, use the TMDb data directly
                logger.info(f"No local movie ID found for TMDb ID {tmdb_id}, using TMDb data directly")
                movie_data = movie_data_raw 
                movie_data['tmdb_id'] = tmdb_id # Ensure tmdb_id is set

            # Fetch TMDb similar movies
            tmdb_similar = []
            try:
                 tmdb_similar_raw = movie_service.get_tmdb_similar_movies(tmdb_id, limit=6)
                 # Again, enrichment or direct display of TMDb data
                 tmdb_similar = movie_service.enrich_movies_list(tmdb_similar_raw, with_tmdb=False)
            except Exception as tmdb_err:
                 logger.error(f"Error getting TMDb similar movies for TMDb ID {tmdb_id}: {tmdb_err}", exc_info=True)

            # Content recommendations are harder here as we don't have the original title easily
            # We could try using the title from TMDb details
            content_recommendations = []
            if local_movie_id: # Use the local ID for content recommendations if we found one
                 try:
                      content_recommendations_raw = movie_service.get_content_recommendations(local_movie_id, top_n=6)
                      # Enrich as before...
                      if content_recommendations_raw:
                           valid_recommendations = [rec for rec in content_recommendations_raw if isinstance(rec, dict) and 'movie' in rec]
                           if valid_recommendations:
                                movies_to_enrich = [rec['movie'] for rec in valid_recommendations]
                                enriched_movies = movie_service.enrich_movies_list(movies_to_enrich)
                                enriched_map = {m['movieId']: m for m in enriched_movies}
                                content_recommendations = []
                                for rec in valid_recommendations:
                                    movie_rec_id = rec['movie'].get('movieId')
                                    if movie_rec_id in enriched_map:
                                        rec['movie'] = enriched_map[movie_rec_id]
                                        content_recommendations.append(rec)
                                    else:
                                        content_recommendations.append(rec)
                           else:
                                content_recommendations = []
                      else:
                           content_recommendations = []
                 except Exception as rec_err:
                      logger.error(f"Error getting content recommendations for TMDb ID {tmdb_id}: {rec_err}")
            else:
                logger.warning(f"Cannot get content recommendations for TMDb movie: Missing required ID (e.g., movieId from local data).")

        except Exception as e:
            logger.error(f"Error fetching details for TMDb ID {tmdb_id}: {e}", exc_info=True)
            return render_template('500.html', genres=genres), 500

        return render_template(
            'movie_detail.html', # Reusing the template
            movie=movie_data,
            similar_movies=content_recommendations,
            tmdb_similar_movies=tmdb_similar,
            genres=genres
        )

    @app.route('/genre/<genre>')
    def genre(genre):
        """Show movies for a specific genre."""
        page = request.args.get('page', 1, type=int)
        per_page = 24
        sort_by = request.args.get('sort_by', 'title')
        sort_order = request.args.get('sort_order', 'asc')

        movies = []
        total = 0
        all_genres = movie_service.get_unique_genres()

        # Validate genre?
        # if genre not in all_genres:
        #     return render_template('404.html', genres=all_genres), 404

        try:
            movies_raw, total = movie_service.get_movies_by_genre(
                genre, page=page, per_page=per_page,
                sort_by=sort_by, sort_order=sort_order
            )

            # Enrich results
            if movies_raw:
                movies = movie_service.enrich_movies_list(movies_raw)
            else:
                movies = []

            # Pagination object
            pagination = {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page,
                'sort_by': sort_by,
                'sort_order': sort_order
            }

        except Exception as e:
            logger.error(f"Error fetching movies for genre '{genre}': {e}", exc_info=True)
            # flash(f"Error loading movies for genre {genre}.", "error")
            pagination = None

        return render_template(
            'genre.html', 
            genre=genre, 
            movies=movies, 
            genres=all_genres, 
            pagination=pagination
        )

    # Register Blueprints (Example - adapt to your structure)
    # from . import auth, user_profile # Example blueprint imports
    # app.register_blueprint(auth.bp)
    # app.register_blueprint(user_profile.bp)

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        genres = movie_service.get_unique_genres()
        return render_template('404.html', genres=genres), 404

    @app.errorhandler(500)
    def server_error(e):
        genres = movie_service.get_unique_genres()
        logger.error(f"Server Error: {e}", exc_info=True) # Log the actual error
        return render_template('500.html', genres=genres), 500

    # Custom command to initialize DB (if needed, e.g., for Flask CLI)
    # @app.cli.command('init-db')
    # def init_db_command():
    #     """Clear the existing data and create new tables."""
    #     init_db(app)
    #     click.echo('Initialized the database.')

    return app

# Note: Removed the direct execution block
# if __name__ == '__main__':
#     app = create_app()
#     app.run(debug=True) # Use Flask CLI or a WSGI server (like gunicorn) for production

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
    print("\nMovie Recommendation System running at: http://127.0.0.1:5000\n")
    app.run(debug=True)