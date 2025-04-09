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
            # First try using the standard service functions
            popular_movies_result = get_all_movies(page=1, per_page=24, sort_by='popularity', sort_order='desc')
            if popular_movies_result and len(popular_movies_result) == 2:
                popular_movies, _ = popular_movies_result
            
            top_rated_result = get_high_rated_movies_for_home(limit=24, min_ratings=5)
            if top_rated_result and len(top_rated_result) == 2:
                top_rated_movies, _ = top_rated_result
            
            # Enrich with TMDb data
            if popular_movies:
                popular_movies = enrich_movies_list(popular_movies)
                popular_movies = [movie for movie in popular_movies if movie.get('tmdb_poster_url')][:8]
                
            if top_rated_movies:
                top_rated_movies = enrich_movies_list(top_rated_movies)
                top_rated_movies = [movie for movie in top_rated_movies if movie.get('tmdb_poster_url')][:8]
        except Exception as e:
            print(f"Error getting movies via services: {e}")
        
        # If we don't have enough movies, try a direct approach
        if len(popular_movies) < 8 or len(top_rated_movies) < 8:
            try:
                # Get movies directly from DataLoader
                from data.data_loader import DataLoader
                data_loader = DataLoader(movies_path='data/movies.csv')
                
                # Get all movies for processing
                all_movies_df = data_loader.get_movies()
                
                # If popular movies are missing, use movies sorted by title
                if len(popular_movies) < 8:
                    try:
                        direct_movies = all_movies_df.sort_values('title').head(16).to_dict('records')
                        direct_movies = enrich_movies_list(direct_movies)
                        popular_movies = [m for m in direct_movies if m.get('tmdb_poster_url')][:8]
                    except Exception as e:
                        print(f"Error getting direct popular movies: {e}")
                
                # If top rated movies are missing, use the next 8 movies
                if len(top_rated_movies) < 8:
                    try:
                        if len(popular_movies) >= 8:
                            # Use different movies than popular_movies
                            remaining = all_movies_df.sample(16).to_dict('records')
                            remaining = enrich_movies_list(remaining)
                            top_rated_movies = [m for m in remaining if m.get('tmdb_poster_url')][:8]
                        else:
                            # Just use some random movies
                            random_movies = all_movies_df.sample(16).to_dict('records')
                            random_movies = enrich_movies_list(random_movies)
                            if len(popular_movies) == 0:
                                # Split between popular and top rated
                                movies_with_posters = [m for m in random_movies if m.get('tmdb_poster_url')]
                                mid = len(movies_with_posters) // 2
                                popular_movies = movies_with_posters[:mid]
                                top_rated_movies = movies_with_posters[mid:mid+8]
                            else:
                                top_rated_movies = [m for m in random_movies if m.get('tmdb_poster_url')][:8]
                    except Exception as e:
                        print(f"Error getting direct top rated movies: {e}")
            except Exception as e:
                print(f"Error using direct DataLoader approach: {e}")
                
                # Final fallback - use simple_app if everything else fails
                if len(popular_movies) < 8 or len(top_rated_movies) < 8:
                    try:
                        from simple_app import load_movies
                        movies_df = load_movies()
                        fallback_movies = movies_df.head(16).to_dict('records')
                        fallback_movies = enrich_movies_list(fallback_movies)
                        
                        if len(popular_movies) < 8:
                            popular_movies = fallback_movies[:8]
                        if len(top_rated_movies) < 8:
                            top_rated_movies = fallback_movies[8:16] if len(fallback_movies) > 8 else fallback_movies[:8]
                    except Exception as e:
                        print(f"Error with final fallback: {e}")
        
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
        from flask import request
        from services.movie_service import search_movies, enrich_movies_list
        
        query = request.args.get('query', '')
        
        if not query:
            return render_template('search.html', movies=[], query='', genres=get_unique_genres())
        
        try:
            # Find movies matching the query using our service
            matching_movies, total = search_movies(query, page=1, per_page=20)
            
            # Enrich with TMDb data
            movies = enrich_movies_list(matching_movies)
        except Exception as e:
            print(f"Error searching movies: {e}")
            # Fallback to simple search
            matching_movies = movies_df[movies_df['title'].str.contains(query, case=False)]
            movies = matching_movies.head(20).to_dict('records')
        
        return render_template('search.html', movies=movies, query=query, genres=get_unique_genres())

    @app.route('/movie/<int:movie_id>')
    def movie_detail(movie_id):
        """Show movie details with TMDb enrichment."""
        # Get the movie with TMDb data
        movie_data = get_movie_by_id(movie_id, with_tmdb=True)
        
        if not movie_data:
            return render_template('404.html', genres=get_unique_genres()), 404
        
        # Get content-based recommendations
        similar_movies = []
        try:
            similar_movies = get_movie_recommendations(movie_data['title'], n=6)
            # Enrich similar movies with TMDb data
            for recommendation in similar_movies:
                if 'movie' in recommendation:
                    movie_obj = recommendation['movie']
                    # Get TMDb data for the movie
                    enriched_movie = get_movie_by_id(movie_obj['movieId'], with_tmdb=True)
                    if enriched_movie:
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
                # Get popular movies as fallback
                popular_movies, _ = get_all_movies(page=1, per_page=6, sort_by='popularity', sort_order='desc')
                popular_movies = enrich_movies_list(popular_movies)
                
                # Format popular movies to match recommendations format
                similar_movies_to_display = []
                for movie in popular_movies:
                    similar_movies_to_display.append({
                        'movie': movie,
                        'score': 0.5  # Default similarity score
                    })
            except Exception as e:
                print(f"Error getting fallback popular movies: {e}")
        
        return render_template(
            'movie.html',
            movie=movie_data,
            similar_movies=similar_movies_to_display,
            rec_method=rec_method,
            has_tmdb_similar=len(tmdb_similar_movies) > 0,
            genres=get_unique_genres()
        )

    @app.route('/movie/tmdb/<int:tmdb_id>')
    def movie_detail_by_tmdb(tmdb_id):
        """Show movie details directly from TMDb ID."""
        from services.tmdb_service import get_movie_details, get_watch_providers, get_similar_movies
        
        # Get movie details from TMDb
        movie_data = get_movie_details(tmdb_id)
        
        if not movie_data:
            return render_template('404.html', genres=get_unique_genres()), 404
        
        # Get watch providers
        watch_providers = get_watch_providers(tmdb_id)
        movie_data['watch_providers'] = watch_providers
        
        # Get similar movies from TMDb
        similar_movies = get_similar_movies(tmdb_id, max_results=8)
        
        # If no similar movies are found, fallback to popular movies
        if not similar_movies:
            try:
                from services.movie_service import get_popular_movies
                
                # Get popular movies as fallback
                popular_movies = get_popular_movies(limit=6)
                popular_movies = enrich_movies_list(popular_movies)
                
                # Convert to the TMDb similar movies format
                similar_movies = []
                for movie in popular_movies:
                    if 'tmdb_id' in movie:
                        tmdb_details = get_movie_details(movie['tmdb_id'])
                        if tmdb_details:
                            similar_movies.append({
                                'id': tmdb_details['id'],
                                'title': tmdb_details['title'],
                                'poster_url': tmdb_details['poster_url'],
                                'vote_average': tmdb_details.get('vote_average'),
                                'release_date': tmdb_details.get('release_date')
                            })
            except Exception as e:
                print(f"Error getting fallback popular movies for TMDb view: {e}")
        
        return render_template(
            'movie.html',
            movie=movie_data,
            similar_movies=similar_movies,
            rec_method='tmdb',
            has_tmdb_similar=True,
            genres=get_unique_genres()
        )

    @app.route('/genre/<genre>')
    def genre(genre):
        """Show movies in a genre."""
        from services.movie_service import get_movies_by_genre, enrich_movies_list
        
        try:
            # Get movies by genre using our service
            genre_movies, total = get_movies_by_genre(genre, page=1, per_page=50)
            
            # Enrich with TMDb data
            movies = enrich_movies_list(genre_movies)
        except Exception as e:
            print(f"Error getting movies by genre: {e}")
            # Fallback to simple search
            genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False)]
            movies = genre_movies.head(50).to_dict('records')
        
        return render_template('genre.html', genres=get_unique_genres(), genre=genre, movies=movies)

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