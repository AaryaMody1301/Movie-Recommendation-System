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
        # Get popular movies with TMDb data
        try:
            # Get more movies than needed to ensure we have enough with posters
            popular_movies, _ = get_all_movies(page=1, per_page=24, sort_by='popularity', sort_order='desc')
            top_rated_movies, _ = get_high_rated_movies(limit=24, min_ratings=5)
            
            # Enrich with TMDb data
            popular_movies = enrich_movies_list(popular_movies)
            top_rated_movies = enrich_movies_list(top_rated_movies)
            
            # Filter for movies with poster URLs and limit to 8
            popular_movies = [movie for movie in popular_movies if movie.get('tmdb_poster_url') and 'placeholder' not in movie.get('tmdb_poster_url', '')][:8]
            top_rated_movies = [movie for movie in top_rated_movies if movie.get('tmdb_poster_url') and 'placeholder' not in movie.get('tmdb_poster_url', '')][:8]
            
            # If we don't have enough movies with posters, fill with the rest
            if len(popular_movies) < 8:
                remaining_movies = [movie for movie in enrich_movies_list(popular_movies, with_tmdb=True) 
                                   if movie not in popular_movies][:8-len(popular_movies)]
                popular_movies.extend(remaining_movies)
                
            if len(top_rated_movies) < 8:
                remaining_movies = [movie for movie in enrich_movies_list(top_rated_movies, with_tmdb=True) 
                                   if movie not in top_rated_movies][:8-len(top_rated_movies)]
                top_rated_movies.extend(remaining_movies)
        except Exception as e:
            print(f"Error getting movies for homepage: {e}")
            # Fallback to simple movies from movies_df
            popular_movies = movies_df.head(8).to_dict('records')
            top_rated_movies = popular_movies
        
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
            handlers=[
                logging.FileHandler(os.path.join(app.instance_path, 'app.log')),
                logging.StreamHandler()
            ]
        )

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 