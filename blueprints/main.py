"""
Main blueprint for home and about pages.
"""
from flask import Blueprint, render_template, current_app, flash
from services.movie_service import get_popular_movies, get_high_rated_movies
from services.recommendation_service import get_unique_genres

main = Blueprint('main', __name__)


@main.route('/')
def index():
    """Render the homepage."""
    # Get popular and highly rated movies for the homepage
    popular_movies = get_popular_movies(limit=8)
    top_rated_movies = get_high_rated_movies(limit=8)
    
    # Get unique genres for navigation
    genres = get_unique_genres()
    
    if not popular_movies:
        flash('No popular movies available at this time.', 'info')
    if not top_rated_movies:
        flash('No top rated movies available at this time.', 'info')
    
    return render_template(
        'index.html',
        popular_movies=popular_movies,
        top_rated_movies=top_rated_movies,
        genres=genres
    )


@main.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')