"""
Movies blueprint for browsing and searching movies.
"""
from flask import Blueprint, render_template, request, jsonify, abort
from services.movie_service import (
    get_all_movies, 
    get_movie_by_id, 
    search_movies, 
    get_movies_by_genre,
    get_popular_movies,
    get_high_rated_movies
)
from services.recommendation_service import get_unique_genres

movies = Blueprint('movies', __name__)


@movies.route('/browse')
def browse():
    """Browse all movies page."""
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    
    # Filtering and sorting parameters
    genre = request.args.get('genre', '')
    sort_by = request.args.get('sort_by', 'title')
    sort_order = request.args.get('sort_order', 'asc')
    
    # Get movies with pagination
    if genre:
        movies_data, total = get_movies_by_genre(
            genre, page=page, per_page=per_page, 
            sort_by=sort_by, sort_order=sort_order
        )
    else:
        movies_data, total = get_all_movies(
            page=page, per_page=per_page, 
            sort_by=sort_by, sort_order=sort_order
        )
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    # Get unique genres for filtering
    genres = get_unique_genres()
    
    return render_template(
        'movies/browse.html',
        movies=movies_data,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_movies=total,
        current_genre=genre,
        genres=genres,
        sort_by=sort_by,
        sort_order=sort_order
    )


@movies.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    """Movie detail page."""
    movie = get_movie_by_id(movie_id)
    
    if not movie:
        abort(404)
    
    # Split genres for display
    movie['genres_list'] = movie['genres'].split('|') if movie['genres'] else []
    
    return render_template(
        'movies/detail.html', 
        movie=movie
    )


@movies.route('/genre/<genre>')
def genre(genre):
    """Movies by genre page."""
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    
    # Get movies by genre with pagination
    movies_data, total = get_movies_by_genre(genre, page=page, per_page=per_page)
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'movies/genre.html',
        genre=genre,
        movies=movies_data,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_movies=total
    )


@movies.route('/search')
def search():
    """Search movies page."""
    query = request.args.get('query', '')
    
    if not query:
        return render_template('movies/search.html', query='', movies=[], total=0)
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    
    # Search movies with pagination
    movies_data, total = search_movies(query, page=page, per_page=per_page)
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'movies/search.html',
        query=query,
        movies=movies_data,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_movies=total
    )


@movies.route('/api/search')
def api_search():
    """API endpoint for searching movies (for autocomplete)."""
    query = request.args.get('query', '')
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify([])
    
    # Search movies with limit
    movies_data, _ = search_movies(query, page=1, per_page=limit)
    
    # Return simplified movie data for autocomplete
    result = [
        {
            'id': movie['movieId'],
            'title': movie['title'],
            'year': movie.get('year', ''),
            'genres': movie['genres']
        }
        for movie in movies_data
    ]
    
    return jsonify(result) 