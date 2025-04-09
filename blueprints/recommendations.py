"""
Recommendations blueprint for personalized recommendations.
"""
from flask import Blueprint, render_template, request, jsonify, abort, current_app
from flask_login import login_required, current_user
from services.recommendation_service import (
    get_recommendations_for_user,
    get_recommendations_for_movie,
    get_similar_movies,
    rate_movie,
    get_user_ratings,
    get_recommendation_explanation
)
from services.movie_service import get_movie_by_id

recommendations = Blueprint('recommendations', __name__)


@recommendations.route('/recommendations')
@login_required
def user_recommendations():
    """Personalized recommendations for the current user."""
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    
    # Strategy parameters (weighted or rank)
    strategy = request.args.get('strategy', 'weighted')
    
    # Get recommendations for user
    recommendations, total = get_recommendations_for_user(
        current_user.id, 
        page=page, 
        per_page=per_page, 
        strategy=strategy
    )
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'recommendations/personal.html',
        recommendations=recommendations,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_recommendations=total,
        strategy=strategy
    )


@recommendations.route('/movie/<int:movie_id>/similar')
def similar_movies(movie_id):
    """Similar movies recommendations."""
    # Get the movie
    movie = get_movie_by_id(movie_id)
    if not movie:
        abort(404)
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 12, type=int)
    
    # Get similar movies
    similar, total = get_similar_movies(movie_id, page=page, per_page=per_page)
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'recommendations/similar.html',
        movie=movie,
        recommendations=similar,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_recommendations=total
    )


@recommendations.route('/movie/<int:movie_id>/personalized')
@login_required
def personalized_movie_recommendations(movie_id):
    """Personalized recommendations based on a movie."""
    # Get the movie
    movie = get_movie_by_id(movie_id)
    if not movie:
        abort(404)
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 12, type=int)
    
    # Strategy parameters (weighted or rank)
    strategy = request.args.get('strategy', 'weighted')
    
    # Get personalized recommendations based on the movie and user
    recommendations, total = get_recommendations_for_movie(
        movie_id, 
        user_id=current_user.id,
        page=page, 
        per_page=per_page, 
        strategy=strategy
    )
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'recommendations/personalized.html',
        movie=movie,
        recommendations=recommendations,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_recommendations=total,
        strategy=strategy
    )


@recommendations.route('/recommendation/<int:movie_id>/explanation')
def recommendation_explanation(movie_id):
    """Explanation for a recommendation."""
    # Get the movie
    movie = get_movie_by_id(movie_id)
    if not movie:
        abort(404)
    
    # Source movie ID (if any)
    source_id = request.args.get('source_id', type=int)
    source_movie = get_movie_by_id(source_id) if source_id else None
    
    # User ID (if authenticated)
    user_id = current_user.id if current_user.is_authenticated else None
    
    # Get explanation
    explanation = get_recommendation_explanation(movie_id, user_id, source_id)
    
    return render_template(
        'recommendations/explanation.html',
        movie=movie,
        source_movie=source_movie,
        explanation=explanation
    )


@recommendations.route('/api/rate', methods=['POST'])
@login_required
def api_rate_movie():
    """API endpoint for rating a movie."""
    movie_id = request.json.get('movieId')
    rating = request.json.get('rating')
    
    if not movie_id or not rating:
        return jsonify({'error': 'Missing movie ID or rating'}), 400
    
    try:
        # Convert to proper types
        movie_id = int(movie_id)
        rating = float(rating)
        
        # Validate rating range
        if rating < 1.0 or rating > 5.0:
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        # Check if movie exists
        movie = get_movie_by_id(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        
        # Add or update rating
        success = rate_movie(current_user.id, movie_id, rating)
        
        if success:
            return jsonify({'success': True, 'message': 'Rating saved'})
        else:
            return jsonify({'error': 'Failed to save rating'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid movie ID or rating'}), 400
    except Exception as e:
        current_app.logger.error(f"Error rating movie: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500


@recommendations.route('/api/recommendations', methods=['GET'])
@login_required
def api_get_recommendations():
    """API endpoint for getting recommendations."""
    limit = request.args.get('limit', 10, type=int)
    strategy = request.args.get('strategy', 'weighted')
    
    try:
        recommendations, _ = get_recommendations_for_user(
            current_user.id, 
            page=1, 
            per_page=limit, 
            strategy=strategy
        )
        
        return jsonify(recommendations)
    
    except Exception as e:
        current_app.logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500 