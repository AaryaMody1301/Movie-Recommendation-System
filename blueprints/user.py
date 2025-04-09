"""
User blueprint for user profiles and watchlists.
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, abort
from flask_login import login_required, current_user
from services.user_service import (
    get_user_profile,
    get_user_ratings,
    get_user_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    update_watchlist_notes
)
from services.movie_service import get_movie_by_id

user = Blueprint('user', __name__)


@user.route('/profile')
@login_required
def profile():
    """User profile page."""
    # Get user profile data
    profile_data = get_user_profile(current_user.id)
    
    # Get recent ratings
    ratings = get_user_ratings(current_user.id, limit=5)
    
    # Get watchlist
    watchlist = get_user_watchlist(current_user.id, limit=5)
    
    return render_template(
        'user/profile.html',
        profile=profile_data,
        ratings=ratings,
        watchlist=watchlist
    )


@user.route('/ratings')
@login_required
def ratings():
    """User ratings page."""
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    
    # Sorting parameters
    sort_by = request.args.get('sort_by', 'date')
    sort_order = request.args.get('sort_order', 'desc')
    
    # Get user ratings with pagination
    ratings, total = get_user_ratings(
        current_user.id,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'user/ratings.html',
        ratings=ratings,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_ratings=total,
        sort_by=sort_by,
        sort_order=sort_order
    )


@user.route('/watchlist')
@login_required
def watchlist():
    """User watchlist page."""
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    
    # Sorting parameters
    sort_by = request.args.get('sort_by', 'date')
    sort_order = request.args.get('sort_order', 'desc')
    
    # Get user watchlist with pagination
    watchlist_items, total = get_user_watchlist(
        current_user.id,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page
    
    return render_template(
        'user/watchlist.html',
        watchlist=watchlist_items,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_items=total,
        sort_by=sort_by,
        sort_order=sort_order
    )


@user.route('/api/watchlist/add', methods=['POST'])
@login_required
def api_add_to_watchlist():
    """API endpoint for adding a movie to the watchlist."""
    movie_id = request.json.get('movieId')
    notes = request.json.get('notes', '')
    
    if not movie_id:
        return jsonify({'error': 'Missing movie ID'}), 400
    
    try:
        # Convert to proper type
        movie_id = int(movie_id)
        
        # Check if movie exists
        movie = get_movie_by_id(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        
        # Add to watchlist
        success = add_to_watchlist(current_user.id, movie_id, notes)
        
        if success:
            return jsonify({'success': True, 'message': 'Added to watchlist'})
        else:
            return jsonify({'error': 'Failed to add to watchlist'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid movie ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@user.route('/api/watchlist/remove', methods=['POST'])
@login_required
def api_remove_from_watchlist():
    """API endpoint for removing a movie from the watchlist."""
    movie_id = request.json.get('movieId')
    
    if not movie_id:
        return jsonify({'error': 'Missing movie ID'}), 400
    
    try:
        # Convert to proper type
        movie_id = int(movie_id)
        
        # Remove from watchlist
        success = remove_from_watchlist(current_user.id, movie_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Removed from watchlist'})
        else:
            return jsonify({'error': 'Failed to remove from watchlist'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid movie ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@user.route('/api/watchlist/update', methods=['POST'])
@login_required
def api_update_watchlist_notes():
    """API endpoint for updating watchlist item notes."""
    movie_id = request.json.get('movieId')
    notes = request.json.get('notes', '')
    
    if not movie_id:
        return jsonify({'error': 'Missing movie ID'}), 400
    
    try:
        # Convert to proper type
        movie_id = int(movie_id)
        
        # Update notes
        success = update_watchlist_notes(current_user.id, movie_id, notes)
        
        if success:
            return jsonify({'success': True, 'message': 'Notes updated'})
        else:
            return jsonify({'error': 'Failed to update notes'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid movie ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500 