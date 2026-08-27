"""Routes for personalized and movie-based recommendations."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, flash, jsonify, render_template, request
from flask_login import current_user, login_required

from services.movie_service import get_movie_by_id, get_unique_genres
from services.recommendation_service import (
    VALID_STRATEGIES,
    get_recommendation_explanation,
    get_recommendations_for_movie,
    get_recommendations_for_user,
    get_similar_movies,
    rate_movie,
)

recommendations = Blueprint("recommendations", __name__)


def _page_args(default_per_page: int):
    page = max(1, request.args.get("page", 1, type=int) or 1)
    per_page = request.args.get("per_page", default_per_page, type=int) or default_per_page
    return page, min(100, max(1, per_page))


def _strategy() -> str:
    value = request.args.get("strategy", "weighted")
    return value if value in VALID_STRATEGIES else "weighted"


@recommendations.route("/recommendations")
@login_required
def user_recommendations():
    page, per_page = _page_args(24)
    strategy = _strategy()
    items, total = get_recommendations_for_user(
        current_user.id,
        page=page,
        per_page=per_page,
        strategy=strategy,
    )
    if total == 0:
        flash("No personalized recommendations available yet.", "info")

    return render_template(
        "recommendations/personal.html",
        recommendations=items,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        total_recommendations=total,
        strategy=strategy,
        genres=get_unique_genres(),
    )


@recommendations.route("/movie/<int:movie_id>/similar")
def similar_movies(movie_id):
    movie = get_movie_by_id(movie_id, with_tmdb=False)
    if not movie:
        abort(404)

    page, per_page = _page_args(12)
    items, total = get_similar_movies(movie_id, page=page, per_page=per_page)
    return render_template(
        "recommendations/similar.html",
        movie=movie,
        recommendations=items,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        total_recommendations=total,
        genres=get_unique_genres(),
    )


@recommendations.route("/movie/<int:movie_id>/personalized")
@login_required
def personalized_movie_recommendations(movie_id):
    movie = get_movie_by_id(movie_id, with_tmdb=False)
    if not movie:
        abort(404)

    page, per_page = _page_args(12)
    strategy = _strategy()
    items, total = get_recommendations_for_movie(
        movie_id,
        user_id=current_user.id,
        page=page,
        per_page=per_page,
        strategy=strategy,
    )
    return render_template(
        "recommendations/personalized.html",
        movie=movie,
        recommendations=items,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        total_recommendations=total,
        strategy=strategy,
        genres=get_unique_genres(),
    )


@recommendations.route("/recommendation/<int:movie_id>/explanation")
def recommendation_explanation(movie_id):
    movie = get_movie_by_id(movie_id, with_tmdb=False)
    if not movie:
        abort(404)

    source_id = request.args.get("source_id", type=int)
    source_movie = get_movie_by_id(source_id, with_tmdb=False) if source_id else None
    user_id = current_user.id if current_user.is_authenticated else None
    explanation = get_recommendation_explanation(movie_id, user_id, source_id)
    return render_template(
        "recommendations/explanation.html",
        movie=movie,
        source_movie=source_movie,
        explanation=explanation,
        genres=get_unique_genres(),
    )


@recommendations.route("/api/rate", methods=["POST"])
@login_required
def api_rate_movie():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    try:
        movie_id = int(payload.get("movieId"))
        rating = float(payload.get("rating"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing movie ID/rating"}), 400

    if not 0.5 <= rating <= 5.0:
        return jsonify({"error": "Rating must be between 0.5 and 5.0"}), 400
    if not get_movie_by_id(movie_id, with_tmdb=False):
        return jsonify({"error": "Movie not found"}), 404

    try:
        if rate_movie(current_user.id, movie_id, rating):
            return jsonify({"success": True, "message": "Rating saved"})
        return jsonify({"error": "Failed to save rating"}), 400
    except Exception:
        current_app.logger.exception("Failed to rate movie")
        return jsonify({"error": "An error occurred"}), 500


@recommendations.route("/api/recommendations", methods=["GET"])
@login_required
def api_get_recommendations():
    limit = request.args.get("limit", 10, type=int) or 10
    limit = min(100, max(1, limit))
    strategy = _strategy()
    try:
        items, _ = get_recommendations_for_user(
            current_user.id,
            page=1,
            per_page=limit,
            strategy=strategy,
        )
        return jsonify(items)
    except Exception:
        current_app.logger.exception("Failed to get recommendations")
        return jsonify({"error": "An error occurred"}), 500
