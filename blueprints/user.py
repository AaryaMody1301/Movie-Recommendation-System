"""User profile, ratings, and watchlist routes."""

from __future__ import annotations

from flask import Blueprint, flash, jsonify, render_template, request
from flask_login import current_user, login_required

from services.movie_service import get_movie_by_id, get_unique_genres
from services.user_service import (
    add_to_watchlist,
    delete_user_rating,
    get_user_profile,
    get_user_ratings,
    get_user_watchlist,
    remove_from_watchlist,
    update_watchlist_notes,
)

user = Blueprint("user", __name__)


def _page_args(default_per_page: int = 24):
    page = max(1, request.args.get("page", 1, type=int) or 1)
    per_page = request.args.get("per_page", default_per_page, type=int) or default_per_page
    per_page = min(100, max(1, per_page))
    return page, per_page


def _json_object():
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else None


@user.route("/profile")
@login_required
def profile():
    profile_data = get_user_profile(current_user.id)
    if not profile_data:
        flash("Could not load your profile information.", "danger")
        return render_template(
            "user/profile.html",
            profile=None,
            ratings=[],
            watchlist=[],
            genres=get_unique_genres(),
        )

    ratings, _ = get_user_ratings(current_user.id, page=1, per_page=5)
    watchlist, _ = get_user_watchlist(current_user.id, page=1, per_page=5)
    return render_template(
        "user/profile.html",
        profile=profile_data,
        ratings=ratings,
        watchlist=watchlist,
        genres=get_unique_genres(),
    )


@user.route("/ratings")
@login_required
def ratings():
    page, per_page = _page_args()
    sort_by = request.args.get("sort_by", "date")
    sort_order = request.args.get("sort_order", "desc").lower()
    if sort_by not in {"date", "rating", "movie_id"}:
        sort_by = "date"
    if sort_order not in {"asc", "desc"}:
        sort_order = "desc"

    items, total = get_user_ratings(
        current_user.id,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return render_template(
        "user/ratings.html",
        ratings=items,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        total_ratings=total,
        sort_by=sort_by,
        sort_order=sort_order,
        genres=get_unique_genres(),
    )


@user.route("/watchlist")
@login_required
def watchlist():
    page, per_page = _page_args()
    sort_by = request.args.get("sort_by", "date")
    sort_order = request.args.get("sort_order", "desc").lower()
    if sort_by not in {"date", "movie_id"}:
        sort_by = "date"
    if sort_order not in {"asc", "desc"}:
        sort_order = "desc"

    items, total = get_user_watchlist(
        current_user.id,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return render_template(
        "user/watchlist.html",
        watchlist=items,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        total_items=total,
        sort_by=sort_by,
        sort_order=sort_order,
        genres=get_unique_genres(),
    )


@user.route("/api/watchlist/add", methods=["POST"])
@login_required
def api_add_to_watchlist():
    payload = _json_object()
    if payload is None:
        return jsonify({"error": "Expected a JSON object"}), 400

    try:
        movie_id = int(payload.get("movieId"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing movie ID"}), 400

    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        return jsonify({"error": "Notes must be text"}), 400
    if len(notes) > 2000:
        return jsonify({"error": "Notes must be 2000 characters or fewer"}), 400
    if not get_movie_by_id(movie_id, with_tmdb=False):
        return jsonify({"error": "Movie not found"}), 404

    result = add_to_watchlist(current_user.id, movie_id, notes)
    if result.get("success"):
        return jsonify(result), 201
    status = 409 if "already" in result.get("error", "").lower() else 400
    return jsonify(result), status


@user.route("/api/watchlist/remove", methods=["POST"])
@login_required
def api_remove_from_watchlist():
    payload = _json_object()
    if payload is None:
        return jsonify({"error": "Expected a JSON object"}), 400
    try:
        movie_id = int(payload.get("movieId"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing movie ID"}), 400

    result = remove_from_watchlist(current_user.id, movie_id)
    return (jsonify(result), 200) if result.get("success") else (jsonify(result), 404)


@user.route("/api/watchlist/update", methods=["POST"])
@login_required
def api_update_watchlist_notes():
    payload = _json_object()
    if payload is None:
        return jsonify({"error": "Expected a JSON object"}), 400
    try:
        movie_id = int(payload.get("movieId"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing movie ID"}), 400

    notes = payload.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 2000:
        return jsonify({"error": "Notes must be text up to 2000 characters"}), 400

    result = update_watchlist_notes(current_user.id, movie_id, notes)
    return (jsonify(result), 200) if result.get("success") else (jsonify(result), 404)


@user.route("/api/ratings/remove", methods=["POST"])
@login_required
def api_remove_rating():
    payload = _json_object()
    if payload is None:
        return jsonify({"error": "Expected a JSON object"}), 400
    try:
        movie_id = int(payload.get("movieId"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing movie ID"}), 400

    result = delete_user_rating(current_user.id, movie_id)
    return (jsonify(result), 200) if result.get("success") else (jsonify(result), 404)
