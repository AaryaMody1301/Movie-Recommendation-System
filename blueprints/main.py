"""Main blueprint for top-level pages."""

import logging

from flask import Blueprint, render_template

import services.movie_service as movie_service

logger = logging.getLogger(__name__)
main = Blueprint("main", __name__)


@main.route("/")
def index():
    """Render the homepage using the active movie service."""
    popular_movies = []
    top_rated_movies = []
    genres = movie_service.get_unique_genres()

    try:
        popular_raw = movie_service.get_popular_movies(limit=24)
        top_rated_raw = movie_service.get_high_rated_movies(limit=24, min_ratings=5)

        unique_movies = []
        seen_ids = set()
        for movie in popular_raw + top_rated_raw:
            movie_id = movie.get("movieId")
            if movie_id not in seen_ids:
                seen_ids.add(movie_id)
                unique_movies.append(movie)

        enriched = movie_service.enrich_movies_list(unique_movies) if unique_movies else []
        enriched_by_id = {movie.get("movieId"): movie for movie in enriched}

        # Prefer enriched records, but do not hide all local content when TMDb is disabled.
        popular_movies = [enriched_by_id.get(movie["movieId"], movie) for movie in popular_raw][:8]
        top_rated_movies = [enriched_by_id.get(movie["movieId"], movie) for movie in top_rated_raw][:8]
    except Exception:
        logger.exception("Failed to build homepage movie lists")

    return render_template(
        "index.html",
        popular_movies=popular_movies,
        top_rated_movies=top_rated_movies,
        genres=genres,
    )


@main.route("/about")
def about():
    return render_template("about.html", genres=movie_service.get_unique_genres())
