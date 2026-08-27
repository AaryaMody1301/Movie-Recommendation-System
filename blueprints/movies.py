"""Movie browsing, search, and detail routes."""

import logging

from flask import Blueprint, flash, jsonify, render_template, request

import services.movie_service as movie_service
from services.tmdb_service import get_movie_details as get_tmdb_movie_details

logger = logging.getLogger(__name__)
movies = Blueprint("movies", __name__)


def _genres():
    return movie_service.get_unique_genres()


def _pagination(page, per_page, total, **extra):
    data = {
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": (total + per_page - 1) // per_page if per_page else 0,
    }
    data.update(extra)
    return data


def _tmdb_for_template(movie):
    """Normalize a TMDb-only record to fields understood by current templates."""
    if not movie:
        return None
    normalized = dict(movie)
    normalized["tmdb_id"] = normalized.get("tmdb_id") or normalized.get("id")
    normalized["tmdb_poster_url"] = normalized.get("tmdb_poster_url") or normalized.get("poster_url")
    normalized["tmdb_backdrop_url"] = normalized.get("tmdb_backdrop_url") or normalized.get("backdrop_url")

    genres = normalized.get("genres")
    if isinstance(genres, list):
        normalized["genres"] = "|".join(
            genre.get("name", "") if isinstance(genre, dict) else str(genre)
            for genre in genres
            if genre
        )
    return normalized


def _content_recommendations(movie_id, top_n=6):
    """Return recommendations in the nested shape expected by movie_detail.html."""
    raw = movie_service.get_content_recommendations(movie_id, top_n=top_n) or []
    normalized = []

    nested = [rec for rec in raw if isinstance(rec, dict) and isinstance(rec.get("movie"), dict)]
    if nested:
        movies_to_enrich = [rec["movie"] for rec in nested]
        enriched = movie_service.enrich_movies_list(movies_to_enrich) if movies_to_enrich else []
        enriched_by_id = {movie.get("movieId"): movie for movie in enriched}
        for rec in nested:
            rec = dict(rec)
            movie = dict(rec["movie"])
            rec["movie"] = enriched_by_id.get(movie.get("movieId"), movie)
            normalized.append(rec)
        return normalized

    # Fallbacks from movie_service are plain movie dictionaries. Preserve them
    # instead of silently discarding them.
    flat = [rec for rec in raw if isinstance(rec, dict) and rec.get("movieId") is not None]
    enriched = movie_service.enrich_movies_list(flat) if flat else []
    enriched_by_id = {movie.get("movieId"): movie for movie in enriched}
    for movie in flat:
        normalized.append({"movie": enriched_by_id.get(movie.get("movieId"), movie)})
    return normalized


@movies.route("/browse")
def browse():
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(max(request.args.get("per_page", 24, type=int), 1), 100)
    genre = request.args.get("genre", "").strip()
    sort_by = request.args.get("sort_by", "title")
    sort_order = request.args.get("sort_order", "asc")

    if genre:
        movie_rows, total = movie_service.get_movies_by_genre(
            genre,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    else:
        movie_rows, total = movie_service.get_all_movies(
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    display_movies = movie_service.enrich_movies_list(movie_rows) if movie_rows else []
    return render_template(
        "browse.html",
        movies=display_movies,
        genres=_genres(),
        current_genre=genre,
        sort_by=sort_by,
        sort_order=sort_order,
        pagination=_pagination(page, per_page, total, sort_by=sort_by, sort_order=sort_order),
    )


@movies.route("/search")
def search():
    query = request.args.get("query", "").strip()
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = 20
    genres = _genres()

    if not query:
        return render_template(
            "search.html",
            movies=[],
            query="",
            genres=genres,
            pagination=None,
        )

    try:
        results, total = movie_service.search_movies(query, page=page, per_page=per_page)
        display_movies = movie_service.enrich_movies_list(results) if results else []
        if not display_movies:
            flash(f'No movies found matching "{query}".', "info")
        pagination = _pagination(page, per_page, total)
    except Exception:
        logger.exception("Movie search failed for %r", query)
        flash("Movie search failed. Please try again.", "danger")
        display_movies = []
        pagination = None

    return render_template(
        "search.html",
        movies=display_movies,
        query=query,
        genres=genres,
        pagination=pagination,
    )


@movies.route("/genre/<genre>")
def genre(genre):
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = 24
    sort_by = request.args.get("sort_by", "title")
    sort_order = request.args.get("sort_order", "asc")
    genres = _genres()

    movie_rows, total = movie_service.get_movies_by_genre(
        genre,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    display_movies = movie_service.enrich_movies_list(movie_rows) if movie_rows else []

    return render_template(
        "genre.html",
        genre=genre,
        movies=display_movies,
        genres=genres,
        pagination=_pagination(
            page,
            per_page,
            total,
            sort_by=sort_by,
            sort_order=sort_order,
        ),
    )


@movies.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    genres = _genres()
    movie = movie_service.get_movie_by_id(movie_id, with_tmdb=True)
    if not movie:
        return render_template("404.html", genres=genres), 404

    try:
        content_recommendations = _content_recommendations(movie_id)
    except Exception:
        logger.exception("Content recommendations failed for movie %s", movie_id)
        content_recommendations = []

    tmdb_similar = []
    tmdb_id = movie.get("tmdb_id")
    if tmdb_id:
        try:
            tmdb_similar = [
                _tmdb_for_template(item)
                for item in movie_service.get_tmdb_similar_movies(tmdb_id, limit=6)
            ]
            tmdb_similar = [item for item in tmdb_similar if item]
        except Exception:
            logger.exception("TMDb similar lookup failed for movie %s", movie_id)

    return render_template(
        "movie_detail.html",
        movie=movie,
        similar_movies=content_recommendations,
        tmdb_similar_movies=tmdb_similar,
        genres=genres,
    )


@movies.route("/movie/tmdb/<int:tmdb_id>")
def movie_detail_by_tmdb(tmdb_id):
    genres = _genres()
    local_movie_id = movie_service.find_local_id_from_tmdb_id(tmdb_id)
    if local_movie_id:
        return movie_detail(local_movie_id)

    movie = _tmdb_for_template(get_tmdb_movie_details(tmdb_id))
    if not movie:
        return render_template("404.html", genres=genres), 404

    try:
        tmdb_similar = [
            _tmdb_for_template(item)
            for item in movie_service.get_tmdb_similar_movies(tmdb_id, limit=6)
        ]
        tmdb_similar = [item for item in tmdb_similar if item]
    except Exception:
        logger.exception("TMDb similar lookup failed for TMDb ID %s", tmdb_id)
        tmdb_similar = []

    return render_template(
        "movie_detail.html",
        movie=movie,
        similar_movies=[],
        tmdb_similar_movies=tmdb_similar,
        genres=genres,
    )


@movies.route("/api/search")
def api_search():
    query = request.args.get("query", "").strip()
    limit = min(max(request.args.get("limit", 10, type=int), 1), 50)
    if not query:
        return jsonify([])

    rows, _ = movie_service.search_movies(query, page=1, per_page=limit)
    return jsonify(
        [
            {
                "id": movie.get("movieId"),
                "title": movie.get("title"),
                "year": movie.get("year", ""),
                "genres": movie.get("genres", ""),
            }
            for movie in rows
        ]
    )
