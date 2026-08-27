"""Canonical movie catalog service.

The Flask application owns exactly one ``DataLoader`` and one content recommender.
This module never creates independent loader/model instances; services therefore see
the same catalog and baseline ratings throughout a request/application context.
"""

from __future__ import annotations

from functools import wraps
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from flask import current_app, has_app_context

from services.tmdb_service import (
    find_tmdb_id_for_movie,
    get_movie_details,
    get_similar_movies as get_tmdb_similar_movies_api,
    get_watch_providers,
)

logger = logging.getLogger(__name__)

_cache = None
_TMDB_ID_CACHE: Dict[int, int] = {}


def set_cache(cache_obj):
    """Bind the Flask-Caching extension initialized by the application factory."""
    global _cache
    _cache = cache_obj


def memoize_or_pass_through(timeout: int = 300):
    """Create a memoized wrapper lazily after Flask-Caching has been initialized."""
    def decorator(func):
        cached_func = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cached_func
            if _cache is None:
                return func(*args, **kwargs)
            if cached_func is None:
                cached_func = _cache.memoize(timeout=timeout)(func)
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator


def get_data_loader():
    """Return the single DataLoader owned by the active Flask application."""
    if not has_app_context():
        raise RuntimeError("Movie services require a Flask application context")
    loader = getattr(current_app, "data_loader", None)
    if loader is None:
        raise RuntimeError("Application DataLoader is not available")
    return loader


def _get_data_loader():
    """Compatibility alias for legacy callers."""
    return get_data_loader()


def _get_recommender():
    if not has_app_context():
        raise RuntimeError("Recommendation services require a Flask application context")
    return getattr(current_app, "recommender", None)


def _extract_year_from_title(title: str) -> Optional[int]:
    match = re.search(r"\((\d{4})\)$", title or "")
    return int(match.group(1)) if match else None


def _validate_page(page: int, per_page: int) -> Tuple[int, int]:
    return max(1, int(page)), min(100, max(1, int(per_page)))


def _fallback_recommendations(movie_id: int, top_n: int) -> List[Dict]:
    """Return popular movies in the same shape as model recommendations."""
    limit = max(1, int(top_n))
    popular = get_popular_movies(limit=limit + 1)
    recommendations = []
    for movie in popular:
        if movie.get("movieId") == int(movie_id):
            continue
        recommendations.append(
            {
                "movie": dict(movie),
                "score": 0.0,
                "reason": "Popular fallback while content similarity is unavailable.",
            }
        )
        if len(recommendations) >= limit:
            break
    return recommendations


def get_all_movies(
    page: int = 1,
    per_page: int = 24,
    sort_by: str = "title",
    sort_order: str = "asc",
) -> Tuple[List[Dict], int]:
    """Return catalog movies with pagination and sorting over the full catalog."""
    try:
        page, per_page = _validate_page(page, per_page)
        loader = get_data_loader()
        movies_df = loader.get_movies().copy()
        ascending = str(sort_order).lower() != "desc"

        if sort_by == "year" and "year" in movies_df.columns:
            movies_df = movies_df.sort_values(
                ["year", "title"],
                ascending=[ascending, True],
                na_position="last",
            )
        elif sort_by == "rating":
            stats = loader.get_movie_rating_stats()
            movies_df = movies_df.merge(stats, on="movieId", how="left")
            movies_df = movies_df.sort_values(
                ["average_rating", "rating_count", "title"],
                ascending=[ascending, ascending, True],
                na_position="last",
            )
        else:
            movies_df = movies_df.sort_values("title", ascending=ascending, na_position="last")

        total = len(movies_df)
        offset = (page - 1) * per_page
        return movies_df.iloc[offset : offset + per_page].to_dict("records"), total
    except Exception:
        logger.exception("Failed to get catalog movies")
        return [], 0


def get_movie_by_id(movie_id: int, with_tmdb: bool = True) -> Optional[Dict]:
    """Return a local movie record, optionally enriched with TMDb metadata."""
    try:
        movie = get_data_loader().get_movie_by_id(int(movie_id))
        if movie is None:
            return None
        result = movie.to_dict()
        return enrich_movie_with_tmdb(result) if with_tmdb else result
    except Exception:
        logger.exception("Failed to get movie %s", movie_id)
        return None


def search_movies(query: str, page: int = 1, per_page: int = 24) -> Tuple[List[Dict], int]:
    """Search all titles literally, then paginate the complete result set."""
    try:
        page, per_page = _validate_page(page, per_page)
        results = get_data_loader().search_movies(str(query))
        total = len(results)
        offset = (page - 1) * per_page
        return results.iloc[offset : offset + per_page].to_dict("records"), total
    except Exception:
        logger.exception("Failed to search movies for %r", query)
        return [], 0


def get_movies_by_genre(
    genre: str,
    page: int = 1,
    per_page: int = 24,
    sort_by: str = "title",
    sort_order: str = "asc",
) -> Tuple[List[Dict], int]:
    """Filter by an exact genre token, then sort and paginate the full result set."""
    try:
        page, per_page = _validate_page(page, per_page)
        loader = get_data_loader()
        frame = loader.get_movies_by_genre(str(genre)).copy()
        ascending = str(sort_order).lower() != "desc"

        if sort_by == "year" and "year" in frame.columns:
            frame = frame.sort_values(
                ["year", "title"],
                ascending=[ascending, True],
                na_position="last",
            )
        elif sort_by == "rating":
            stats = loader.get_movie_rating_stats()
            frame = frame.merge(stats, on="movieId", how="left")
            frame = frame.sort_values(
                ["average_rating", "rating_count", "title"],
                ascending=[ascending, ascending, True],
                na_position="last",
            )
        elif "title" in frame.columns:
            frame = frame.sort_values("title", ascending=ascending, na_position="last")

        total = len(frame)
        offset = (page - 1) * per_page
        return frame.iloc[offset : offset + per_page].to_dict("records"), total
    except Exception:
        logger.exception("Failed to get movies for genre %r", genre)
        return [], 0


@memoize_or_pass_through(timeout=3600)
def get_popular_movies(limit: int = 10) -> List[Dict]:
    try:
        frame = get_data_loader().get_popular_movies(n=max(1, int(limit)))
        return frame.to_dict("records")
    except Exception:
        logger.exception("Failed to get popular movies")
        return []


@memoize_or_pass_through(timeout=3600)
def get_high_rated_movies(limit: int = 10, min_ratings: int = 10) -> List[Dict]:
    try:
        frame = get_data_loader().get_high_rated_movies(
            min_ratings=max(1, int(min_ratings)),
            n=max(1, int(limit)),
        )
        return frame.to_dict("records")
    except Exception:
        logger.exception("Failed to get high-rated movies")
        return []


@memoize_or_pass_through(timeout=600)
def get_content_recommendations(movie_id: int, top_n: int = 10) -> List[Dict]:
    """Return normalized recommendations from the app-owned content recommender."""
    movie_id = int(movie_id)
    top_n = max(1, int(top_n))
    recommender = _get_recommender()
    if recommender is None:
        return _fallback_recommendations(movie_id, top_n)

    try:
        recommendations = recommender.get_recommendations(movie_id, top_n=top_n)
        if recommendations:
            return recommendations
        return _fallback_recommendations(movie_id, top_n)
    except ValueError:
        logger.info("Content recommendations unavailable for movie %s; using popular fallback", movie_id)
        return _fallback_recommendations(movie_id, top_n)
    except Exception:
        logger.exception("Failed to get content recommendations for movie %s", movie_id)
        return _fallback_recommendations(movie_id, top_n)


def update_movie_metadata(movie_id: int, metadata: Dict[str, Any]) -> bool:
    """Update supported fields in the in-memory catalog (legacy helper)."""
    try:
        frame = get_data_loader().get_movies()
        matches = frame.index[frame["movieId"] == int(movie_id)].tolist()
        if not matches:
            return False
        allowed = {"title", "genres", "year", "overview", "poster_url"}
        for field, value in metadata.items():
            if field in allowed and field in frame.columns:
                frame.at[matches[0], field] = value
        return True
    except Exception:
        logger.exception("Failed to update movie metadata for %s", movie_id)
        return False


def get_tmdb_id_for_movie(movie: Dict) -> Optional[int]:
    """Resolve and cache the TMDb ID for a local movie."""
    try:
        movie_id = int(movie.get("movieId"))
    except (TypeError, ValueError):
        return None
    if movie_id in _TMDB_ID_CACHE:
        return _TMDB_ID_CACHE[movie_id]

    title = str(movie.get("title", ""))
    year = _extract_year_from_title(title)
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", title)
    try:
        tmdb_id = find_tmdb_id_for_movie(clean_title, year)
    except Exception:
        logger.exception("TMDb lookup failed for movie %s", movie_id)
        return None
    if tmdb_id:
        _TMDB_ID_CACHE[movie_id] = int(tmdb_id)
        return int(tmdb_id)
    return None


def enrich_movie_with_tmdb(movie: Dict) -> Dict:
    """Return a copy of a local movie enriched with available TMDb metadata."""
    enriched = dict(movie)
    enriched.setdefault("tmdb_id", None)
    enriched.setdefault("tmdb_poster_url", None)
    enriched.setdefault("tmdb_backdrop_url", None)

    tmdb_id = get_tmdb_id_for_movie(movie)
    if not tmdb_id:
        return enriched
    enriched["tmdb_id"] = tmdb_id

    try:
        details = get_movie_details(tmdb_id)
        if not details:
            return enriched
        providers = get_watch_providers(tmdb_id)
        fields = {
            "tmdb_poster_url": details.get("poster_url"),
            "tmdb_backdrop_url": details.get("backdrop_url"),
            "overview": details.get("overview"),
            "release_date": details.get("release_date"),
            "runtime": details.get("runtime"),
            "vote_average": details.get("vote_average"),
            "vote_count": details.get("vote_count"),
            "popularity": details.get("popularity"),
            "tmdb_genres": details.get("genres", []),
            "director": details.get("director"),
            "cast": details.get("cast", []),
            "trailers": details.get("trailers", []),
            "keywords": details.get("keywords", []),
            "production_companies": details.get("production_companies", []),
            "production_countries": details.get("production_countries", []),
            "watch_providers": providers,
        }
        enriched.update({key: value for key, value in fields.items() if value is not None})
        return enriched
    except Exception:
        logger.exception("TMDb enrichment failed for movie %s", movie.get("movieId"))
        return enriched


def get_tmdb_similar_movies(movie_id: int, limit: int = 10) -> List[Dict]:
    try:
        return get_tmdb_similar_movies_api(int(movie_id), max_results=max(1, int(limit))) or []
    except Exception:
        logger.exception("TMDb similar lookup failed for TMDb ID %s", movie_id)
        return []


def enrich_movies_list(movies: List[Dict], with_tmdb: bool = True) -> List[Dict]:
    """Enrich valid movie dictionaries while deduplicating remote work per call."""
    if not movies:
        return []
    if not with_tmdb:
        return [dict(movie) for movie in movies if isinstance(movie, dict)]

    cache: Dict[int, Dict] = {}
    result = []
    for movie in movies:
        if not isinstance(movie, dict):
            continue
        movie_id = movie.get("movieId")
        if movie_id is None:
            result.append(dict(movie))
            continue
        try:
            key = int(movie_id)
        except (TypeError, ValueError):
            result.append(dict(movie))
            continue
        if key not in cache:
            cache[key] = enrich_movie_with_tmdb(movie)
            if not cache[key].get("tmdb_poster_url"):
                cache[key]["tmdb_poster_url"] = "/static/img/movie-placeholder.jpg"
        result.append(dict(cache[key]))
    return result


@memoize_or_pass_through(timeout=86400)
def get_unique_genres() -> List[str]:
    try:
        return get_data_loader().get_unique_genres()
    except Exception:
        logger.exception("Failed to get genres")
        return []


def find_local_id_from_tmdb_id(tmdb_id: int) -> Optional[int]:
    """Resolve a local ID from already-known TMDb mappings without mass API scans."""
    try:
        tmdb_id = int(tmdb_id)
    except (TypeError, ValueError):
        return None

    for local_id, cached_tmdb_id in _TMDB_ID_CACHE.items():
        if cached_tmdb_id == tmdb_id:
            return local_id

    frame = get_data_loader().get_movies()
    if "tmdb_id" in frame.columns:
        matches = frame[frame["tmdb_id"] == tmdb_id]
        if not matches.empty:
            return int(matches.iloc[0]["movieId"])
    return None
