"""Canonical movie catalog service with durable, bounded TMDb enrichment.

The Flask application owns one DataLoader and content recommender.  TMDb identity
mappings are persisted in SQLAlchemy, and list enrichment is cache-only by default so
Browse/Search/Genre/Home cannot fan out into dozens of synchronous remote requests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import wraps
import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from flask import current_app, has_app_context

from database.db import db
from database.models import MovieTmdbMapping
from services.tmdb_service import (
    find_movie_by_external_id,
    find_tmdb_id_for_movie,
    get_movie_details,
    get_similar_movies as get_tmdb_similar_movies_api,
)

logger = logging.getLogger(__name__)

_cache = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
    """Return a local movie, refreshing TMDb only for detail-oriented calls."""
    try:
        movie = get_data_loader().get_movie_by_id(int(movie_id))
        if movie is None:
            return None
        result = movie.to_dict()
        return enrich_movie_with_tmdb(result, allow_remote=True) if with_tmdb else result
    except Exception:
        logger.exception("Failed to get movie %s", movie_id)
        return None


def search_movies(query: str, page: int = 1, per_page: int = 24) -> Tuple[List[Dict], int]:
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
    movie_id = int(movie_id)
    top_n = max(1, int(top_n))
    recommender = _get_recommender()
    if recommender is None:
        return _fallback_recommendations(movie_id, top_n)

    try:
        recommendations = recommender.get_recommendations(movie_id, top_n=top_n)
        return recommendations or _fallback_recommendations(movie_id, top_n)
    except ValueError:
        logger.info("Content recommendations unavailable for movie %s; using popular fallback", movie_id)
        return _fallback_recommendations(movie_id, top_n)
    except Exception:
        logger.exception("Failed to get content recommendations for movie %s", movie_id)
        return _fallback_recommendations(movie_id, top_n)


def update_movie_metadata(movie_id: int, metadata: Dict[str, Any]) -> bool:
    try:
        frame = get_data_loader().get_movies()
        matches = frame.index[frame["movieId"] == int(movie_id)].tolist()
        if not matches:
            return False
        allowed = {"title", "genres", "year", "overview", "poster_url", "tmdb_id"}
        for field, value in metadata.items():
            if field in allowed and field in frame.columns:
                frame.at[matches[0], field] = value
        return True
    except Exception:
        logger.exception("Failed to update movie metadata for %s", movie_id)
        return False


def _valid_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _imdb_id(movie: Dict) -> Optional[str]:
    value = movie.get("imdbId") or movie.get("imdb_id")
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("tt") and text[2:].isdigit():
        return text
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return "tt" + digits.zfill(7)


def _catalog_key(movie: Dict) -> str:
    title = str(movie.get("title", ""))
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", title).strip().casefold()
    year = movie.get("year") or _extract_year_from_title(title) or ""
    direct_tmdb = _valid_int(movie.get("tmdb_id") or movie.get("tmdbId")) or ""
    identity = f"{clean_title}|{year}|{_imdb_id(movie) or ''}|{direct_tmdb}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _mapping_ttl(status: str) -> int:
    name = "TMDB_MAPPING_TTL" if status == "resolved" else "TMDB_NEGATIVE_MAPPING_TTL"
    default = 30 * 24 * 60 * 60 if status == "resolved" else 24 * 60 * 60
    try:
        return max(60, int(current_app.config.get(name, default)))
    except (TypeError, ValueError):
        return default


def _mapping_row(local_movie_id: int) -> Optional[MovieTmdbMapping]:
    if not has_app_context():
        return None
    try:
        return db.session.get(MovieTmdbMapping, int(local_movie_id))
    except Exception:
        logger.exception("Failed reading TMDb mapping for local movie %s", local_movie_id)
        return None


def _mapping_fresh(row: MovieTmdbMapping) -> bool:
    expires_at = _aware(row.expires_at)
    return bool(expires_at and expires_at > _utcnow())


def _store_mapping(
    local_movie_id: int,
    catalog_key: str,
    tmdb_id: Optional[int],
    *,
    matched_by: Optional[str],
) -> None:
    if not has_app_context():
        return
    status = "resolved" if tmdb_id else "not_found"
    now = _utcnow()
    try:
        row = db.session.get(MovieTmdbMapping, int(local_movie_id))
        if row is None:
            row = MovieTmdbMapping(local_movie_id=int(local_movie_id))
            db.session.add(row)
        row.tmdb_id = int(tmdb_id) if tmdb_id else None
        row.catalog_key = catalog_key
        row.status = status
        row.matched_by = matched_by
        row.checked_at = now
        row.expires_at = now + timedelta(seconds=_mapping_ttl(status))
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Failed persisting TMDb mapping for local movie %s", local_movie_id)


def get_tmdb_id_for_movie(movie: Dict, *, allow_remote: bool = True) -> Optional[int]:
    """Resolve a local movie to TMDb, preferring durable/direct identifiers.

    Cache-only callers never perform title/IMDb network resolution.  Expired resolved
    mappings may still be used by cache-only list pages; detail requests refresh them.
    Negative mappings are honored until their shorter expiry.
    """
    local_movie_id = _valid_int(movie.get("movieId"))
    if local_movie_id is None:
        return None
    key = _catalog_key(movie)

    direct_tmdb = _valid_int(movie.get("tmdb_id") or movie.get("tmdbId"))
    if direct_tmdb:
        _store_mapping(local_movie_id, key, direct_tmdb, matched_by="catalog_tmdb_id")
        return direct_tmdb

    row = _mapping_row(local_movie_id)
    if row is not None and row.catalog_key == key:
        if row.status == "resolved" and row.tmdb_id:
            if _mapping_fresh(row) or not allow_remote:
                return int(row.tmdb_id)
        elif row.status == "not_found" and _mapping_fresh(row):
            return None

    if not allow_remote:
        return None

    resolved_tmdb_id = None
    matched_by = None
    imdb_id = _imdb_id(movie)
    if imdb_id:
        try:
            external_match = find_movie_by_external_id(imdb_id, external_source="imdb_id")
            if external_match:
                resolved_tmdb_id = _valid_int(external_match.get("id"))
                matched_by = "imdb_id" if resolved_tmdb_id else None
        except Exception:
            logger.exception("IMDb->TMDb lookup failed for local movie %s", local_movie_id)

    if not resolved_tmdb_id:
        title = str(movie.get("title", ""))
        year = _valid_int(movie.get("year")) or _extract_year_from_title(title)
        clean_title = re.sub(r"\s*\(\d{4}\)$", "", title).strip()
        try:
            resolved_tmdb_id = find_tmdb_id_for_movie(clean_title, year)
            matched_by = "title_year" if resolved_tmdb_id else None
        except Exception:
            logger.exception("Title/year TMDb lookup failed for local movie %s", local_movie_id)

    _store_mapping(local_movie_id, key, resolved_tmdb_id, matched_by=matched_by)
    return int(resolved_tmdb_id) if resolved_tmdb_id else None


def enrich_movie_with_tmdb(
    movie: Dict,
    *,
    allow_remote: bool = True,
    allow_stale: bool = True,
) -> Dict:
    """Return a local movie plus cached/refreshed TMDb metadata."""
    enriched = dict(movie)
    enriched.setdefault("tmdb_id", None)
    enriched.setdefault("tmdb_poster_url", None)
    enriched.setdefault("tmdb_backdrop_url", None)

    tmdb_id = get_tmdb_id_for_movie(movie, allow_remote=allow_remote)
    if not tmdb_id:
        return enriched
    enriched["tmdb_id"] = tmdb_id

    try:
        details = get_movie_details(
            tmdb_id,
            allow_remote=allow_remote,
            allow_stale=allow_stale,
        )
        if not details:
            return enriched
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
            "watch_providers": details.get("watch_providers", {}),
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


def enrich_movies_list(
    movies: List[Dict],
    with_tmdb: bool = True,
    *,
    allow_remote: bool = False,
) -> List[Dict]:
    """Enrich a list from durable cache by default, never fanning out remote work.

    Set ``allow_remote=True`` only for explicit cache-warming/admin workflows.  Normal
    list pages intentionally use cached/stale data and placeholders for uncached movies.
    """
    if not movies:
        return []
    if not with_tmdb:
        return [dict(movie) for movie in movies if isinstance(movie, dict)]

    per_call_cache: Dict[int, Dict] = {}
    result = []
    for movie in movies:
        if not isinstance(movie, dict):
            continue
        local_id = _valid_int(movie.get("movieId"))
        if local_id is None:
            result.append(dict(movie))
            continue
        if local_id not in per_call_cache:
            per_call_cache[local_id] = enrich_movie_with_tmdb(
                movie,
                allow_remote=allow_remote,
                allow_stale=True,
            )
            if not per_call_cache[local_id].get("tmdb_poster_url"):
                per_call_cache[local_id]["tmdb_poster_url"] = "/static/img/movie-placeholder.jpg"
        result.append(dict(per_call_cache[local_id]))
    return result


@memoize_or_pass_through(timeout=86400)
def get_unique_genres() -> List[str]:
    try:
        return get_data_loader().get_unique_genres()
    except Exception:
        logger.exception("Failed to get genres")
        return []


def find_local_id_from_tmdb_id(tmdb_id: int) -> Optional[int]:
    """Resolve TMDb->local identity from persisted mappings without catalog-wide API scans."""
    tmdb_id = _valid_int(tmdb_id)
    if tmdb_id is None:
        return None

    if has_app_context():
        try:
            mapping = db.session.execute(
                db.select(MovieTmdbMapping)
                .where(
                    MovieTmdbMapping.tmdb_id == tmdb_id,
                    MovieTmdbMapping.status == "resolved",
                )
                .order_by(MovieTmdbMapping.checked_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if mapping is not None:
                return int(mapping.local_movie_id)
        except Exception:
            logger.exception("Failed reverse TMDb mapping lookup for %s", tmdb_id)

    frame = get_data_loader().get_movies()
    for column in ("tmdb_id", "tmdbId"):
        if column in frame.columns:
            matches = frame[frame[column] == tmdb_id]
            if not matches.empty:
                return int(matches.iloc[0]["movieId"])
    return None
