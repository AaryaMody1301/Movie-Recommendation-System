"""TMDb API integration helpers.

All credentials are read from configuration/environment variables. If TMDb is not
configured or is unavailable, helpers fail closed and return empty results instead
of fabricating movie data.
"""

import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from flask import current_app, has_app_context

logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
CACHE_EXPIRY_TIME = 7 * 24 * 60 * 60
API_CACHE: Dict[str, tuple[Dict[str, Any], float]] = {}
_LOGGED_MISSING_KEY = False


def _config_value(name: str, default: Any = None) -> Any:
    if has_app_context():
        return current_app.config.get(name, os.environ.get(name, default))
    return os.environ.get(name, default)


def _get_tmdb_api_key() -> Optional[str]:
    global _LOGGED_MISSING_KEY
    api_key = _config_value("TMDB_API_KEY")
    if api_key:
        return str(api_key).strip()

    if not _LOGGED_MISSING_KEY:
        logger.warning("TMDB_API_KEY is not configured; TMDb features are disabled.")
        _LOGGED_MISSING_KEY = True
    return None


def _get_timeout() -> float:
    try:
        return max(float(_config_value("TMDB_REQUEST_TIMEOUT", 10)), 1.0)
    except (TypeError, ValueError):
        return 10.0


def _make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Call the TMDb v3 API with caching and a bounded request timeout."""
    api_key = _get_tmdb_api_key()
    if not api_key:
        return None

    request_params = dict(params or {})
    params_json = json.dumps(request_params, sort_keys=True, default=str)
    cache_key = f"{endpoint}:{params_json}"

    cached = API_CACHE.get(cache_key)
    if cached:
        cached_data, cached_at = cached
        if time.time() - cached_at < CACHE_EXPIRY_TIME:
            return cached_data
        API_CACHE.pop(cache_key, None)

    request_params["api_key"] = api_key
    url = f"{TMDB_BASE_URL}{endpoint}"

    try:
        response = requests.get(url, params=request_params, timeout=_get_timeout())
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            API_CACHE[cache_key] = (data, time.time())
            return data
        logger.error("TMDb returned a non-object JSON response for %s", endpoint)
    except requests.Timeout:
        logger.warning("TMDb request timed out for %s", endpoint)
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning("TMDb request failed for %s (status=%s): %s", endpoint, status, exc)
    except ValueError as exc:
        logger.warning("TMDb returned invalid JSON for %s: %s", endpoint, exc)
    except Exception:
        logger.exception("Unexpected TMDb error for %s", endpoint)

    return None


def get_poster_url(poster_path: Optional[str], size: str = "w500") -> Optional[str]:
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE_URL}{size}{poster_path}"


def get_backdrop_url(backdrop_path: Optional[str], size: str = "w1280") -> Optional[str]:
    if not backdrop_path:
        return None
    return f"{TMDB_IMAGE_BASE_URL}{size}{backdrop_path}"


@lru_cache(maxsize=1000)
def search_movie_by_title(title: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"query": title}
    if year:
        params["year"] = year
    response = _make_request("/search/movie", params)
    return response.get("results", []) if response else []


@lru_cache(maxsize=1000)
def get_movie_details(movie_id: int) -> Optional[Dict[str, Any]]:
    movie_details = _make_request(
        f"/movie/{movie_id}",
        {"append_to_response": "keywords,videos,credits"},
    )
    if not movie_details:
        return None

    credits = movie_details.get("credits") or {}
    crew = credits.get("crew") or []
    cast = (credits.get("cast") or [])[:10]
    director = next((member for member in crew if member.get("job") == "Director"), None)

    videos = movie_details.get("videos") or {}
    trailers = [
        video
        for video in videos.get("results", [])
        if video.get("type") == "Trailer" and video.get("site") == "YouTube"
    ]

    keyword_container = movie_details.get("keywords") or {}
    keywords = keyword_container.get("keywords", keyword_container.get("results", []))

    return {
        "id": movie_details.get("id"),
        "title": movie_details.get("title"),
        "original_title": movie_details.get("original_title"),
        "overview": movie_details.get("overview"),
        "release_date": movie_details.get("release_date"),
        "runtime": movie_details.get("runtime"),
        "genres": movie_details.get("genres", []),
        "vote_average": movie_details.get("vote_average"),
        "vote_count": movie_details.get("vote_count"),
        "popularity": movie_details.get("popularity"),
        "poster_path": movie_details.get("poster_path"),
        "backdrop_path": movie_details.get("backdrop_path"),
        "director": director,
        "cast": cast,
        "trailers": trailers,
        "keywords": keywords,
        "production_companies": movie_details.get("production_companies", []),
        "production_countries": movie_details.get("production_countries", []),
        "poster_url": get_poster_url(movie_details.get("poster_path")),
        "backdrop_url": get_backdrop_url(movie_details.get("backdrop_path")),
    }


def get_watch_providers(movie_id: int, region: str = "IN") -> Dict[str, List[Dict[str, Any]]]:
    response = _make_request(f"/movie/{movie_id}/watch/providers")
    results = response.get("results", {}) if response else {}
    providers = results.get(region.upper(), {})
    return {
        "flatrate": providers.get("flatrate", []),
        "rent": providers.get("rent", []),
        "buy": providers.get("buy", []),
    }


def get_similar_movies(movie_id: int, max_results: int = 10) -> List[Dict[str, Any]]:
    response = _make_request(f"/movie/{movie_id}/similar")
    similar_movies = response.get("results", []) if response else []
    result: List[Dict[str, Any]] = []
    for movie in similar_movies[:max_results]:
        item = dict(movie)
        item["tmdb_id"] = item.get("id")
        item["poster_url"] = get_poster_url(item.get("poster_path"))
        result.append(item)
    return result


def find_tmdb_id_for_movie(title: str, year: Optional[int] = None) -> Optional[int]:
    search_results = search_movie_by_title(title, year)
    if not search_results:
        return None

    if year:
        for result in search_results:
            release_date = result.get("release_date") or ""
            if release_date.startswith(str(year)):
                return result.get("id")

    return search_results[0].get("id")


def get_movie_keywords(movie_id: int) -> List[Dict[str, Any]]:
    response = _make_request(f"/movie/{movie_id}/keywords")
    if not response:
        return []
    return response.get("keywords", response.get("results", []))


def get_movie_credits(movie_id: int) -> Dict[str, List[Dict[str, Any]]]:
    response = _make_request(f"/movie/{movie_id}/credits") or {}
    return {
        "cast": response.get("cast", []),
        "crew": response.get("crew", []),
    }


def get_movie_videos(movie_id: int) -> List[Dict[str, Any]]:
    response = _make_request(f"/movie/{movie_id}/videos")
    return response.get("results", []) if response else []
