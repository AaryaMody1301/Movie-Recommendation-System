"""TMDb API integration with bounded retries and durable enrichment caching.

Credentials remain configuration-only.  Raw HTTP responses receive a short process
cache, while normalized movie detail/provider payloads are persisted in SQLAlchemy so
page loads do not repeatedly call TMDb after process restarts.  Expired persisted
payloads may be served as stale data when TMDb is unavailable, subject to a bounded
stale-cache window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import current_app, has_app_context
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from database.db import db
from database.models import TmdbEnrichmentCache

logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
API_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
_LOGGED_MISSING_KEY = False
_SESSION: Optional[requests.Session] = None
_SESSION_SIGNATURE: Optional[Tuple[int, float]] = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _config_value(name: str, default: Any = None) -> Any:
    if has_app_context():
        return current_app.config.get(name, os.environ.get(name, default))
    return os.environ.get(name, default)


def _config_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(_config_value(name, default)))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _config_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(_config_value(name, default)))
    except (TypeError, ValueError):
        return max(minimum, float(default))


def _language(value: Optional[str] = None) -> str:
    return str(value or _config_value("TMDB_LANGUAGE", "en-US")).strip() or "en-US"


def _region(value: Optional[str] = None) -> str:
    region = str(value or _config_value("TMDB_WATCH_REGION", "IN")).strip().upper()
    return region or "IN"


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
    return _config_float("TMDB_REQUEST_TIMEOUT", 10.0, minimum=1.0)


def _get_session() -> requests.Session:
    """Return a process-scoped requests Session with GET retry/backoff policy."""
    global _SESSION, _SESSION_SIGNATURE

    retry_total = _config_int("TMDB_RETRY_TOTAL", 2, minimum=0)
    retry_backoff = _config_float("TMDB_RETRY_BACKOFF", 0.5, minimum=0.0)
    signature = (retry_total, retry_backoff)
    if _SESSION is not None and _SESSION_SIGNATURE == signature:
        return _SESSION

    retry = Retry(
        total=retry_total,
        connect=retry_total,
        read=retry_total,
        status=retry_total,
        backoff_factor=retry_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})

    _SESSION = session
    _SESSION_SIGNATURE = signature
    return session


def clear_process_cache() -> None:
    """Clear process-local TMDb response state (primarily useful for tests/admin work)."""
    API_CACHE.clear()


def _make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Call TMDb with bounded timeout, retries, Retry-After support, and short caching."""
    api_key = _get_tmdb_api_key()
    if not api_key:
        return None

    request_params = dict(params or {})
    params_json = json.dumps(request_params, sort_keys=True, default=str)
    cache_key = f"{endpoint}:{params_json}"
    cache_ttl = _config_int("TMDB_HTTP_CACHE_TTL", 900, minimum=0)

    cached = API_CACHE.get(cache_key)
    if cached:
        cached_data, cached_at = cached
        if time.time() - cached_at < cache_ttl:
            return dict(cached_data)
        API_CACHE.pop(cache_key, None)

    request_params["api_key"] = api_key
    url = f"{TMDB_BASE_URL}{endpoint}"

    try:
        response = _get_session().get(url, params=request_params, timeout=_get_timeout())
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if cache_ttl > 0:
                API_CACHE[cache_key] = (dict(data), time.time())
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


def _normalized_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^\w]+", " ", normalized.casefold(), flags=re.UNICODE)
    return " ".join(normalized.split())


def search_movie_by_title(
    title: str,
    year: Optional[int] = None,
    language: Optional[str] = None,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "query": str(title),
        "include_adult": False,
        "language": _language(language),
    }
    if year:
        params["year"] = int(year)
    if region:
        params["region"] = _region(region)
    response = _make_request("/search/movie", params)
    return response.get("results", []) if response else []


def find_movie_by_external_id(
    external_id: str,
    external_source: str = "imdb_id",
    language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a TMDb movie from a supported external identifier such as IMDb."""
    external_id = str(external_id or "").strip()
    if not external_id:
        return None
    response = _make_request(
        f"/find/{external_id}",
        {
            "external_source": external_source,
            "language": _language(language),
        },
    )
    movies = response.get("movie_results", []) if response else []
    return dict(movies[0]) if movies else None


def find_tmdb_id_for_movie(
    title: str,
    year: Optional[int] = None,
    language: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[int]:
    """Resolve title/year conservatively instead of blindly taking the first search hit."""
    search_results = search_movie_by_title(title, year, language=language, region=region)
    if not search_results:
        return None

    wanted_title = _normalized_title(title)
    ranked = []
    for result in search_results:
        candidate_titles = {
            _normalized_title(result.get("title", "")),
            _normalized_title(result.get("original_title", "")),
        }
        exact_title = wanted_title in candidate_titles
        if not exact_title:
            continue

        release_date = str(result.get("release_date") or "")
        result_year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else None
        if year is not None and result_year is not None and result_year != int(year):
            continue

        score = 100
        if year is not None and result_year == int(year):
            score += 25
        score += min(float(result.get("vote_count") or 0), 10000.0) / 10000.0
        ranked.append((score, int(result["id"])))

    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def _read_persisted_details(
    movie_id: int,
    language: str,
    region: str,
) -> Optional[TmdbEnrichmentCache]:
    if not has_app_context():
        return None
    try:
        return db.session.execute(
            db.select(TmdbEnrichmentCache).where(
                TmdbEnrichmentCache.tmdb_id == int(movie_id),
                TmdbEnrichmentCache.language == language,
                TmdbEnrichmentCache.region == region,
            )
        ).scalar_one_or_none()
    except Exception:
        logger.exception("Failed reading persisted TMDb cache for movie %s", movie_id)
        return None


def _row_is_fresh(row: TmdbEnrichmentCache) -> bool:
    expires_at = _aware(row.expires_at)
    return bool(expires_at and expires_at > _utcnow())


def _row_is_usable_stale(row: TmdbEnrichmentCache) -> bool:
    fetched_at = _aware(row.fetched_at)
    if fetched_at is None:
        return False
    stale_ttl = _config_int("TMDB_STALE_CACHE_TTL", 30 * 24 * 60 * 60, minimum=0)
    return _utcnow() - fetched_at <= timedelta(seconds=stale_ttl)


def _persist_details(movie_id: int, language: str, region: str, payload: Dict[str, Any]) -> None:
    if not has_app_context():
        return
    now = _utcnow()
    ttl = _config_int("TMDB_ENRICHMENT_TTL", 7 * 24 * 60 * 60, minimum=60)
    try:
        row = db.session.execute(
            db.select(TmdbEnrichmentCache).where(
                TmdbEnrichmentCache.tmdb_id == int(movie_id),
                TmdbEnrichmentCache.language == language,
                TmdbEnrichmentCache.region == region,
            )
        ).scalar_one_or_none()
        if row is None:
            row = TmdbEnrichmentCache(
                tmdb_id=int(movie_id),
                language=language,
                region=region,
                payload=dict(payload),
                fetched_at=now,
                expires_at=now + timedelta(seconds=ttl),
            )
            db.session.add(row)
        else:
            row.payload = dict(payload)
            row.fetched_at = now
            row.expires_at = now + timedelta(seconds=ttl)
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Failed persisting TMDb enrichment cache for movie %s", movie_id)


def get_watch_providers(
    movie_id: int,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Return provider availability for the configured ISO country/region."""
    selected_region = _region(region)
    response = _make_request(f"/movie/{int(movie_id)}/watch/providers")
    results = response.get("results", {}) if response else {}
    providers = results.get(selected_region, {})
    return {
        "region": selected_region,
        "link": providers.get("link"),
        "flatrate": providers.get("flatrate", []),
        "rent": providers.get("rent", []),
        "buy": providers.get("buy", []),
    }


def _normalize_movie_details(
    movie_details: Dict[str, Any],
    providers: Dict[str, Any],
) -> Dict[str, Any]:
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
        "watch_providers": providers,
    }


def get_movie_details(
    movie_id: int,
    language: Optional[str] = None,
    region: Optional[str] = None,
    *,
    allow_remote: bool = True,
    allow_stale: bool = True,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return normalized detail/provider data with durable cache and stale-if-error."""
    movie_id = int(movie_id)
    selected_language = _language(language)
    selected_region = _region(region)
    cached_row = _read_persisted_details(movie_id, selected_language, selected_region)

    if cached_row is not None and not force_refresh and _row_is_fresh(cached_row):
        return dict(cached_row.payload)

    stale_payload = None
    if cached_row is not None and allow_stale and _row_is_usable_stale(cached_row):
        stale_payload = dict(cached_row.payload)

    if not allow_remote:
        return stale_payload

    movie_details = _make_request(
        f"/movie/{movie_id}",
        {
            "append_to_response": "keywords,videos,credits",
            "language": selected_language,
        },
    )
    if not movie_details:
        return stale_payload

    providers = get_watch_providers(movie_id, region=selected_region)
    payload = _normalize_movie_details(movie_details, providers)
    _persist_details(movie_id, selected_language, selected_region, payload)
    return payload


def get_similar_movies(movie_id: int, max_results: int = 10) -> List[Dict[str, Any]]:
    response = _make_request(
        f"/movie/{int(movie_id)}/similar",
        {"language": _language()},
    )
    similar_movies = response.get("results", []) if response else []
    result: List[Dict[str, Any]] = []
    for movie in similar_movies[: max(1, int(max_results))]:
        item = dict(movie)
        item["tmdb_id"] = item.get("id")
        item["poster_url"] = get_poster_url(item.get("poster_path"))
        result.append(item)
    return result


def get_movie_keywords(movie_id: int) -> List[Dict[str, Any]]:
    response = _make_request(f"/movie/{int(movie_id)}/keywords")
    if not response:
        return []
    return response.get("keywords", response.get("results", []))


def get_movie_credits(movie_id: int) -> Dict[str, List[Dict[str, Any]]]:
    response = _make_request(f"/movie/{int(movie_id)}/credits") or {}
    return {
        "cast": response.get("cast", []),
        "crew": response.get("crew", []),
    }


def get_movie_videos(movie_id: int) -> List[Dict[str, Any]]:
    response = _make_request(
        f"/movie/{int(movie_id)}/videos",
        {"language": _language()},
    )
    return response.get("results", []) if response else []
