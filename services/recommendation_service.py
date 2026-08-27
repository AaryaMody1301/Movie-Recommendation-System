"""Recommendation service contracts built on application-owned services.

Phase 3 deliberately removes the second DataLoader/model stack that previously lived
in this module.  Content similarity comes from the app-scoped recommender exposed by
``movie_service``; application-user interactions come from the SQLAlchemy-backed
``user_service``.  Model-quality work remains a Phase 5 concern.
"""

from __future__ import annotations

import logging
from math import log1p
from typing import Dict, List, Optional, Tuple

from flask import current_app

import services.movie_service as movie_service
import services.user_service as user_service

logger = logging.getLogger(__name__)

VALID_STRATEGIES = {"weighted", "rank"}


def _data_loader():
    loader = getattr(current_app, "data_loader", None)
    if loader is None:
        raise RuntimeError("Application DataLoader is not available")
    return loader


def _normalize_recommendation(item: Dict, default_reason: str = "") -> Optional[Dict]:
    """Normalize recommender and fallback movie records into one stable shape."""
    if not isinstance(item, dict):
        return None

    if isinstance(item.get("movie"), dict):
        movie = dict(item["movie"])
        raw_score = item.get("score", item.get("similarity_score"))
        reason = item.get("reason") or default_reason
    else:
        movie = dict(item)
        raw_score = item.get("score", item.get("similarity_score"))
        reason = default_reason

    movie_id = movie.get("movieId")
    if movie_id is None:
        return None

    try:
        movie["movieId"] = int(movie_id)
    except (TypeError, ValueError):
        return None

    score = None
    if raw_score is not None:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = None

    return {"movie": movie, "score": score, "reason": reason or "Recommended for you."}


def _paginate(items: List[Dict], page: int, per_page: int) -> Tuple[List[Dict], int]:
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    total = len(items)
    start = (page - 1) * per_page
    return items[start : start + per_page], total


def _content_candidates(movie_id: int, count: int) -> List[Dict]:
    """Return normalized content candidates, excluding the source movie itself."""
    count = max(1, min(int(count), 200))
    raw = movie_service.get_content_recommendations(int(movie_id), top_n=count)
    normalized = []
    seen = set()
    for rank, item in enumerate(raw, start=1):
        rec = _normalize_recommendation(
            item,
            default_reason=f"Similar to movie {movie_id} based on content signals.",
        )
        if rec is None:
            continue
        candidate_id = rec["movie"]["movieId"]
        if candidate_id == int(movie_id) or candidate_id in seen:
            continue
        if rec["score"] is None:
            # Fallback lists do not have a similarity score. Preserve deterministic rank.
            rec["score"] = 1.0 / rank
        seen.add(candidate_id)
        normalized.append(rec)
    return normalized


def get_unique_genres() -> List[str]:
    """Expose the canonical movie-service genre contract."""
    return movie_service.get_unique_genres()


def get_similar_movies(
    movie_id: int,
    page: int = 1,
    per_page: int = 12,
    method: str = "content",
) -> Tuple[List[Dict], int]:
    """Return paginated similar movies in the canonical recommendation shape."""
    # Phase 3 has one production content model. Other methods are retained as accepted
    # labels for compatibility and will be implemented fully in Phase 5.
    if method not in {"content", "collaborative", "hybrid"}:
        method = "content"
    candidates = _content_candidates(movie_id, max(page * per_page, per_page * 2))
    return _paginate(candidates, page, per_page)


def get_recommendations_for_user(
    user_id: int,
    page: int = 1,
    per_page: int = 24,
    strategy: str = "weighted",
) -> Tuple[List[Dict], int]:
    """Build personalized candidates from persisted user ratings.

    ``weighted`` combines source rating strength with content similarity. ``rank``
    combines source rating strength with reciprocal recommendation rank.  This is a
    deterministic Phase 3 bridge; collaborative/hybrid model work is deferred to
    Phase 5.
    """
    strategy = strategy if strategy in VALID_STRATEGIES else "weighted"
    ratings = user_service.get_user_rating_records(int(user_id))

    if not ratings:
        popular = [
            _normalize_recommendation(
                movie,
                "Popular fallback because you have not rated any movies yet.",
            )
            for movie in movie_service.get_popular_movies(limit=max(page * per_page, per_page))
        ]
        popular = [item for item in popular if item is not None]
        return _paginate(popular, page, per_page)

    rated_ids = {item["movieId"] for item in ratings}
    sources = sorted(
        ratings,
        key=lambda item: (item["rating"], item.get("timestamp") or 0),
        reverse=True,
    )[:10]

    candidates: Dict[int, Dict] = {}
    per_source = max(10, min(40, per_page * 2))
    for source in sources:
        source_id = source["movieId"]
        source_movie = movie_service.get_movie_by_id(source_id, with_tmdb=False) or {}
        source_title = source_movie.get("title", f"movie {source_id}")
        rating_weight = max(0.1, float(source["rating"]) / 5.0)

        for rank, rec in enumerate(_content_candidates(source_id, per_source), start=1):
            candidate_id = rec["movie"]["movieId"]
            if candidate_id in rated_ids:
                continue

            base_score = float(rec.get("score") or 0.0)
            contribution = (
                rating_weight / rank
                if strategy == "rank"
                else rating_weight * base_score
            )

            entry = candidates.setdefault(
                candidate_id,
                {
                    "movie": rec["movie"],
                    "score": 0.0,
                    "sources": [],
                },
            )
            entry["score"] += contribution
            entry["sources"].append(
                {
                    "movieId": source_id,
                    "title": source_title,
                    "rating": float(source["rating"]),
                    "contribution": contribution,
                }
            )

    ranked = []
    for entry in candidates.values():
        entry["sources"].sort(key=lambda source: source["contribution"], reverse=True)
        top_sources = entry["sources"][:2]
        source_text = ", ".join(
            f"{source['title']} ({source['rating']:.1f}/5)" for source in top_sources
        )
        ranked.append(
            {
                "movie": entry["movie"],
                "score": round(float(entry["score"]), 6),
                "reason": f"Based on movies you rated highly: {source_text}.",
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["movie"].get("title", "")), reverse=True)
    if not ranked:
        popular = [
            _normalize_recommendation(movie, "Popular fallback recommendation.")
            for movie in movie_service.get_popular_movies(limit=max(page * per_page, per_page))
            if movie.get("movieId") not in rated_ids
        ]
        ranked = [item for item in popular if item is not None]

    return _paginate(ranked, page, per_page)


def get_recommendations_for_movie(
    movie_id: int,
    user_id: Optional[int] = None,
    page: int = 1,
    per_page: int = 12,
    strategy: str = "weighted",
) -> Tuple[List[Dict], int]:
    """Return movie-similarity candidates with optional user-history filtering."""
    candidates = _content_candidates(movie_id, max(page * per_page, per_page * 3))
    if user_id is not None:
        rated_ids = {item["movieId"] for item in user_service.get_user_rating_records(user_id)}
        favorite_genres = {
            item["genre"] for item in user_service.get_user_favorite_genres(user_id, top_n=5)
        }
        for rec in candidates:
            candidate_genres = set(str(rec["movie"].get("genres", "")).split("|"))
            overlap = sorted(candidate_genres.intersection(favorite_genres))
            if overlap:
                rec["score"] = float(rec.get("score") or 0.0) + 0.05 * len(overlap)
                rec["reason"] = (
                    rec["reason"] + " Matches your preferred genres: " + ", ".join(overlap) + "."
                )
        candidates = [rec for rec in candidates if rec["movie"]["movieId"] not in rated_ids]
        candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return _paginate(candidates, page, per_page)


def rate_movie(user_id: int, movie_id: int, rating: float) -> bool:
    """Persist a rating through the canonical user service."""
    return bool(user_service.add_user_rating(user_id, movie_id, rating).get("success"))


def get_user_ratings(user_id: int) -> List[Dict]:
    """Compatibility helper returning persisted user ratings."""
    return user_service.get_user_rating_records(user_id)


def get_recommendation_explanation(
    movie_id: int,
    user_id: Optional[int] = None,
    source_id: Optional[int] = None,
) -> Dict:
    """Return a simple evidence-backed explanation for a recommendation."""
    movie = movie_service.get_movie_by_id(int(movie_id), with_tmdb=False)
    if movie is None:
        return {"error": "Movie not found"}

    reasons = []
    genre_match = []
    source_movie = None
    if source_id is not None:
        source_movie = movie_service.get_movie_by_id(int(source_id), with_tmdb=False)
        if source_movie:
            movie_genres = set(str(movie.get("genres", "")).split("|"))
            source_genres = set(str(source_movie.get("genres", "")).split("|"))
            common = sorted(genre for genre in movie_genres.intersection(source_genres) if genre)
            if common:
                reasons.append(
                    f"It shares these genres with {source_movie.get('title', 'the source movie')}: "
                    + ", ".join(common)
                    + "."
                )

    if user_id is not None:
        favorites = user_service.get_user_favorite_genres(int(user_id), top_n=5)
        movie_genres = set(str(movie.get("genres", "")).split("|"))
        for favorite in favorites:
            if favorite["genre"] in movie_genres:
                genre_match.append(favorite)
        if genre_match:
            reasons.append(
                "It matches genres you tend to rate highly: "
                + ", ".join(item["genre"] for item in genre_match)
                + "."
            )

    if not reasons:
        reasons.append("It was selected from the active content-based recommendation model.")

    return {
        "movie": movie,
        "source_movie": source_movie,
        "reasons": reasons,
        "genre_match": genre_match,
    }


def get_movie_popularity_score(movie_id: int) -> float:
    """Return a 0-10 popularity score from the immutable baseline ratings dataset."""
    try:
        ratings = _data_loader().get_movie_ratings(int(movie_id))
        all_ratings = _data_loader().get_ratings()
        movies = _data_loader().get_movies()
        if ratings.empty or all_ratings.empty or movies.empty:
            return 0.0
        num_ratings = len(ratings)
        average = float(ratings["rating"].mean())
        denominator = log1p(max(1.0, len(all_ratings) / len(movies)))
        popularity = (log1p(num_ratings) / denominator) * 10 if denominator else 0.0
        return min(10.0, round(popularity * (average / 5.0), 1))
    except Exception:
        logger.exception("Failed to calculate movie popularity")
        return 0.0


# Backward-compatible names used by older modules/tests.  New code should use the
# paginated contracts above.
def get_user_recommendations(user_id: int, limit: int = 10, method: str = "content") -> List[Dict]:
    items, _ = get_recommendations_for_user(user_id, page=1, per_page=limit)
    return items


def get_recommendations_from_user_history(user_id: int, limit: int = 10) -> List[Dict]:
    return get_user_recommendations(user_id, limit=limit)


def get_explanation(user_id: int, movie_id: int) -> Dict:
    return get_recommendation_explanation(movie_id, user_id=user_id)
