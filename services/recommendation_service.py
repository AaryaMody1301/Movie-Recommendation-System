"""Production recommendation contracts built from persistent user interactions.

Content candidates come from the one app-owned content recommender. Collaborative
signals are trained lazily from SQLAlchemy-backed application-user ratings only; the
CSV baseline ratings dataset is not used for online personalization. Hybrid fusion is
performed on bounded candidate lists, never on a dense full-catalog similarity matrix.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import current_app
from sqlalchemy import func

from database.db import db
from database.models import Rating
from models.collaborative_filtering import CollaborativeRecommender
from models.hybrid_recommender import HybridRecommender
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

    try:
        movie["movieId"] = int(movie["movieId"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        score = None if raw_score is None else float(raw_score)
    except (TypeError, ValueError):
        score = None
    return {
        "movie": movie,
        "score": score,
        "reason": reason or "Recommended from available signals.",
    }


def _paginate(items: List[Dict], page: int, per_page: int) -> Tuple[List[Dict], int]:
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    total = len(items)
    start = (page - 1) * per_page
    return items[start : start + per_page], total


def _content_candidates(movie_id: int, count: int) -> List[Dict]:
    count = max(1, min(int(count), 300))
    raw = movie_service.get_content_recommendations(int(movie_id), top_n=count)
    normalized = []
    seen = set()
    for rank, item in enumerate(raw, start=1):
        rec = _normalize_recommendation(
            item,
            default_reason=f"Content similarity to movie {movie_id} contributed to this result.",
        )
        if rec is None:
            continue
        candidate_id = rec["movie"]["movieId"]
        if candidate_id == int(movie_id) or candidate_id in seen:
            continue
        if rec["score"] is None:
            rec["score"] = 1.0 / rank
        seen.add(candidate_id)
        normalized.append(rec)
    return normalized


def _rating_signature() -> Tuple[int, Optional[str], float]:
    """Cheaply detect persisted-rating changes that require collaborative retraining."""
    count, latest, rating_sum = db.session.execute(
        db.select(
            func.count(Rating.id),
            func.max(Rating.timestamp),
            func.coalesce(func.sum(Rating.rating), 0.0),
        )
    ).one()
    latest_value = latest.isoformat() if latest is not None else None
    return int(count), latest_value, round(float(rating_sum or 0.0), 6)


def _persisted_ratings_frame() -> pd.DataFrame:
    rows = db.session.execute(
        db.select(Rating).order_by(Rating.user_id, Rating.movie_id)
    ).scalars().all()
    return pd.DataFrame(
        [
            {
                "userId": int(row.user_id),
                "movieId": int(row.movie_id),
                "rating": float(row.rating),
            }
            for row in rows
        ],
        columns=["userId", "movieId", "rating"],
    )


def _get_collaborative_model() -> Optional[CollaborativeRecommender]:
    """Return an app-scoped SVD model trained only from persisted application ratings."""
    signature = _rating_signature()
    if getattr(current_app, "_collaborative_signature", None) == signature:
        return getattr(current_app, "_collaborative_recommender", None)

    frame = _persisted_ratings_frame()
    min_ratings = max(1, int(current_app.config.get("COLLAB_MIN_RATINGS", 5)))
    min_users = max(2, int(current_app.config.get("COLLAB_MIN_USERS", 2)))
    min_items = max(2, int(current_app.config.get("COLLAB_MIN_ITEMS", 2)))

    model = None
    if (
        len(frame) >= min_ratings
        and frame["userId"].nunique() >= min_users
        and frame["movieId"].nunique() >= min_items
    ):
        try:
            model = CollaborativeRecommender(
                n_factors=int(current_app.config.get("N_FACTORS", 100)),
                n_epochs=int(current_app.config.get("COLLAB_N_EPOCHS", 20)),
                lr_all=float(current_app.config.get("COLLAB_LR_ALL", 0.005)),
                reg_all=float(current_app.config.get("COLLAB_REG_ALL", 0.02)),
                random_state=int(current_app.config.get("RANDOM_STATE", 42)),
            )
            model.fit(
                frame,
                _data_loader().get_movies(),
                test_size=float(current_app.config.get("TEST_SIZE", 0.2)),
            )
        except Exception:
            logger.exception("Persisted collaborative model build failed; using content-only mode")
            model = None

    current_app._collaborative_signature = signature
    current_app._collaborative_recommender = model
    return model


def invalidate_collaborative_model() -> None:
    """Force the next personalized request to rebuild collaborative state."""
    current_app._collaborative_signature = None
    current_app._collaborative_recommender = None


def _persisted_popular_candidates(limit: int, exclude_ids=None) -> List[Dict]:
    """Rank movies using real application-user interactions, never baseline CSV ratings."""
    excluded = {int(value) for value in (exclude_ids or [])}
    rows = db.session.execute(
        db.select(
            Rating.movie_id,
            func.count(Rating.id).label("rating_count"),
            func.avg(Rating.rating).label("average_rating"),
        )
        .group_by(Rating.movie_id)
        .order_by(func.count(Rating.id).desc(), func.avg(Rating.rating).desc(), Rating.movie_id.asc())
        .limit(max(1, int(limit)) + len(excluded))
    ).all()

    result = []
    for movie_id, rating_count, average_rating in rows:
        movie_id = int(movie_id)
        if movie_id in excluded:
            continue
        movie = movie_service.get_movie_by_id(movie_id, with_tmdb=False)
        if movie is None:
            continue
        result.append(
            {
                "movie": movie,
                "score": min(1.0, max(0.0, float(average_rating or 0.0) / 5.0)),
                "reason": (
                    f"Popular with registered users: {int(rating_count)} rating"
                    f"{'s' if int(rating_count) != 1 else ''}, average {float(average_rating):.1f}/5."
                ),
            }
        )
        if len(result) >= max(1, int(limit)):
            break
    return result


def _catalog_fallback(limit: int, exclude_ids=None) -> List[Dict]:
    """Deterministic cold-start fallback from catalog metadata, not baseline ratings."""
    excluded = {int(value) for value in (exclude_ids or [])}
    frame = _data_loader().get_movies().copy()
    if "title" in frame.columns:
        frame = frame.sort_values(["title", "movieId"], ascending=[True, True], na_position="last")
    result = []
    for _, row in frame.iterrows():
        movie = row.to_dict()
        movie_id = int(movie["movieId"])
        if movie_id in excluded:
            continue
        result.append(
            {
                "movie": movie,
                "score": 0.0,
                "reason": "Catalog fallback while more real user-rating evidence is collected.",
            }
        )
        if len(result) >= max(1, int(limit)):
            break
    return result


def _user_content_candidates(user_id: int, limit: int) -> List[Dict]:
    ratings = user_service.get_user_rating_records(int(user_id))
    if not ratings:
        return []

    minimum_seed = float(current_app.config.get("PERSONALIZATION_MIN_RATING", 3.5))
    positive = [item for item in ratings if float(item["rating"]) >= minimum_seed]
    sources = positive or sorted(ratings, key=lambda item: item["rating"], reverse=True)[:3]
    sources = sorted(
        sources,
        key=lambda item: (float(item["rating"]), item.get("timestamp") or 0),
        reverse=True,
    )[:10]

    combined: Dict[int, Dict] = {}
    per_source = max(10, min(60, int(limit)))
    for source in sources:
        source_id = int(source["movieId"])
        source_movie = movie_service.get_movie_by_id(source_id, with_tmdb=False) or {}
        source_title = source_movie.get("title", f"movie {source_id}")
        rating_weight = max(0.1, (float(source["rating"]) - 0.5) / 4.5)
        for rec in _content_candidates(source_id, per_source):
            candidate_id = rec["movie"]["movieId"]
            entry = combined.setdefault(
                candidate_id,
                {
                    "movie": rec["movie"],
                    "weighted_sum": 0.0,
                    "weight_sum": 0.0,
                    "sources": [],
                },
            )
            score = min(1.0, max(0.0, float(rec.get("score") or 0.0)))
            entry["weighted_sum"] += score * rating_weight
            entry["weight_sum"] += rating_weight
            entry["sources"].append(
                {
                    "title": source_title,
                    "rating": float(source["rating"]),
                    "score": score,
                }
            )

    result = []
    for entry in combined.values():
        entry["sources"].sort(key=lambda item: (item["rating"], item["score"]), reverse=True)
        top = entry["sources"][:2]
        source_text = ", ".join(f"{item['title']} ({item['rating']:.1f}/5)" for item in top)
        result.append(
            {
                "movie": entry["movie"],
                "score": entry["weighted_sum"] / max(entry["weight_sum"], 1e-9),
                "reason": f"Content signal from movies you rated highly: {source_text}.",
            }
        )
    result.sort(key=lambda item: (float(item["score"]), item["movie"].get("title", "")), reverse=True)
    return result[: max(1, int(limit))]


def _fuser(collab_model: Optional[CollaborativeRecommender], has_content: bool) -> HybridRecommender:
    content_weight = float(current_app.config.get("CONTENT_WEIGHT", 0.5))
    collab_weight = float(current_app.config.get("COLLAB_WEIGHT", 0.5))
    if collab_model is None:
        content_weight, collab_weight = (1.0, 0.0)
    elif not has_content:
        content_weight, collab_weight = (0.0, 1.0)
    return HybridRecommender(
        content_recommender=getattr(current_app, "recommender", None),
        collaborative_recommender=collab_model,
        content_weight=content_weight,
        collab_weight=collab_weight,
    )


def get_unique_genres() -> List[str]:
    return movie_service.get_unique_genres()


def get_similar_movies(
    movie_id: int,
    page: int = 1,
    per_page: int = 12,
    method: str = "content",
) -> Tuple[List[Dict], int]:
    """Public movie similarity remains content-based without a user identity."""
    candidates = _content_candidates(movie_id, max(page * per_page, per_page * 3))
    return _paginate(candidates, page, per_page)


def get_recommendations_for_user(
    user_id: int,
    page: int = 1,
    per_page: int = 24,
    strategy: str = "weighted",
) -> Tuple[List[Dict], int]:
    """Return personalized hybrid recommendations from persisted user interactions."""
    strategy = strategy if strategy in VALID_STRATEGIES else "weighted"
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    required = page * per_page
    pool_size = min(
        500,
        max(required * 4, int(current_app.config.get("COLLAB_CANDIDATE_POOL", 200))),
    )

    ratings = user_service.get_user_rating_records(int(user_id))
    rated_ids = {int(item["movieId"]) for item in ratings}
    if not ratings:
        cold = _persisted_popular_candidates(required, exclude_ids=rated_ids)
        if not cold:
            cold = _catalog_fallback(required, exclude_ids=rated_ids)
        return _paginate(cold, page, per_page)

    content_recs = _user_content_candidates(int(user_id), pool_size)
    popular = _persisted_popular_candidates(pool_size, exclude_ids=rated_ids)
    candidate_ids = {
        int(rec["movie"]["movieId"])
        for rec in content_recs + popular
        if isinstance(rec.get("movie"), dict)
    }

    collab_model = _get_collaborative_model()
    collab_recs = []
    if collab_model is not None and collab_model.knows_user(int(user_id)):
        collab_recs = collab_model.get_recommendations(
            int(user_id),
            n=pool_size,
            candidate_ids=candidate_ids or None,
        )

    if not content_recs and not collab_recs:
        fallback = popular or _catalog_fallback(required, exclude_ids=rated_ids)
        return _paginate(fallback, page, per_page)

    hybrid = _fuser(collab_model, has_content=bool(content_recs))
    ranked = hybrid.combine(
        content_recs,
        collab_recs,
        n=pool_size,
        strategy=strategy,
        exclude_ids=rated_ids,
    )
    return _paginate(ranked, page, per_page)


def get_recommendations_for_movie(
    movie_id: int,
    user_id: Optional[int] = None,
    page: int = 1,
    per_page: int = 12,
    strategy: str = "weighted",
) -> Tuple[List[Dict], int]:
    strategy = strategy if strategy in VALID_STRATEGIES else "weighted"
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    pool_size = min(300, max(page * per_page * 4, per_page * 4))
    content_recs = _content_candidates(int(movie_id), pool_size)

    rated_ids = {int(movie_id)}
    collab_recs = []
    collab_model = None
    if user_id is not None:
        rated_ids |= {
            int(item["movieId"])
            for item in user_service.get_user_rating_records(int(user_id))
        }
        collab_model = _get_collaborative_model()
        if collab_model is not None and collab_model.knows_user(int(user_id)):
            candidate_ids = [rec["movie"]["movieId"] for rec in content_recs]
            collab_recs = collab_model.get_recommendations(
                int(user_id),
                n=pool_size,
                candidate_ids=candidate_ids,
            )

    hybrid = _fuser(collab_model, has_content=bool(content_recs))
    ranked = hybrid.combine(
        content_recs,
        collab_recs,
        n=pool_size,
        strategy=strategy,
        exclude_ids=rated_ids,
    )
    return _paginate(ranked, page, per_page)


def rate_movie(user_id: int, movie_id: int, rating: float) -> bool:
    result = user_service.add_user_rating(user_id, movie_id, rating)
    if result.get("success"):
        invalidate_collaborative_model()
    return bool(result.get("success"))


def get_user_ratings(user_id: int) -> List[Dict]:
    return user_service.get_user_rating_records(user_id)


def get_recommendation_explanation(
    movie_id: int,
    user_id: Optional[int] = None,
    source_id: Optional[int] = None,
) -> Dict:
    """Explain recommendation evidence from content, preference, and collaborative signals."""
    movie = movie_service.get_movie_by_id(int(movie_id), with_tmdb=False)
    if movie is None:
        return {"error": "Movie not found"}

    reasons = []
    signals: Dict[str, Optional[float]] = {"content": None, "collaborative_prediction": None}
    genre_match = []
    source_movie = None

    if source_id is not None:
        source_movie = movie_service.get_movie_by_id(int(source_id), with_tmdb=False)
        for rec in _content_candidates(int(source_id), 100):
            if rec["movie"]["movieId"] == int(movie_id):
                signals["content"] = round(float(rec.get("score") or 0.0), 6)
                reasons.append(rec.get("reason") or "Content similarity contributed to this result.")
                break

    if user_id is not None:
        favorites = user_service.get_user_favorite_genres(int(user_id), top_n=5)
        movie_genres = set(str(movie.get("genres", "")).split("|"))
        genre_match = [item for item in favorites if item["genre"] in movie_genres]
        if genre_match:
            reasons.append(
                "Preference signal: matches genres you tend to rate highly: "
                + ", ".join(item["genre"] for item in genre_match)
                + "."
            )

        collab_model = _get_collaborative_model()
        if (
            collab_model is not None
            and collab_model.knows_user(int(user_id))
            and collab_model.knows_movie(int(movie_id))
        ):
            prediction = collab_model.predict_rating(int(user_id), int(movie_id))
            signals["collaborative_prediction"] = round(float(prediction), 4)
            reasons.append(
                f"Collaborative signal: persisted user-rating patterns predict about {prediction:.1f}/5."
            )

    if not reasons:
        reasons.append("Selected from the available content and catalog signals.")

    return {
        "movie": movie,
        "source_movie": source_movie,
        "reasons": reasons,
        "genre_match": genre_match,
        "signals": signals,
    }


def get_movie_popularity_score(movie_id: int) -> float:
    """Return a 0-10 popularity score from persisted application-user ratings only."""
    count, average = db.session.execute(
        db.select(func.count(Rating.id), func.avg(Rating.rating)).where(
            Rating.movie_id == int(movie_id)
        )
    ).one()
    if not count:
        return 0.0
    # Rating quality dominates; evidence count adds a bounded confidence boost.
    quality = float(average or 0.0) / 5.0
    confidence = min(1.0, int(count) / 10.0)
    return round(min(10.0, 10.0 * quality * (0.75 + 0.25 * confidence)), 1)


# Backward-compatible aliases used by older modules/tests.
def get_user_recommendations(user_id: int, limit: int = 10, method: str = "hybrid") -> List[Dict]:
    items, _ = get_recommendations_for_user(user_id, page=1, per_page=limit)
    return items


def get_recommendations_from_user_history(user_id: int, limit: int = 10) -> List[Dict]:
    return get_user_recommendations(user_id, limit=limit)


def get_explanation(user_id: int, movie_id: int) -> Dict:
    return get_recommendation_explanation(movie_id, user_id=user_id)
