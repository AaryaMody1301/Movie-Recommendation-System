"""Persistent user profile, rating, and watchlist services.

Application-user interactions are stored in SQLAlchemy.  The CSV ``DataLoader`` is
used only as the movie catalog / baseline dataset and is never mutated by this
module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from math import sqrt
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import current_app
from sqlalchemy import func

from database.db import db
from database.models import Rating, User, Watchlist
import services.movie_service as movie_service

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_data_loader():
    """Return the application-owned DataLoader instance."""
    loader = getattr(current_app, "data_loader", None)
    if loader is None:
        raise RuntimeError("Application DataLoader is not available")
    return loader


def _movie_exists(movie_id: int) -> bool:
    return _get_data_loader().get_movie_by_id(int(movie_id)) is not None


def _movie_dict(movie_id: int) -> Optional[Dict]:
    """Return local catalog details without triggering remote TMDb enrichment."""
    return movie_service.get_movie_by_id(int(movie_id), with_tmdb=False)


def _to_timestamp(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def get_user_rating_records(user_id: int) -> List[Dict]:
    """Return all persisted ratings for a user in a lightweight service format."""
    rows = db.session.execute(
        db.select(Rating)
        .where(Rating.user_id == int(user_id))
        .order_by(Rating.timestamp.desc())
    ).scalars().all()
    return [
        {
            "userId": row.user_id,
            "movieId": row.movie_id,
            "rating": float(row.rating),
            "timestamp": _to_timestamp(row.timestamp),
            "date": _to_iso(row.timestamp),
        }
        for row in rows
    ]


def get_user_profile(user_id: int) -> Optional[Dict]:
    """Return account and interaction summary data for an application user."""
    user = db.session.get(User, int(user_id))
    if user is None:
        return None

    statistics = get_user_statistics(user.id)
    latest_rating = db.session.execute(
        db.select(Rating)
        .where(Rating.user_id == user.id)
        .order_by(Rating.timestamp.desc())
        .limit(1)
    ).scalar_one_or_none()

    last_active = user.last_login
    if latest_rating and (last_active is None or latest_rating.timestamp > last_active):
        last_active = latest_rating.timestamp

    return {
        "userId": user.id,
        "username": user.username,
        "email": user.email,
        "memberSince": _to_iso(user.created_at),
        "lastActive": _to_iso(last_active),
        "ratingCount": statistics["totalRatings"],
        "ratedMovieCount": statistics["totalRatings"],
        "averageRating": statistics["averageRating"],
        "favoriteGenres": statistics["favoriteGenres"],
        "watchlistCount": statistics["watchlistCount"],
    }


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Backward-compatible alias for profile-oriented callers."""
    return get_user_profile(user_id)


def get_user_ratings(
    user_id: int,
    page: int = 1,
    per_page: int = 24,
    sort_by: str = "date",
    sort_order: str = "desc",
) -> Tuple[List[Dict], int]:
    """Return paginated persisted ratings enriched with local movie details."""
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    sort_map = {
        "date": Rating.timestamp,
        "timestamp": Rating.timestamp,
        "rating": Rating.rating,
        "movie_id": Rating.movie_id,
        "movieId": Rating.movie_id,
    }
    sort_column = sort_map.get(sort_by, Rating.timestamp)
    ordering = sort_column.asc() if str(sort_order).lower() == "asc" else sort_column.desc()

    total = db.session.execute(
        db.select(func.count(Rating.id)).where(Rating.user_id == int(user_id))
    ).scalar_one()
    rows = db.session.execute(
        db.select(Rating)
        .where(Rating.user_id == int(user_id))
        .order_by(ordering)
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).scalars().all()

    result = []
    for row in rows:
        movie = _movie_dict(row.movie_id)
        if movie is None:
            continue
        result.append(
            {
                "id": row.id,
                "userId": row.user_id,
                "movieId": row.movie_id,
                "rating": float(row.rating),
                "timestamp": _to_timestamp(row.timestamp),
                "date": _to_iso(row.timestamp),
                "movie": movie,
            }
        )
    return result, int(total)


def get_user_favorite_genres(user_id: int, top_n: int = 3) -> List[Dict]:
    """Calculate favorite genres from persisted ratings and the local catalog."""
    ratings = get_user_rating_records(user_id)
    if not ratings:
        return []

    movies_df = _get_data_loader().get_movies()
    ratings_df = pd.DataFrame(ratings)[["movieId", "rating"]]
    rated_movies = movies_df[movies_df["movieId"].isin(ratings_df["movieId"])][
        ["movieId", "genres"]
    ].copy()
    merged = rated_movies.merge(ratings_df, on="movieId", how="inner")

    genre_ratings = []
    for _, row in merged.iterrows():
        genres = row["genres"].split("|") if isinstance(row["genres"], str) else []
        for genre in genres:
            if genre and genre != "(no genres listed)":
                genre_ratings.append({"genre": genre, "rating": float(row["rating"])})

    if not genre_ratings:
        return []

    genre_df = pd.DataFrame(genre_ratings)
    stats = (
        genre_df.groupby("genre")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "averageRating"})
    )
    # Deterministic, interpretable weighting: average rating first, then evidence count.
    stats["score"] = stats["averageRating"] * (1 + stats["count"] / stats["count"].max())
    stats = stats.sort_values(["score", "count", "genre"], ascending=[False, False, True])

    return [
        {
            "genre": row.genre,
            "averageRating": round(float(row.averageRating), 2),
            "count": int(row["count"]),
            "score": round(float(row.score), 3),
        }
        for _, row in stats.head(max(1, int(top_n))).iterrows()
    ]


def add_user_rating(user_id: int, movie_id: int, rating: float) -> Dict:
    """Create or update a persisted user rating."""
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)
        rating = float(rating)
    except (TypeError, ValueError):
        return {"success": False, "error": "Invalid user, movie, or rating value"}

    if not 0.5 <= rating <= 5.0:
        return {"success": False, "error": "Rating must be between 0.5 and 5.0"}
    if db.session.get(User, user_id) is None:
        return {"success": False, "error": "User not found"}
    if not _movie_exists(movie_id):
        return {"success": False, "error": f"Movie with ID {movie_id} not found"}

    row = db.session.execute(
        db.select(Rating).where(
            Rating.user_id == user_id,
            Rating.movie_id == movie_id,
        )
    ).scalar_one_or_none()
    action = "updated" if row else "added"
    if row is None:
        row = Rating(user_id=user_id, movie_id=movie_id, rating=rating)
        db.session.add(row)
    else:
        row.rating = rating
        row.timestamp = _utcnow()

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Failed to persist rating")
        return {"success": False, "error": "Could not save rating"}

    return {
        "success": True,
        "action": action,
        "rating": {
            "id": row.id,
            "userId": row.user_id,
            "movieId": row.movie_id,
            "rating": float(row.rating),
            "timestamp": _to_timestamp(row.timestamp),
            "date": _to_iso(row.timestamp),
        },
        "movie": _movie_dict(movie_id),
    }


def delete_user_rating(user_id: int, movie_id: int) -> Dict:
    """Delete a persisted user rating."""
    row = db.session.execute(
        db.select(Rating).where(
            Rating.user_id == int(user_id),
            Rating.movie_id == int(movie_id),
        )
    ).scalar_one_or_none()
    if row is None:
        return {"success": False, "error": "Rating not found"}

    db.session.delete(row)
    db.session.commit()
    return {"success": True, "message": "Rating deleted"}


def get_user_activity(user_id: int, limit: int = 20) -> List[Dict]:
    """Return recent rating activity for the user."""
    ratings, _ = get_user_ratings(user_id, page=1, per_page=max(1, int(limit)))
    return [
        {
            "type": "rating",
            "userId": item["userId"],
            "timestamp": item["timestamp"],
            "date": item["date"],
            "details": {
                "movieId": item["movieId"],
                "rating": item["rating"],
                "movieTitle": item["movie"].get("title"),
            },
        }
        for item in ratings
    ]


def get_user_statistics(user_id: int) -> Dict:
    """Return persisted interaction statistics for a user."""
    ratings = get_user_rating_records(user_id)
    values = [item["rating"] for item in ratings]
    distribution = {str(value / 2): 0 for value in range(1, 11)}
    for value in values:
        bucket = round(value * 2) / 2
        distribution[str(bucket)] = distribution.get(str(bucket), 0) + 1

    watchlist_count = db.session.execute(
        db.select(func.count(Watchlist.id)).where(Watchlist.user_id == int(user_id))
    ).scalar_one()

    activity_by_month: Dict[str, int] = {}
    for item in ratings:
        if item["date"]:
            month = item["date"][:7]
            activity_by_month[month] = activity_by_month.get(month, 0) + 1

    return {
        "userId": int(user_id),
        "totalRatings": len(values),
        "averageRating": round(sum(values) / len(values), 2) if values else 0.0,
        "ratingDistribution": distribution,
        "favoriteGenres": get_user_favorite_genres(user_id, top_n=10) if values else [],
        "activityByMonth": [
            {"month": month, "count": count}
            for month, count in sorted(activity_by_month.items())
        ],
        "watchlistCount": int(watchlist_count),
    }


def get_user_watchlist(
    user_id: int,
    page: int = 1,
    per_page: int = 24,
    sort_by: str = "date",
    sort_order: str = "desc",
) -> Tuple[List[Dict], int]:
    """Return a user's persisted watchlist with pagination."""
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    sort_map = {
        "date": Watchlist.added_at,
        "added_at": Watchlist.added_at,
        "movie_id": Watchlist.movie_id,
        "movieId": Watchlist.movie_id,
    }
    sort_column = sort_map.get(sort_by, Watchlist.added_at)
    ordering = sort_column.asc() if str(sort_order).lower() == "asc" else sort_column.desc()

    total = db.session.execute(
        db.select(func.count(Watchlist.id)).where(Watchlist.user_id == int(user_id))
    ).scalar_one()
    rows = db.session.execute(
        db.select(Watchlist)
        .where(Watchlist.user_id == int(user_id))
        .order_by(ordering)
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).scalars().all()

    items = []
    for row in rows:
        movie = _movie_dict(row.movie_id)
        if movie is None:
            continue
        items.append(
            {
                "id": row.id,
                "userId": row.user_id,
                "movieId": row.movie_id,
                "addedAt": _to_iso(row.added_at),
                "timestamp": _to_timestamp(row.added_at),
                "notes": row.notes or "",
                "movie": movie,
            }
        )
    return items, int(total)


def add_to_watchlist(user_id: int, movie_id: int, notes: str = "") -> Dict:
    """Persist a movie on a user's watchlist."""
    user_id = int(user_id)
    movie_id = int(movie_id)
    if db.session.get(User, user_id) is None:
        return {"success": False, "error": "User not found"}
    if not _movie_exists(movie_id):
        return {"success": False, "error": "Movie not found"}

    existing = db.session.execute(
        db.select(Watchlist).where(
            Watchlist.user_id == user_id,
            Watchlist.movie_id == movie_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return {"success": False, "error": "Movie is already in the watchlist"}

    row = Watchlist(user_id=user_id, movie_id=movie_id, notes=(notes or "").strip() or None)
    db.session.add(row)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Failed to add watchlist item")
        return {"success": False, "error": "Could not add watchlist item"}
    return {"success": True, "message": "Added to watchlist", "id": row.id}


def remove_from_watchlist(user_id: int, movie_id: int) -> Dict:
    """Remove a movie from a user's watchlist."""
    row = db.session.execute(
        db.select(Watchlist).where(
            Watchlist.user_id == int(user_id),
            Watchlist.movie_id == int(movie_id),
        )
    ).scalar_one_or_none()
    if row is None:
        return {"success": False, "error": "Watchlist item not found"}
    db.session.delete(row)
    db.session.commit()
    return {"success": True, "message": "Removed from watchlist"}


def update_watchlist_notes(user_id: int, movie_id: int, notes: str) -> Dict:
    """Update notes for a persisted watchlist item."""
    row = db.session.execute(
        db.select(Watchlist).where(
            Watchlist.user_id == int(user_id),
            Watchlist.movie_id == int(movie_id),
        )
    ).scalar_one_or_none()
    if row is None:
        return {"success": False, "error": "Watchlist item not found"}
    row.notes = (notes or "").strip() or None
    db.session.commit()
    return {"success": True, "message": "Watchlist notes updated"}


def calculate_user_similarity(user_id1: int, user_id2: int) -> float:
    """Calculate cosine similarity between two persisted user-rating vectors."""
    ratings1 = {item["movieId"]: item["rating"] for item in get_user_rating_records(user_id1)}
    ratings2 = {item["movieId"]: item["rating"] for item in get_user_rating_records(user_id2)}
    common = sorted(set(ratings1).intersection(ratings2))
    if len(common) < 3:
        return 0.0

    vector1 = [ratings1[movie_id] for movie_id in common]
    vector2 = [ratings2[movie_id] for movie_id in common]
    dot = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = sqrt(sum(value * value for value in vector1))
    magnitude2 = sqrt(sum(value * value for value in vector2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return max(0.0, min(1.0, float(dot / (magnitude1 * magnitude2))))


def find_similar_users(user_id: int, limit: int = 10) -> List[Dict]:
    """Return application users with overlapping persisted rating histories."""
    other_ids = db.session.execute(
        db.select(Rating.user_id).where(Rating.user_id != int(user_id)).distinct()
    ).scalars().all()
    similarities = [
        {"userId": int(other_id), "similarity": round(calculate_user_similarity(user_id, other_id), 3)}
        for other_id in other_ids
    ]
    similarities = [item for item in similarities if item["similarity"] > 0]
    similarities.sort(key=lambda item: item["similarity"], reverse=True)
    return similarities[: max(1, int(limit))]
