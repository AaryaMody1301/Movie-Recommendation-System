"""Database models for application users, interactions, and durable TMDb metadata."""

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from database.db import db


def _utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    """Application user used by authentication and personalization features."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow)
    last_login = db.Column(db.DateTime(timezone=True), nullable=True)

    ratings = db.relationship(
        "Rating",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    watchlist_items = db.relationship(
        "Watchlist",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class Rating(db.Model):
    """A rating supplied by an application user for a local movie ID."""

    __tablename__ = "ratings"
    __table_args__ = (
        db.UniqueConstraint("user_id", "movie_id", name="user_movie_rating"),
        db.CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="rating_range"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    movie_id = db.Column(db.Integer, nullable=False, index=True)
    rating = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow, index=True)

    user = db.relationship("User", back_populates="ratings")

    def __repr__(self):
        return f"<Rating user_id={self.user_id} movie_id={self.movie_id} rating={self.rating}>"


class Watchlist(db.Model):
    """A movie saved to an application user's watchlist."""

    __tablename__ = "watchlist"
    __table_args__ = (
        db.UniqueConstraint("user_id", "movie_id", name="user_movie_watchlist"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    movie_id = db.Column(db.Integer, nullable=False, index=True)
    added_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    notes = db.Column(db.Text, nullable=True)

    user = db.relationship("User", back_populates="watchlist_items")

    def __repr__(self):
        return f"<Watchlist user_id={self.user_id} movie_id={self.movie_id}>"


class MovieTmdbMapping(db.Model):
    """Durable mapping between a local catalog movie ID and a TMDb movie ID.

    ``catalog_key`` fingerprints the local title/year/external-ID inputs used for the
    match.  If those catalog inputs change, the mapping is re-resolved instead of
    silently trusting stale identity data.  ``status=not_found`` is a negative cache
    entry and is intentionally assigned a shorter expiry by the service layer.
    """

    __tablename__ = "movie_tmdb_mappings"

    local_movie_id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, nullable=True, index=True)
    catalog_key = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="resolved", index=True)
    matched_by = db.Column(db.String(32), nullable=True)
    checked_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True)

    def __repr__(self):
        return (
            f"<MovieTmdbMapping local_movie_id={self.local_movie_id} "
            f"tmdb_id={self.tmdb_id} status={self.status}>"
        )


class TmdbEnrichmentCache(db.Model):
    """Durable normalized TMDb detail/provider payload for one locale/region."""

    __tablename__ = "tmdb_enrichment_cache"
    __table_args__ = (
        db.UniqueConstraint(
            "tmdb_id",
            "language",
            "region",
            name="tmdb_enrichment_locale",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, nullable=False, index=True)
    language = db.Column(db.String(16), nullable=False, default="en-US")
    region = db.Column(db.String(8), nullable=False, default="IN")
    payload = db.Column(db.JSON, nullable=False)
    fetched_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True)

    def __repr__(self):
        return (
            f"<TmdbEnrichmentCache tmdb_id={self.tmdb_id} "
            f"language={self.language} region={self.region}>"
        )
