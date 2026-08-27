"""Database models for application users and their movie interactions."""

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
