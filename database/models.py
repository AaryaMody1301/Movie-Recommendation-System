"""
SQLAlchemy models for user data.
This is a stub implementation for Python 3.13 compatibility.
"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from database.db import db

# Stub models that don't rely on actual SQLAlchemy ORM functionality
class User(db.Model):
    """User model for authentication and profile management."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Rating:
    """User movie ratings."""
    __tablename__ = 'ratings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, user_id, movie_id, rating):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating
        self.timestamp = datetime.utcnow()
    
    def __repr__(self):
        return f'<Rating user_id={self.user_id} movie_id={self.movie_id} rating={self.rating}>'


class Watchlist:
    """User watchlist items."""
    __tablename__ = 'watchlist'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    
    def __init__(self, user_id, movie_id, notes=None):
        self.user_id = user_id
        self.movie_id = movie_id
        self.added_at = datetime.utcnow()
        self.notes = notes
    
    def __repr__(self):
        return f'<Watchlist user_id={self.user_id} movie_id={self.movie_id}>' 