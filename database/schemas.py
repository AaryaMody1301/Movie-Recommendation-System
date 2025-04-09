"""
Marshmallow schemas for serialization.
"""
from marshmallow import Schema, fields, validate, ValidationError


class UserSchema(Schema):
    """Schema for User model."""
    id = fields.Int(dump_only=True)
    username = fields.Str(required=True, validate=validate.Length(min=3, max=64))
    email = fields.Email(required=True)
    password = fields.Str(load_only=True, required=True, validate=validate.Length(min=6))
    joined_at = fields.DateTime(dump_only=True)
    
    # For nesting
    ratings = fields.List(fields.Nested('RatingSchema', exclude=('user',)), dump_only=True)
    watchlist_items = fields.List(fields.Nested('WatchlistSchema', exclude=('user',)), dump_only=True)


class RatingSchema(Schema):
    """Schema for Rating model."""
    id = fields.Int(dump_only=True)
    user_id = fields.Int(required=True)
    movie_id = fields.Int(required=True)
    rating = fields.Float(required=True, validate=validate.Range(min=1, max=5))
    timestamp = fields.DateTime(dump_only=True)
    
    # For nesting
    user = fields.Nested(UserSchema, exclude=('ratings', 'watchlist_items'), dump_only=True)


class WatchlistSchema(Schema):
    """Schema for Watchlist model."""
    id = fields.Int(dump_only=True)
    user_id = fields.Int(required=True)
    movie_id = fields.Int(required=True)
    added_at = fields.DateTime(dump_only=True)
    notes = fields.Str()
    
    # For nesting
    user = fields.Nested(UserSchema, exclude=('ratings', 'watchlist_items'), dump_only=True)


class MovieSchema(Schema):
    """Schema for Movie data (from CSV)."""
    movieId = fields.Int(data_key='id')
    title = fields.Str()
    genres = fields.Str()
    
    # Additional fields if available in your CSV
    year = fields.Int(missing=None)
    poster_url = fields.Url(missing=None)
    overview = fields.Str(missing=None)


class RecommendationSchema(Schema):
    """Schema for movie recommendations."""
    movie = fields.Nested(MovieSchema)
    score = fields.Float()
    reason = fields.Str(missing=None)  # For "Why Recommended?" feature 