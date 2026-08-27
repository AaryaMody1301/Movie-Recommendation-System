-- Reference schema for Movie Recommendation System.
-- Runtime table creation is handled by Flask-SQLAlchemy models in database/models.py.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);

CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);
CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    UNIQUE (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS ix_ratings_user_id ON ratings (user_id);
CREATE INDEX IF NOT EXISTS ix_ratings_movie_id ON ratings (movie_id);
CREATE INDEX IF NOT EXISTS ix_ratings_timestamp ON ratings (timestamp);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    UNIQUE (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS ix_watchlist_user_id ON watchlist (user_id);
CREATE INDEX IF NOT EXISTS ix_watchlist_movie_id ON watchlist (movie_id);
CREATE INDEX IF NOT EXISTS ix_watchlist_added_at ON watchlist (added_at);
