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

CREATE TABLE IF NOT EXISTS movie_tmdb_mappings (
    local_movie_id INTEGER PRIMARY KEY,
    tmdb_id INTEGER,
    catalog_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'resolved',
    matched_by TEXT,
    checked_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_movie_tmdb_mappings_tmdb_id ON movie_tmdb_mappings (tmdb_id);
CREATE INDEX IF NOT EXISTS ix_movie_tmdb_mappings_status ON movie_tmdb_mappings (status);
CREATE INDEX IF NOT EXISTS ix_movie_tmdb_mappings_expires_at ON movie_tmdb_mappings (expires_at);

CREATE TABLE IF NOT EXISTS tmdb_enrichment_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER NOT NULL,
    language TEXT NOT NULL DEFAULT 'en-US',
    region TEXT NOT NULL DEFAULT 'IN',
    payload JSON NOT NULL,
    fetched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    UNIQUE (tmdb_id, language, region)
);

CREATE INDEX IF NOT EXISTS ix_tmdb_enrichment_cache_tmdb_id ON tmdb_enrichment_cache (tmdb_id);
CREATE INDEX IF NOT EXISTS ix_tmdb_enrichment_cache_expires_at ON tmdb_enrichment_cache (expires_at);
