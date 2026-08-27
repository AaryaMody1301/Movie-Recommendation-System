import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base application configuration."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    # Flask-SQLAlchemy resolves relative SQLite paths against Flask's instance path.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URI',
        'sqlite:///movie_recommender.db',
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', '300'))

    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'instance/uploads')
    MOVIES_CSV = os.environ.get('MOVIES_CSV', 'data/movies.csv')
    RATINGS_CSV = os.environ.get('RATINGS_CSV', 'data/ratings.csv')

    CONTENT_MODEL_PATH = os.environ.get(
        'CONTENT_MODEL_PATH',
        'instance/content_model.pkl',
    )
    EMBEDDINGS_CACHE_PATH = os.environ.get(
        'EMBEDDINGS_CACHE_PATH',
        'instance/embeddings_cache.pkl',
    )
    EMBEDDING_BATCH_SIZE = int(os.environ.get('EMBEDDING_BATCH_SIZE', '32'))
    # Retained for offline model-training/export tooling. Online collaborative
    # personalization is trained lazily from persisted SQLAlchemy ratings.
    COLLAB_MODEL_PATH = os.environ.get(
        'COLLAB_MODEL_PATH',
        'instance/collaborative_model.pkl',
    )

    MAX_RECOMMENDATIONS = int(os.environ.get('MAX_RECOMMENDATIONS', '10'))
    CONTENT_WEIGHT = float(os.environ.get('CONTENT_WEIGHT', '0.5'))
    COLLAB_WEIGHT = float(os.environ.get('COLLAB_WEIGHT', '0.5'))
    TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
    RANDOM_STATE = int(os.environ.get('RANDOM_STATE', '42'))
    N_FACTORS = int(os.environ.get('N_FACTORS', '100'))
    COLLAB_N_EPOCHS = int(os.environ.get('COLLAB_N_EPOCHS', '20'))
    COLLAB_LR_ALL = float(os.environ.get('COLLAB_LR_ALL', '0.005'))
    COLLAB_REG_ALL = float(os.environ.get('COLLAB_REG_ALL', '0.02'))
    COLLAB_MIN_RATINGS = int(os.environ.get('COLLAB_MIN_RATINGS', '5'))
    COLLAB_MIN_USERS = int(os.environ.get('COLLAB_MIN_USERS', '2'))
    COLLAB_MIN_ITEMS = int(os.environ.get('COLLAB_MIN_ITEMS', '2'))
    COLLAB_CANDIDATE_POOL = int(os.environ.get('COLLAB_CANDIDATE_POOL', '200'))
    PERSONALIZATION_MIN_RATING = float(os.environ.get('PERSONALIZATION_MIN_RATING', '3.5'))
    TRANSFORMER_MODEL = os.environ.get(
        'TRANSFORMER_MODEL',
        'sentence-transformers/all-MiniLM-L6-v2',
    )

    TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
    TMDB_LANGUAGE = os.environ.get('TMDB_LANGUAGE', 'en-US')
    TMDB_WATCH_REGION = os.environ.get('TMDB_WATCH_REGION', 'IN').upper()
    TMDB_REQUEST_TIMEOUT = float(os.environ.get('TMDB_REQUEST_TIMEOUT', '10'))
    TMDB_RETRY_TOTAL = int(os.environ.get('TMDB_RETRY_TOTAL', '2'))
    TMDB_RETRY_BACKOFF = float(os.environ.get('TMDB_RETRY_BACKOFF', '0.5'))
    # Short process-local HTTP response cache; normalized detail payloads below are
    # also persisted in SQLAlchemy and survive application restarts.
    TMDB_HTTP_CACHE_TTL = int(os.environ.get('TMDB_HTTP_CACHE_TTL', '900'))
    TMDB_ENRICHMENT_TTL = int(os.environ.get('TMDB_ENRICHMENT_TTL', '604800'))
    TMDB_STALE_CACHE_TTL = int(os.environ.get('TMDB_STALE_CACHE_TTL', '2592000'))
    TMDB_MAPPING_TTL = int(os.environ.get('TMDB_MAPPING_TTL', '2592000'))
    TMDB_NEGATIVE_MAPPING_TTL = int(os.environ.get('TMDB_NEGATIVE_MAPPING_TTL', '86400'))

    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}


def get_config():
    """Return the configuration class selected by FLASK_ENV."""
    env = os.environ.get('FLASK_ENV', 'development').strip().lower()
    return config.get(env, config['default'])
