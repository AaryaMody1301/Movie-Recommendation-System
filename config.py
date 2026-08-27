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
    TRANSFORMER_MODEL = os.environ.get(
        'TRANSFORMER_MODEL',
        'sentence-transformers/all-MiniLM-L6-v2',
    )
    MAX_EMBEDDING_MOVIES = int(os.environ.get('MAX_EMBEDDING_MOVIES', '1000'))

    TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
    TMDB_REQUEST_TIMEOUT = float(os.environ.get('TMDB_REQUEST_TIMEOUT', '10'))

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
