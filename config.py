import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # SQLite database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///instance/movie_recommender.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File paths
    MOVIES_CSV = os.environ.get('MOVIES_CSV', 'data/movies.csv')
    RATINGS_CSV = os.environ.get('RATINGS_CSV', 'data/ratings.csv')
    
    # Model settings
    CONTENT_MODEL_PATH = os.environ.get('CONTENT_MODEL_PATH', 'instance/content_model.pkl')
    COLLAB_MODEL_PATH = os.environ.get('COLLAB_MODEL_PATH', 'instance/collaborative_model.pkl')
    
    # Recommendation settings
    MAX_RECOMMENDATIONS = int(os.environ.get('MAX_RECOMMENDATIONS', 10))
    CONTENT_WEIGHT = float(os.environ.get('CONTENT_WEIGHT', 0.5))  # Weight for content-based recommendations in hybrid model
    COLLAB_WEIGHT = float(os.environ.get('COLLAB_WEIGHT', 0.5))    # Weight for collaborative recommendations in hybrid model
    
    # Model training settings
    TEST_SIZE = float(os.environ.get('TEST_SIZE', 0.2))            # Fraction of data to use for testing
    RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))         # Random seed for reproducibility
    N_FACTORS = int(os.environ.get('N_FACTORS', 100))              # Number of factors for matrix factorization
    
    # Sentence transformer model for content-based filtering
    TRANSFORMER_MODEL = os.environ.get('TRANSFORMER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # In production, SECRET_KEY should be set in environment variables
    

# Dictionary of configuration classes
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Function to get configuration class based on environment variable
def get_config():
    """
    Returns the appropriate configuration class based on the FLASK_ENV
    environment variable. Defaults to development if not set.
    """
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default']) 