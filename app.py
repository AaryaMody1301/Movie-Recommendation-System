"""Application factory for the Movie Recommendation System."""

import logging
import os

import dotenv
from flask import Flask, g, render_template
from flask_caching import Cache
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from blueprints import register_blueprints
from config import get_config
from data.data_loader import DataLoader
from database.db import init_app as init_database
from models.content_based import ContentBasedRecommender
from services.auth_service import get_user_by_id
import services.movie_service as movie_service

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

cache = Cache()
csrf = CSRFProtect()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"


def _initialize_recommender(app, embedding_args):
    """Initialize the app-scoped data loader and content recommender."""
    settings = {
        "rebuild_embeddings": False,
        "max_movies": app.config.get("MAX_EMBEDDING_MOVIES", 1000),
    }
    if embedding_args:
        settings.update({key: value for key, value in embedding_args.items() if value is not None})

    max_movies = settings.get("max_movies")
    if not isinstance(max_movies, int) or max_movies <= 0:
        max_movies = app.config.get("MAX_EMBEDDING_MOVIES", 1000)

    try:
        app.data_loader = DataLoader(
            movies_path=app.config["MOVIES_CSV"],
            ratings_path=app.config.get("RATINGS_CSV"),
        )
        movies_df = app.data_loader.get_movies()

        required_columns = {"movieId", "title", "genres", "clean_title", "overview"}
        if not required_columns.issubset(movies_df.columns):
            missing = sorted(required_columns.difference(movies_df.columns))
            raise ValueError(f"Movies dataset is missing required columns: {missing}")

        app.recommender = ContentBasedRecommender(
            transformer_model=app.config.get(
                "TRANSFORMER_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        app.recommender.fit(
            movies_df,
            max_items=max_movies,
            force_rebuild=bool(settings.get("rebuild_embeddings", False)),
        )
    except Exception:
        logger.exception("Data/recommender initialization failed")
        app.data_loader = None
        app.recommender = None


def create_app(test_config=None, embedding_args=None):
    """Create and configure a Flask application instance."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(get_config())

    # Optional instance-specific overrides apply to normal runtime configuration.
    app.config.from_pyfile("config.py", silent=True)

    # Explicit test configuration must win over both environment and instance files.
    if test_config:
        app.config.from_mapping(test_config)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    cache.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    init_database(app)

    movie_service.set_cache(cache)
    _initialize_recommender(app, embedding_args)

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return get_user_by_id(int(user_id))
        except (TypeError, ValueError):
            return None

    @app.before_request
    def expose_request_services():
        g.data_loader = getattr(app, "data_loader", None)
        g.recommender = getattr(app, "recommender", None)

    register_blueprints(app)

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template(
            "404.html",
            genres=movie_service.get_unique_genres(),
        ), 404

    @app.errorhandler(500)
    def server_error(error):
        logger.error("Unhandled server error: %s", error, exc_info=True)
        return render_template(
            "500.html",
            genres=movie_service.get_unique_genres(),
        ), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=application.config.get("DEBUG", False))
