"""Application factory for the Movie Recommendation System."""

import logging
import os
import time

import dotenv
from flask import Flask, g, render_template, request
from flask_caching import Cache
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from blueprints import register_blueprints
from config import INSECURE_SECRET_KEYS, get_config
from data.data_loader import DataLoader
from database.db import init_app as init_database
from models.content_based import ContentBasedRecommender
from observability import configure_logging
from services.auth_service import get_user_by_id
import services.movie_service as movie_service

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

cache = Cache()
csrf = CSRFProtect()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"


def _validate_runtime_config(app):
    """Fail closed for unsafe or ambiguous production configuration."""
    if app.config.get("ENVIRONMENT") != "production":
        return

    secret_key = app.config.get("SECRET_KEY")
    if not secret_key or secret_key in INSECURE_SECRET_KEYS:
        raise RuntimeError(
            "Production requires SECRET_KEY to be set to a private, non-development value"
        )


def _initialize_recommender(app, embedding_args):
    """Initialize the catalog first, then attach the optional content recommender."""
    settings = {
        "rebuild_embeddings": False,
        "batch_size": app.config.get("EMBEDDING_BATCH_SIZE", 32),
        "cache_path": app.config.get("EMBEDDINGS_CACHE_PATH", "instance/embeddings_cache.pkl"),
    }
    if embedding_args:
        settings.update({key: value for key, value in embedding_args.items() if value is not None})

    try:
        batch_size = max(1, int(settings.get("batch_size", 32)))
    except (TypeError, ValueError):
        batch_size = 32

    try:
        app.data_loader = DataLoader(
            movies_path=app.config["MOVIES_CSV"],
            ratings_path=app.config.get("RATINGS_CSV"),
        )
    except Exception:
        logger.exception("Catalog data initialization failed")
        app.data_loader = None
        app.recommender = None
        return

    movies_df = app.data_loader.get_movies()
    required_columns = {"movieId", "title", "genres", "clean_title", "overview"}
    if not required_columns.issubset(movies_df.columns):
        missing = sorted(required_columns.difference(movies_df.columns))
        logger.error("Movies dataset is missing required columns: %s", missing)
        app.recommender = None
        return

    if not app.config.get("RECOMMENDER_ENABLED", True):
        app.recommender = None
        logger.info("Content recommender initialization is disabled by configuration")
        return

    try:
        app.recommender = ContentBasedRecommender(
            transformer_model=app.config.get(
                "TRANSFORMER_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        app.recommender.fit(
            movies_df,
            force_rebuild=bool(settings.get("rebuild_embeddings", False)),
            cache_path=str(settings.get("cache_path") or app.config["EMBEDDINGS_CACHE_PATH"]),
            batch_size=batch_size,
        )
    except Exception:
        logger.exception(
            "Content recommender initialization failed; catalog browsing remains available"
        )
        app.recommender = None


def create_app(test_config=None, embedding_args=None):
    """Create and configure a Flask application instance."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(get_config())

    app.config.from_pyfile("config.py", silent=True)

    if test_config:
        app.config.from_mapping(test_config)

    _validate_runtime_config(app)

    configure_logging(
        app.config.get("LOG_LEVEL", "INFO"),
        app.config.get("LOG_FORMAT", "text"),
    )

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
        g.request_started_at = time.perf_counter()

    @app.after_request
    def log_request(response):
        started_at = getattr(g, "request_started_at", None)
        duration_ms = (
            (time.perf_counter() - started_at) * 1000.0
            if started_at is not None
            else 0.0
        )
        logger.info(
            "request_complete method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        return response

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
