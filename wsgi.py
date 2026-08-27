"""WSGI entry point for production servers such as Gunicorn."""

import os

from app import create_app


app = create_app(
    embedding_args={
        "rebuild_embeddings": False,
        "max_movies": int(os.environ.get("MAX_EMBEDDING_MOVIES", "1000")),
    }
)
