"""WSGI entry point for production servers such as Gunicorn."""

from app import create_app


app = create_app(
    embedding_args={
        "rebuild_embeddings": False,
    }
)
