"""WSGI entry point for production servers such as Gunicorn."""

import os

from app import create_app


environment = os.environ.get("FLASK_ENV", "").strip().lower()
if environment != "production":
    raise RuntimeError("wsgi:app requires FLASK_ENV=production")

app = create_app(
    embedding_args={
        "rebuild_embeddings": False,
    }
)
