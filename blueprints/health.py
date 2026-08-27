"""Operational health endpoints for deployment probes and CI smoke tests."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify
from sqlalchemy import text

from database.db import db

logger = logging.getLogger(__name__)
health = Blueprint("health", __name__)


@health.get("/health/live")
def liveness():
    """Report that the Flask process can serve requests."""
    return jsonify(
        {
            "status": "ok",
            "service": "movie-recommendation-system",
            "version": current_app.config.get("APP_VERSION", "dev"),
        }
    ), 200


@health.get("/health/ready")
def readiness():
    """Verify critical dependencies required to serve the local catalog."""
    checks = {}
    ready = True

    try:
        db.session.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception:
        db.session.rollback()
        logger.exception("Readiness database check failed")
        checks["database"] = {"status": "error"}
        ready = False

    loader = getattr(current_app, "data_loader", None)
    if loader is None:
        checks["catalog"] = {"status": "error", "movies": 0}
        ready = False
    else:
        try:
            movie_count = len(loader.get_movies())
            checks["catalog"] = {
                "status": "ok" if movie_count > 0 else "error",
                "movies": int(movie_count),
            }
            if movie_count <= 0:
                ready = False
        except Exception:
            logger.exception("Readiness catalog check failed")
            checks["catalog"] = {"status": "error", "movies": 0}
            ready = False

    recommender = getattr(current_app, "recommender", None)
    recommender_enabled = bool(current_app.config.get("RECOMMENDER_ENABLED", True))
    if not recommender_enabled:
        recommender_status = "disabled"
    elif recommender is None:
        recommender_status = "degraded"
    else:
        recommender_status = "ok"
    checks["recommender"] = {"status": recommender_status}

    return jsonify(
        {
            "status": "ready" if ready else "not_ready",
            "service": "movie-recommendation-system",
            "version": current_app.config.get("APP_VERSION", "dev"),
            "checks": checks,
        }
    ), 200 if ready else 503
