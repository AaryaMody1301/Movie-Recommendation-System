"""Final hardening tests for production configuration and schema migrations."""

import pandas as pd
import pytest
from sqlalchemy import inspect, text

import app as app_module
from config import DEVELOPMENT_SECRET_KEY, get_config
from database.db import db


def _write_catalog(tmp_path):
    path = tmp_path / "movies.csv"
    pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Migration Test (2026)",
                "genres": "Drama",
                "overview": "Schema migration smoke test",
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_production_rejects_missing_or_development_secret(tmp_path):
    common = {
        "ENVIRONMENT": "production",
        "TESTING": True,
        "RECOMMENDER_ENABLED": False,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
    }

    with pytest.raises(RuntimeError, match="Production requires SECRET_KEY"):
        app_module.create_app(test_config={**common, "SECRET_KEY": None})

    with pytest.raises(RuntimeError, match="Production requires SECRET_KEY"):
        app_module.create_app(
            test_config={**common, "SECRET_KEY": DEVELOPMENT_SECRET_KEY}
        )


def test_unknown_flask_environment_fails_closed(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "definitely-not-a-real-environment")
    with pytest.raises(RuntimeError, match="Unsupported FLASK_ENV"):
        get_config()


def test_versioned_migration_creates_current_schema(tmp_path):
    database_path = tmp_path / "migration-test.db"
    app = app_module.create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "migration-test-secret",
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{database_path}",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "MOVIES_CSV": str(_write_catalog(tmp_path)),
            "RATINGS_CSV": "",
            "RECOMMENDER_ENABLED": False,
            "WTF_CSRF_ENABLED": False,
        }
    )

    with app.app_context():
        assert inspect(db.engine).get_table_names() == []

    runner = app.test_cli_runner()
    upgrade = runner.invoke(args=["db", "upgrade"])
    assert upgrade.exit_code == 0, upgrade.output

    with app.app_context():
        table_names = set(inspect(db.engine).get_table_names())
        assert {
            "alembic_version",
            "users",
            "ratings",
            "watchlist",
            "movie_tmdb_mappings",
            "tmdb_enrichment_cache",
        }.issubset(table_names)
        revision = db.session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        assert revision == "0001_initial"

    check = runner.invoke(args=["db", "check"])
    assert check.exit_code == 0, check.output
