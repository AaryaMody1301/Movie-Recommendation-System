"""Smoke tests for the Phase 2 Flask and database architecture."""

from sqlalchemy import text

import app as app_module
from database.db import db
from database.models import User


def _create_test_app(monkeypatch, tmp_path):
    def skip_recommender(app, embedding_args):
        app.data_loader = None
        app.recommender = None

    monkeypatch.setattr(app_module, "_initialize_recommender", skip_recommender)
    return app_module.create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "phase2-test-secret",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "WTF_CSRF_ENABLED": False,
        }
    )


def test_canonical_blueprints_are_registered(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}

    assert "main.index" in endpoints
    assert "main.about" in endpoints
    assert "movies.search" in endpoints
    assert "movies.movie_detail" in endpoints
    assert "movies.genre" in endpoints
    assert "auth.login" in endpoints
    assert "auth.register" in endpoints
    assert "auth.logout" in endpoints

    root_rules = [rule for rule in app.url_map.iter_rules() if rule.rule == "/"]
    assert len(root_rules) == 1
    assert root_rules[0].endpoint == "main.index"


def test_sqlite_foreign_keys_are_enabled(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    with app.app_context():
        enabled = db.session.execute(text("PRAGMA foreign_keys")).scalar_one()
        assert enabled == 1


def test_registration_login_and_logout(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    client = app.test_client()

    register_response = client.post(
        "/register",
        data={
            "username": "phase2user",
            "email": "phase2@example.com",
            "password": "correct-horse-battery-staple",
            "confirm_password": "correct-horse-battery-staple",
        },
        follow_redirects=False,
    )
    assert register_response.status_code == 302
    assert register_response.headers["Location"].endswith("/login")

    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "phase2user")
        ).scalar_one()
        assert user.email == "phase2@example.com"
        assert user.check_password("correct-horse-battery-staple")
        assert user.password_hash != "correct-horse-battery-staple"

    login_response = client.post(
        "/login",
        data={
            "username": "phase2user",
            "password": "correct-horse-battery-staple",
        },
        follow_redirects=False,
    )
    assert login_response.status_code == 302

    logout_response = client.post("/logout", follow_redirects=False)
    assert logout_response.status_code == 302
