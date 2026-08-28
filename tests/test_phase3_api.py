"""Route-level tests for Phase 3 persisted interaction APIs."""

import pandas as pd

import app as app_module
from data.data_loader import DataLoader
from database.db import db
from database.models import Rating, User, Watchlist


class TinyRecommender:
    def __init__(self, loader):
        self.loader = loader

    def get_recommendations(self, movie_id, top_n=10):
        rows = [
            row
            for row in self.loader.get_movies().to_dict("records")
            if row["movieId"] != movie_id
        ]
        return [
            {"movie": row, "score": 0.8, "reason": "test recommendation"}
            for row in rows[:top_n]
        ]


def _app(monkeypatch, tmp_path):
    movies = tmp_path / "movies.csv"
    ratings = tmp_path / "ratings.csv"
    pd.DataFrame(
        [
            {"movieId": 1, "title": "Alpha (2020)", "genres": "Action"},
            {"movieId": 2, "title": "Beta (2021)", "genres": "Drama"},
        ]
    ).to_csv(movies, index=False)
    pd.DataFrame(
        [{"userId": 99, "movieId": 1, "rating": 4.0, "timestamp": 1}]
    ).to_csv(ratings, index=False)

    def initialize(app, embedding_args):
        app.data_loader = DataLoader(str(movies), str(ratings))
        app.recommender = TinyRecommender(app.data_loader)

    monkeypatch.setattr(app_module, "_initialize_recommender", initialize)
    app = app_module.create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "phase3-api-secret",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "WTF_CSRF_ENABLED": False,
            "CACHE_TYPE": "SimpleCache",
        }
    )
    with app.app_context():
        db.create_all()
    return app


def _login(app):
    with app.app_context():
        user = User(username="apiuser", email="api@example.com")
        user.set_password("correct-horse-battery-staple")
        db.session.add(user)
        db.session.commit()
        user_id = user.id

    client = app.test_client()
    response = client.post(
        "/login",
        data={"username": "apiuser", "password": "correct-horse-battery-staple"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    return client, user_id


def test_rating_and_watchlist_json_validation_and_persistence(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    client, user_id = _login(app)

    assert client.post("/api/rate", json={"movieId": 1}).status_code == 400
    rating_response = client.post("/api/rate", json={"movieId": 1, "rating": 4.5})
    assert rating_response.status_code == 200

    invalid_notes = client.post(
        "/api/watchlist/add", json={"movieId": 2, "notes": ["not", "text"]}
    )
    assert invalid_notes.status_code == 400
    watchlist_response = client.post(
        "/api/watchlist/add", json={"movieId": 2, "notes": "Friday"}
    )
    assert watchlist_response.status_code == 201
    duplicate_response = client.post("/api/watchlist/add", json={"movieId": 2})
    assert duplicate_response.status_code == 409

    with app.app_context():
        rating = db.session.execute(
            db.select(Rating).where(Rating.user_id == user_id, Rating.movie_id == 1)
        ).scalar_one()
        watchlist = db.session.execute(
            db.select(Watchlist).where(Watchlist.user_id == user_id, Watchlist.movie_id == 2)
        ).scalar_one()
        assert rating.rating == 4.5
        assert watchlist.notes == "Friday"
