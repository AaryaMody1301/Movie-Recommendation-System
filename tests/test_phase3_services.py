"""Phase 3 service-contract, persistence, and blueprint tests."""

import pandas as pd

import app as app_module
from data.data_loader import DataLoader
from database.db import db
from database.models import Rating, User, Watchlist
import services.movie_service as movie_service
import services.recommendation_service as recommendation_service
import services.user_service as user_service


class FakeRecommender:
    def __init__(self, loader):
        self.loader = loader

    def get_recommendations(self, movie_id, top_n=10):
        candidates = [movie for movie in self.loader.get_movies().to_dict("records") if movie["movieId"] != movie_id]
        return [
            {
                "movie": {"movieId": movie["movieId"], "title": movie["title"], "genres": movie["genres"]},
                "score": 0.9 - index * 0.1,
                "reason": f"Similar to {movie_id}",
            }
            for index, movie in enumerate(candidates[:top_n])
        ]


def _create_test_app(monkeypatch, tmp_path):
    movies_path = tmp_path / "movies.csv"
    ratings_path = tmp_path / "ratings.csv"
    pd.DataFrame(
        [
            {"movieId": 1, "title": "Alpha (2020)", "genres": "Action|Comedy"},
            {"movieId": 2, "title": "Beta (2021)", "genres": "Action|Drama"},
            {"movieId": 3, "title": "Gamma (2022)", "genres": "Comedy"},
            {"movieId": 4, "title": "Delta (2023)", "genres": "Drama"},
        ]
    ).to_csv(movies_path, index=False)
    pd.DataFrame(
        [
            {"userId": 100, "movieId": 1, "rating": 5.0, "timestamp": 1},
            {"userId": 100, "movieId": 2, "rating": 4.0, "timestamp": 2},
            {"userId": 101, "movieId": 2, "rating": 4.5, "timestamp": 3},
            {"userId": 101, "movieId": 3, "rating": 3.5, "timestamp": 4},
        ]
    ).to_csv(ratings_path, index=False)

    def initialize(app, embedding_args):
        app.data_loader = DataLoader(str(movies_path), str(ratings_path))
        app.recommender = FakeRecommender(app.data_loader)

    monkeypatch.setattr(app_module, "_initialize_recommender", initialize)
    app = app_module.create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "phase3-test-secret",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "WTF_CSRF_ENABLED": False,
            "CACHE_TYPE": "SimpleCache",
        }
    )
    return app


def _create_user(username="phase3user"):
    user = User(username=username, email=f"{username}@example.com", password_hash="placeholder")
    user.set_password("correct-horse-battery-staple")
    db.session.add(user)
    db.session.commit()
    return user


def test_phase3_blueprints_are_registered(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    assert "user.profile" in endpoints
    assert "user.ratings" in endpoints
    assert "user.watchlist" in endpoints
    assert "recommendations.user_recommendations" in endpoints
    assert "recommendations.api_rate_movie" in endpoints


def test_movie_service_uses_application_owned_loader(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    with app.app_context():
        assert movie_service.get_data_loader() is app.data_loader
        assert movie_service.get_data_loader().get_movies() is app.data_loader.movies_df


def test_persisted_rating_upsert_does_not_mutate_baseline_csv_state(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    with app.app_context():
        user = _create_user()
        baseline_count = len(app.data_loader.get_ratings())

        first = user_service.add_user_rating(user.id, 1, 4.0)
        second = user_service.add_user_rating(user.id, 1, 4.5)
        assert first["success"] and first["action"] == "added"
        assert second["success"] and second["action"] == "updated"

        rows = db.session.execute(
            db.select(Rating).where(Rating.user_id == user.id, Rating.movie_id == 1)
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].rating == 4.5
        assert len(app.data_loader.get_ratings()) == baseline_count

        profile = user_service.get_user_profile(user.id)
        assert profile["ratingCount"] == 1
        assert profile["averageRating"] == 4.5


def test_watchlist_is_persistent_and_unique(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    with app.app_context():
        user = _create_user("watcher")
        added = user_service.add_to_watchlist(user.id, 2, "Weekend movie")
        duplicate = user_service.add_to_watchlist(user.id, 2)
        assert added["success"] is True
        assert duplicate["success"] is False

        updated = user_service.update_watchlist_notes(user.id, 2, "Friday night")
        assert updated["success"] is True
        items, total = user_service.get_user_watchlist(user.id)
        assert total == 1
        assert items[0]["notes"] == "Friday night"
        assert db.session.execute(db.select(Watchlist)).scalars().one().movie_id == 2

        removed = user_service.remove_from_watchlist(user.id, 2)
        assert removed["success"] is True
        assert db.session.execute(db.select(Watchlist)).scalars().all() == []


def test_personalized_recommendations_read_persisted_user_ratings(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path)
    with app.app_context():
        user = _create_user("recommendee")
        assert user_service.add_user_rating(user.id, 1, 5.0)["success"]

        items, total = recommendation_service.get_recommendations_for_user(user.id, page=1, per_page=3)
        assert total > 0
        assert items
        assert all(item["movie"]["movieId"] != 1 for item in items)
        assert any("Alpha" in item["reason"] for item in items)
