"""Regression tests for Phase 5 collaborative and hybrid personalization."""

from __future__ import annotations

import pandas as pd
from flask import Flask

from database.db import db
from database.models import Rating, User
from models.collaborative_filtering import CollaborativeRecommender
from models.hybrid_recommender import HybridRecommender
import services.movie_service as movie_service
import services.recommendation_service as recommendation_service


def _movies_frame():
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "One (2020)", "genres": "Drama", "clean_title": "One", "overview": "One"},
            {"movieId": 2, "title": "Two (2020)", "genres": "Drama|Comedy", "clean_title": "Two", "overview": "Two"},
            {"movieId": 3, "title": "Three (2020)", "genres": "Comedy", "clean_title": "Three", "overview": "Three"},
            {"movieId": 4, "title": "Four (2020)", "genres": "Action", "clean_title": "Four", "overview": "Four"},
            {"movieId": 5, "title": "Five (2020)", "genres": "Drama", "clean_title": "Five", "overview": "Five"},
        ]
    )


def _ratings_frame():
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0},
            {"userId": 1, "movieId": 2, "rating": 4.0},
            {"userId": 2, "movieId": 1, "rating": 4.5},
            {"userId": 2, "movieId": 3, "rating": 5.0},
            {"userId": 2, "movieId": 4, "rating": 3.0},
            {"userId": 3, "movieId": 2, "rating": 4.5},
            {"userId": 3, "movieId": 3, "rating": 4.0},
            {"userId": 3, "movieId": 4, "rating": 4.5},
        ]
    )


def test_collaborative_raw_ids_and_trainset_survive_serialization(tmp_path):
    model = CollaborativeRecommender(n_factors=4, n_epochs=5, random_state=7)
    model.fit(_ratings_frame(), _movies_frame(), test_size=0.25)

    assert model.knows_user(1)
    assert model.knows_movie(3)
    prediction = model.predict_rating(1, 3)
    assert 0.5 <= prediction <= 5.0

    recommendations = model.get_recommendations(1, n=3, candidate_ids=[2, 3, 4, 5])
    assert recommendations
    assert all(rec["movie"]["movieId"] not in {1, 2} for rec in recommendations)
    # Movie 5 has no persisted collaborative rating and must not get an unknown-item default signal.
    assert all(rec["movie"]["movieId"] != 5 for rec in recommendations)

    path = tmp_path / "collaborative.pkl"
    model.save(str(path))
    loaded = CollaborativeRecommender.load(str(path))

    assert loaded.trainset is not None
    assert loaded.model.trainset is loaded.trainset
    assert loaded.knows_user(1)
    assert loaded.knows_movie(3)
    assert abs(loaded.predict_rating(1, 3) - prediction) < 1e-9


def test_hybrid_weighting_normalizes_different_score_scales():
    collab = CollaborativeRecommender(rating_scale=(0.5, 5.0))
    hybrid = HybridRecommender(
        collaborative_recommender=collab,
        content_weight=0.5,
        collab_weight=0.5,
    )
    content = [
        {"movie": {"movieId": 10, "title": "Both"}, "score": 0.8, "reason": "content"},
        {"movie": {"movieId": 11, "title": "Content only"}, "score": 0.95, "reason": "content"},
    ]
    collaborative = [
        {"movie": {"movieId": 10, "title": "Both"}, "score": 4.5, "reason": "collab"},
    ]

    ranked = hybrid.combine(content, collaborative, n=5, strategy="weighted")
    assert ranked[0]["movie"]["movieId"] == 10
    assert 0.0 <= ranked[0]["score"] <= 1.0
    assert ranked[0]["signals"]["content"] == 0.8
    assert 0.88 < ranked[0]["signals"]["collaborative"] < 0.90
    assert "content" in ranked[0]["reason"]
    assert "collab" in ranked[0]["reason"]

    rank_fused = hybrid.combine(content, collaborative, n=5, strategy="rank")
    assert rank_fused[0]["movie"]["movieId"] == 10


class _FakeLoader:
    def __init__(self, movies):
        self.movies = movies.copy()

    def get_movies(self):
        return self.movies

    def get_movie_by_id(self, movie_id):
        match = self.movies[self.movies["movieId"] == int(movie_id)]
        return None if match.empty else match.iloc[0]

    def get_ratings(self):
        raise AssertionError("Online Phase 5 personalization must not read baseline CSV ratings")

    def get_unique_genres(self):
        return sorted({genre for value in self.movies["genres"] for genre in value.split("|")})


class _FakeContentRecommender:
    def __init__(self, movies):
        self.movies = movies

    def get_recommendations(self, movie_id, top_n=10):
        candidates = []
        for _, row in self.movies.iterrows():
            candidate_id = int(row["movieId"])
            if candidate_id == int(movie_id):
                continue
            score = 0.95 if row["genres"].split("|")[0] == "Drama" else 0.65
            candidates.append(
                {
                    "movie": {
                        "movieId": candidate_id,
                        "title": row["title"],
                        "genres": row["genres"],
                    },
                    "score": score,
                    "reason": f"content seed {movie_id}",
                }
            )
        return candidates[:top_n]


def _phase5_app():
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SECRET_KEY="phase5",
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        CONTENT_WEIGHT=0.5,
        COLLAB_WEIGHT=0.5,
        COLLAB_MIN_RATINGS=5,
        COLLAB_MIN_USERS=2,
        COLLAB_MIN_ITEMS=2,
        COLLAB_CANDIDATE_POOL=20,
        N_FACTORS=4,
        COLLAB_N_EPOCHS=5,
        COLLAB_LR_ALL=0.005,
        COLLAB_REG_ALL=0.02,
        TEST_SIZE=0.2,
        RANDOM_STATE=7,
        PERSONALIZATION_MIN_RATING=3.5,
    )
    db.init_app(app)
    movies = _movies_frame()
    app.data_loader = _FakeLoader(movies)
    app.recommender = _FakeContentRecommender(movies)
    return app


def _seed_persisted_ratings():
    users = [
        User(username="u1", email="u1@example.com", password_hash="x"),
        User(username="u2", email="u2@example.com", password_hash="x"),
        User(username="cold", email="cold@example.com", password_hash="x"),
    ]
    db.session.add_all(users)
    db.session.flush()
    db.session.add_all(
        [
            Rating(user_id=users[0].id, movie_id=1, rating=5.0),
            Rating(user_id=users[0].id, movie_id=2, rating=4.0),
            Rating(user_id=users[1].id, movie_id=1, rating=4.5),
            Rating(user_id=users[1].id, movie_id=3, rating=5.0),
            Rating(user_id=users[1].id, movie_id=4, rating=4.0),
        ]
    )
    db.session.commit()
    return users


def test_online_personalization_uses_persisted_ratings_not_baseline_csv():
    app = _phase5_app()
    old_cache = movie_service._cache
    movie_service._cache = None
    try:
        with app.app_context():
            db.create_all()
            users = _seed_persisted_ratings()
            recommendations, total = recommendation_service.get_recommendations_for_user(
                users[0].id,
                page=1,
                per_page=10,
                strategy="weighted",
            )
            assert total == len(recommendations)
            assert recommendations
            assert all(rec["movie"]["movieId"] not in {1, 2} for rec in recommendations)
            assert app._collaborative_recommender is not None
            assert app._collaborative_recommender.knows_user(users[0].id)

            # Cold-start results should come from persisted application popularity before
            # falling back to deterministic catalog order, without touching baseline ratings.
            cold, _ = recommendation_service.get_recommendations_for_user(
                users[2].id,
                page=1,
                per_page=5,
            )
            assert cold
            assert any("registered users" in rec["reason"] for rec in cold)
    finally:
        movie_service._cache = old_cache
