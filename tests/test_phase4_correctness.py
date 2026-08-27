"""Regression tests for Phase 4 catalog and content-recommendation correctness."""

import pickle

import numpy as np
import pandas as pd
from flask import Flask

from data.data_loader import DataLoader
import models.content_based as content_module
from models.content_based import ContentBasedRecommender
import services.movie_service as movie_service


def _catalog_rows():
    rows = []
    for movie_id in range(1, 26):
        rows.append(
            {
                "movieId": movie_id,
                "title": f"[Special] Film {movie_id} (2020)",
                "genres": "Action|Drama",
                "overview": f"Overview {movie_id}",
            }
        )
    rows.extend(
        [
            {
                "movieId": 26,
                "title": "Live Action Example (2021)",
                "genres": "Live Action|Comedy",
                "overview": "Not an Action genre token.",
            },
            {
                "movieId": 27,
                "title": "Actionable Example (2022)",
                "genres": "Actionable|Drama",
                "overview": "Also not an Action genre token.",
            },
        ]
    )
    return rows


def _write_catalog(tmp_path, rows=None):
    path = tmp_path / "movies.csv"
    pd.DataFrame(rows or _catalog_rows()).to_csv(path, index=False)
    return path


def _app_with_loader(loader):
    app = Flask(__name__)
    app.data_loader = loader
    app.recommender = None
    return app


def test_literal_search_exact_genres_and_full_pagination(tmp_path):
    loader = DataLoader(str(_write_catalog(tmp_path)))

    # '[' and ']' are regex metacharacters. The query must be treated literally.
    matches = loader.search_movies("[Special]")
    assert len(matches) == 25

    # Genre matching is token-based, so "Live Action" and "Actionable" do not match Action.
    action_movies = loader.get_movies_by_genre("action")
    assert len(action_movies) == 25
    assert set(action_movies["movieId"]) == set(range(1, 26))

    app = _app_with_loader(loader)
    old_cache = movie_service._cache
    movie_service._cache = None
    try:
        with app.app_context():
            page_two, search_total = movie_service.search_movies(
                "[Special]", page=2, per_page=10
            )
            assert search_total == 25
            assert len(page_two) == 10
            assert page_two[0]["movieId"] == 11

            genre_page, genre_total = movie_service.get_movies_by_genre(
                "Action", page=3, per_page=10
            )
            assert genre_total == 25
            assert len(genre_page) == 5
            assert genre_page[0]["movieId"] == 21

            # Sorting by baseline rating must not drop unrated catalog movies.
            _, browse_total = movie_service.get_all_movies(
                page=1, per_page=10, sort_by="rating", sort_order="desc"
            )
            assert browse_total == 27
    finally:
        movie_service._cache = old_cache


def test_content_fallback_has_stable_nested_shape(tmp_path):
    loader = DataLoader(str(_write_catalog(tmp_path)))
    app = _app_with_loader(loader)
    old_cache = movie_service._cache
    movie_service._cache = None
    try:
        with app.app_context():
            recommendations = movie_service.get_content_recommendations(1, top_n=3)
    finally:
        movie_service._cache = old_cache

    assert len(recommendations) == 3
    assert all(set(item) >= {"movie", "score", "reason"} for item in recommendations)
    assert all(isinstance(item["movie"], dict) for item in recommendations)
    assert all(item["movie"]["movieId"] != 1 for item in recommendations)


class _FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(
        self,
        batch,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    ):
        vectors = []
        for text in batch:
            checksum = sum(ord(char) for char in text)
            vector = np.array(
                [
                    float(checksum % 97 + 1),
                    float(checksum % 89 + 2),
                    float(len(text) % 83 + 3),
                    float((checksum + len(text)) % 79 + 4),
                ],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


class _FailIfConstructed:
    def __init__(self, model_name):
        raise AssertionError("A compatible cache should load without constructing the transformer")


def _small_embedding_catalog():
    return pd.DataFrame(
        [
            {
                "movieId": movie_id,
                "title": f"Movie {movie_id} (2020)",
                "genres": "Drama" if movie_id % 2 else "Comedy",
                "clean_title": f"Movie {movie_id}",
                "overview": f"Plot for movie {movie_id}",
            }
            for movie_id in range(1, 7)
        ]
    )


def test_full_catalog_embeddings_and_cache_fingerprint(monkeypatch, tmp_path):
    cache_path = tmp_path / "embeddings.pkl"
    catalog = _small_embedding_catalog()
    monkeypatch.setattr(content_module, "SentenceTransformer", _FakeSentenceTransformer)

    recommender = ContentBasedRecommender("fake-model")
    recommender.fit(
        catalog,
        max_items=2,  # Legacy cap must no longer truncate runtime coverage.
        force_rebuild=True,
        cache_path=str(cache_path),
        batch_size=2,
    )

    assert len(recommender.movies_df) == 6
    assert set(recommender.id_to_index) == set(range(1, 7))
    assert recommender.get_recommendations(6, top_n=2)

    first_fingerprint = recommender.cache_fingerprint
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    assert payload["cache_version"] == ContentBasedRecommender.CACHE_VERSION
    assert payload["movie_count"] == 6
    assert payload["fingerprint"] == first_fingerprint

    # An unchanged dataset/model should load the validated cache without loading a transformer.
    monkeypatch.setattr(content_module, "SentenceTransformer", _FailIfConstructed)
    cached = ContentBasedRecommender("fake-model")
    cached.fit(catalog, cache_path=str(cache_path))
    assert cached.cache_fingerprint == first_fingerprint
    assert cached.model is None
    assert set(cached.id_to_index) == set(range(1, 7))

    # Changing feature-bearing catalog data invalidates the cache and forces a rebuild.
    changed = catalog.copy()
    changed.loc[0, "overview"] = "A materially changed plot"
    monkeypatch.setattr(content_module, "SentenceTransformer", _FakeSentenceTransformer)
    rebuilt = ContentBasedRecommender("fake-model")
    rebuilt.fit(changed, cache_path=str(cache_path), batch_size=3)
    assert rebuilt.cache_fingerprint != first_fingerprint
    assert len(rebuilt.movie_embeddings) == len(changed)
