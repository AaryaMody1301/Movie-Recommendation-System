"""Content-based movie recommendations using sentence-transformer embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Generate full-catalog content embeddings and cosine-similarity recommendations."""

    CACHE_VERSION = 2
    DEFAULT_CACHE_PATH = os.path.join("instance", "embeddings_cache.pkl")
    REQUIRED_COLUMNS = ("movieId", "title", "genres", "clean_title", "overview")
    OPTIONAL_FEATURE_COLUMNS = ("director", "cast", "keywords")

    def __init__(self, transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.transformer_model = transformer_model
        self.model: Optional[SentenceTransformer] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.movie_embeddings: Optional[np.ndarray] = None
        self.id_to_index: Dict[int, int] = {}
        self.index_to_id: Dict[int, int] = {}
        self.cache_fingerprint: Optional[str] = None

    @staticmethod
    def _stable_value(value) -> str:
        """Serialize feature values deterministically for cache fingerprinting."""
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple)):
            try:
                return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
            except TypeError:
                return str(value)
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def _dataset_fingerprint(self, movies_df: pd.DataFrame) -> str:
        """Fingerprint every field that can affect generated content features."""
        columns = [
            column
            for column in self.REQUIRED_COLUMNS + self.OPTIONAL_FEATURE_COLUMNS
            if column in movies_df.columns
        ]
        normalized = movies_df.loc[:, columns].copy()
        for column in columns:
            normalized[column] = normalized[column].map(self._stable_value)

        digest = hashlib.sha256()
        digest.update(f"cache-version:{self.CACHE_VERSION}\n".encode())
        digest.update(f"model:{self.transformer_model}\n".encode())
        digest.update(f"columns:{','.join(columns)}\n".encode())
        digest.update(f"rows:{len(normalized)}\n".encode())
        hashed_rows = pd.util.hash_pandas_object(normalized, index=False).values
        digest.update(hashed_rows.tobytes())
        return digest.hexdigest()

    def _cache_payload_is_valid(self, payload: dict, fingerprint: str) -> bool:
        """Return whether a cached embedding payload exactly matches this fit request."""
        if not isinstance(payload, dict):
            return False
        if payload.get("cache_version") != self.CACHE_VERSION:
            return False
        if payload.get("transformer_model") != self.transformer_model:
            return False
        if payload.get("fingerprint") != fingerprint:
            return False

        movies_df = payload.get("movies_df")
        embeddings = payload.get("embeddings")
        id_to_index = payload.get("id_to_index")
        index_to_id = payload.get("index_to_id")
        if movies_df is None or embeddings is None or not isinstance(id_to_index, dict) or not isinstance(index_to_id, dict):
            return False
        if len(movies_df) != len(embeddings):
            return False
        if len(id_to_index) != len(movies_df) or len(index_to_id) != len(movies_df):
            return False
        if payload.get("movie_count") != len(movies_df):
            return False
        return True

    def _load_cache(self, cache_path: str, fingerprint: str) -> bool:
        if not cache_path or not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if not self._cache_payload_is_valid(payload, fingerprint):
                logger.info("Ignoring stale or incompatible embedding cache: %s", cache_path)
                return False

            self.movies_df = payload["movies_df"]
            self.movie_embeddings = payload["embeddings"]
            self.id_to_index = payload["id_to_index"]
            self.index_to_id = payload["index_to_id"]
            self.cache_fingerprint = fingerprint
            logger.info("Loaded %s full-catalog embeddings from %s", len(self.movies_df), cache_path)
            return True
        except Exception:
            logger.exception("Failed to read embedding cache %s; rebuilding", cache_path)
            return False

    def _save_cache(self, cache_path: str) -> None:
        if not cache_path:
            return
        payload = {
            "cache_version": self.CACHE_VERSION,
            "transformer_model": self.transformer_model,
            "fingerprint": self.cache_fingerprint,
            "movie_count": len(self.movies_df),
            "movies_df": self.movies_df,
            "embeddings": self.movie_embeddings,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
        }
        directory = os.path.dirname(cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp_path = f"{cache_path}.tmp"
        try:
            with open(temp_path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, cache_path)
            logger.info("Saved fingerprinted embeddings to %s", cache_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def fit(
        self,
        movies_df: pd.DataFrame,
        max_items: Optional[int] = None,
        force_rebuild: bool = False,
        cache_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Fit embeddings for every movie in the supplied catalog.

        ``max_items`` is retained for backwards compatibility but is intentionally
        ignored. Phase 4 guarantees that every local movie ID is represented in the
        content index; sampling here would reintroduce missing recommendations.
        """
        if movies_df is None or movies_df.empty:
            raise ValueError("movies_df is None or empty")

        missing = [column for column in self.REQUIRED_COLUMNS if column not in movies_df.columns]
        if missing:
            raise ValueError(f"Movies DataFrame missing required columns: {missing}")

        if max_items not in (None, 0) and int(max_items) < len(movies_df):
            logger.warning(
                "Ignoring max_items=%s because runtime recommendations require full catalog coverage",
                max_items,
            )

        batch_size = max(1, int(batch_size))
        cache_path = cache_path or self.DEFAULT_CACHE_PATH
        catalog = movies_df.copy().reset_index(drop=True)
        catalog["movieId"] = catalog["movieId"].astype(int)
        if catalog["movieId"].duplicated().any():
            duplicate_ids = catalog.loc[catalog["movieId"].duplicated(), "movieId"].tolist()[:5]
            raise ValueError(f"Duplicate movie IDs are not supported: {duplicate_ids}")

        fingerprint = self._dataset_fingerprint(catalog)
        if not force_rebuild and self._load_cache(cache_path, fingerprint):
            return self

        self.movies_df = catalog
        self.id_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movies_df["movieId"].tolist())
        }
        self.index_to_id = {index: movie_id for movie_id, index in self.id_to_index.items()}
        self.cache_fingerprint = fingerprint

        logger.info("Loading transformer model: %s", self.transformer_model)
        self.model = SentenceTransformer(self.transformer_model)
        content_features = self._create_content_features()
        self.movie_embeddings = self._generate_embeddings(content_features, batch_size=batch_size)
        if len(self.movie_embeddings) != len(self.movies_df):
            raise RuntimeError("Embedding generation returned the wrong number of rows")

        self._save_cache(cache_path)
        logger.info("Content recommender fitted for all %s movies", len(self.movies_df))
        return self

    def _create_content_features(self) -> List[str]:
        """Create deterministic natural-language features from local movie metadata."""
        if self.movies_df is None:
            raise ValueError("Model has no movie catalog")

        content_features: List[str] = []
        for _, row in self.movies_df.iterrows():
            parts = [
                f"Title: {self._stable_value(row.get('clean_title'))}.",
                f"Genres: {self._stable_value(row.get('genres')).replace('|', ', ')}.",
            ]
            overview = self._stable_value(row.get("overview"))
            if overview:
                parts.append(f"Overview: {overview}.")
            director = self._stable_value(row.get("director")) if "director" in self.movies_df.columns else ""
            if director:
                parts.append(f"Director: {director}.")
            cast = self._stable_value(row.get("cast")) if "cast" in self.movies_df.columns else ""
            if cast:
                parts.append(f"Cast: {cast}.")
            keywords = self._stable_value(row.get("keywords")) if "keywords" in self.movies_df.columns else ""
            if keywords:
                parts.append(f"Keywords: {keywords}.")
            content_features.append(" ".join(parts))
        return content_features

    def _generate_embeddings(self, content_features: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized embeddings in bounded batches."""
        if self.model is None:
            raise ValueError("Transformer model is not loaded")
        if not content_features:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        embeddings = []
        total = len(content_features)
        for start in range(0, total, batch_size):
            batch = content_features[start : start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device="cpu",
            )
            embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))
            if start == 0 or start + batch_size >= total or (start // batch_size + 1) % 20 == 0:
                logger.info("Embedded %s/%s movies", min(start + batch_size, total), total)
        return np.vstack(embeddings)

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Return the most similar local movie IDs for any indexed catalog movie."""
        if self.movie_embeddings is None or self.movies_df is None:
            raise ValueError("Model not fitted yet")
        movie_id = int(movie_id)
        if movie_id not in self.id_to_index:
            raise ValueError(f"Movie ID {movie_id} not found")

        movie_idx = self.id_to_index[movie_id]
        query_embedding = self.movie_embeddings[movie_idx].reshape(1, -1)
        scores = cosine_similarity(query_embedding, self.movie_embeddings).ravel()
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for index in ranked_indices:
            similar_id = self.index_to_id[int(index)]
            if similar_id == movie_id:
                continue
            results.append((similar_id, float(scores[index])))
            if len(results) >= max(1, int(top_n)):
                break
        return results

    def get_recommendations(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """Return normalized movie/score/reason recommendation dictionaries."""
        if self.movies_df is None:
            raise ValueError("Model not fitted yet")
        recommendations = []
        for similar_id, score in self.get_similar_movies(movie_id, top_n=top_n):
            movie = self.movies_df[self.movies_df["movieId"] == similar_id].iloc[0]
            movie_data = {
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
            }
            for field in ("year", "overview", "poster_url"):
                if field in movie.index:
                    value = movie[field]
                    try:
                        missing = pd.isna(value)
                    except (TypeError, ValueError):
                        missing = False
                    if not missing and value not in (None, ""):
                        movie_data[field] = value
            recommendations.append(
                {
                    "movie": movie_data,
                    "score": float(score),
                    "reason": self._generate_recommendation_reason(movie_id, similar_id, score),
                }
            )
        return recommendations

    def _generate_recommendation_reason(self, movie_id: int, similar_id: int, score: float) -> str:
        if self.movies_df is None:
            return "Recommended from content similarity."
        movie = self.movies_df[self.movies_df["movieId"] == int(movie_id)].iloc[0]
        similar_movie = self.movies_df[self.movies_df["movieId"] == int(similar_id)].iloc[0]
        common_genres = sorted(
            set(str(movie["genres"]).split("|")) & set(str(similar_movie["genres"]).split("|"))
        )
        if common_genres:
            return f"Similar content with shared genres: {', '.join(common_genres)}."
        return f"Similar to {movie['title']} based on content embeddings."

    def save(self, path: str) -> None:
        """Save a portable fitted recommender snapshot."""
        payload = {
            "cache_version": self.CACHE_VERSION,
            "transformer_model": self.transformer_model,
            "fingerprint": self.cache_fingerprint,
            "movie_count": len(self.movies_df) if self.movies_df is not None else 0,
            "movies_df": self.movies_df,
            "embeddings": self.movie_embeddings,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ContentBasedRecommender":
        """Load a portable fitted recommender snapshot without rebuilding embeddings."""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        recommender = cls(transformer_model=payload["transformer_model"])
        recommender.movies_df = payload["movies_df"]
        recommender.movie_embeddings = payload["embeddings"]
        recommender.id_to_index = payload["id_to_index"]
        recommender.index_to_id = payload["index_to_id"]
        recommender.cache_fingerprint = payload.get("fingerprint")
        return recommender
