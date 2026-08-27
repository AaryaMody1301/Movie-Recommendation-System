"""Hybrid candidate fusion for content and collaborative recommendation signals."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .collaborative_filtering import CollaborativeRecommender
from .content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Combine bounded recommendation candidate lists without dense all-item matrices."""

    SERIALIZATION_VERSION = 2
    VALID_STRATEGIES = {"weighted", "rank"}

    def __init__(
        self,
        content_recommender: Optional[ContentBasedRecommender] = None,
        collaborative_recommender: Optional[CollaborativeRecommender] = None,
        content_weight: float = 0.5,
        collab_weight: Optional[float] = None,
    ):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight, self.collab_weight = self._normalize_weights(
            content_weight,
            1.0 - float(content_weight) if collab_weight is None else collab_weight,
        )
        self.movies_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalize_weights(content_weight: float, collab_weight: float) -> Tuple[float, float]:
        content = max(0.0, float(content_weight))
        collab = max(0.0, float(collab_weight))
        total = content + collab
        if total <= 0:
            return 0.5, 0.5
        return content / total, collab / total

    @staticmethod
    def _movie_id(rec: Dict) -> Optional[int]:
        try:
            return int(rec["movie"]["movieId"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _content_score(value) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _collab_score(self, value) -> float:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return 0.0
        low, high = (0.5, 5.0)
        if self.collaborative_recommender is not None:
            low, high = self.collaborative_recommender.rating_scale
        if high <= low:
            return 0.0
        return min(1.0, max(0.0, (raw - low) / (high - low)))

    def combine(
        self,
        content_recs: Iterable[Dict],
        collab_recs: Iterable[Dict],
        n: int = 10,
        strategy: str = "weighted",
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict]:
        """Fuse already-bounded candidate lists into the canonical response shape."""
        strategy = strategy if strategy in self.VALID_STRATEGIES else "weighted"
        excluded = {int(value) for value in (exclude_ids or [])}
        content = [rec for rec in content_recs if self._movie_id(rec) not in excluded]
        collab = [rec for rec in collab_recs if self._movie_id(rec) not in excluded]

        content_rank = {self._movie_id(rec): rank for rank, rec in enumerate(content, start=1)}
        collab_rank = {self._movie_id(rec): rank for rank, rec in enumerate(collab, start=1)}
        content_by_id = {self._movie_id(rec): rec for rec in content if self._movie_id(rec) is not None}
        collab_by_id = {self._movie_id(rec): rec for rec in collab if self._movie_id(rec) is not None}
        all_ids = set(content_by_id) | set(collab_by_id)

        fused = []
        for movie_id in all_ids:
            content_rec = content_by_id.get(movie_id)
            collab_rec = collab_by_id.get(movie_id)
            if strategy == "rank":
                # Weighted reciprocal-rank fusion. The +60 constant keeps one list from
                # overwhelming the other because of a very small absolute rank.
                content_signal = (
                    1.0 / (60.0 + content_rank[movie_id]) if content_rec is not None else 0.0
                )
                collab_signal = (
                    1.0 / (60.0 + collab_rank[movie_id]) if collab_rec is not None else 0.0
                )
            else:
                content_signal = self._content_score(content_rec.get("score")) if content_rec else 0.0
                collab_signal = self._collab_score(collab_rec.get("score")) if collab_rec else 0.0

            content_contribution = self.content_weight * content_signal
            collab_contribution = self.collab_weight * collab_signal
            score = content_contribution + collab_contribution
            source = content_rec or collab_rec
            movie = dict(source["movie"])

            reasons = []
            if content_rec:
                reasons.append(content_rec.get("reason") or "Content similarity contributed to this result.")
            if collab_rec:
                reasons.append(collab_rec.get("reason") or "Collaborative rating patterns contributed to this result.")

            fused.append(
                {
                    "movie": movie,
                    "score": float(score),
                    "reason": " ".join(reasons) or "Recommended from available signals.",
                    "signals": {
                        "content": round(float(content_signal), 6) if content_rec else None,
                        "collaborative": round(float(collab_signal), 6) if collab_rec else None,
                        "content_contribution": round(float(content_contribution), 6),
                        "collaborative_contribution": round(float(collab_contribution), 6),
                    },
                }
            )

        fused.sort(
            key=lambda item: (float(item["score"]), item["movie"].get("title", "")),
            reverse=True,
        )
        return fused[: max(1, int(n))]

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: Optional[pd.DataFrame] = None,
        train_content: bool = True,
        train_collaborative: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> "HybridRecommender":
        """Offline compatibility helper for model-training scripts."""
        self.movies_df = movies_df.copy()
        if train_content and self.content_recommender is not None:
            self.content_recommender.fit(movies_df)
        if train_collaborative and self.collaborative_recommender is not None:
            if ratings_df is None or ratings_df.empty:
                raise ValueError("Ratings data required for collaborative filtering")
            self.collaborative_recommender.fit(ratings_df, movies_df, test_size=test_size)
        return self

    def get_recommendations_for_user(
        self,
        user_id: int,
        n: int = 10,
        strategy: str = "weighted",
    ) -> List[Dict]:
        """Offline/backward-compatible user recommendations from fitted components."""
        if self.movies_df is None:
            raise ValueError("Model not fitted yet")

        collab_recs = []
        rated_ids: set[int] = set()
        if self.collaborative_recommender is not None:
            rated_ids = self.collaborative_recommender.rated_by_user.get(int(user_id), set())
            collab_recs = self.collaborative_recommender.get_recommendations(
                int(user_id),
                n=max(20, n * 3),
            )

        content_recs: List[Dict] = []
        if self.content_recommender is not None and rated_ids:
            for movie_id in list(sorted(rated_ids))[:10]:
                try:
                    content_recs.extend(
                        self.content_recommender.get_recommendations(movie_id, top_n=max(10, n * 2))
                    )
                except ValueError:
                    continue

        return self.combine(
            content_recs,
            collab_recs,
            n=n,
            strategy=strategy,
            exclude_ids=rated_ids,
        )

    def get_recommendations_for_movie(
        self,
        movie_id: int,
        user_id: Optional[int] = None,
        n: int = 10,
        strategy: str = "weighted",
    ) -> List[Dict]:
        if self.movies_df is None:
            raise ValueError("Model not fitted yet")

        content_recs = []
        if self.content_recommender is not None:
            try:
                content_recs = self.content_recommender.get_recommendations(
                    int(movie_id),
                    top_n=max(10, n * 3),
                )
            except ValueError:
                content_recs = []

        collab_recs = []
        rated_ids: set[int] = {int(movie_id)}
        if user_id is not None and self.collaborative_recommender is not None:
            rated_ids |= self.collaborative_recommender.rated_by_user.get(int(user_id), set())
            candidate_ids = [self._movie_id(rec) for rec in content_recs]
            collab_recs = self.collaborative_recommender.get_recommendations(
                int(user_id),
                n=max(10, n * 3),
                candidate_ids=[value for value in candidate_ids if value is not None],
            )

        return self.combine(
            content_recs,
            collab_recs,
            n=n,
            strategy=strategy,
            exclude_ids=rated_ids,
        )

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "serialization_version": self.SERIALIZATION_VERSION,
            "content_weight": self.content_weight,
            "collab_weight": self.collab_weight,
            "movies_df": self.movies_df,
        }
        temp_path = f"{path}.tmp"
        with open(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)

    @classmethod
    def load(
        cls,
        path: str,
        content_recommender: Optional[ContentBasedRecommender] = None,
        collaborative_recommender: Optional[CollaborativeRecommender] = None,
    ) -> "HybridRecommender":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("serialization_version") != cls.SERIALIZATION_VERSION:
            raise ValueError("Unsupported hybrid model serialization version")
        recommender = cls(
            content_recommender=content_recommender,
            collaborative_recommender=collaborative_recommender,
            content_weight=payload["content_weight"],
            collab_weight=payload["collab_weight"],
        )
        recommender.movies_df = payload.get("movies_df")
        return recommender
