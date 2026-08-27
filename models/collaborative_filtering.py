"""Collaborative filtering using Surprise SVD.

The model consumes application-user ratings using one stable raw-ID type (integers).
Surprise owns conversion between those raw IDs and its internal integer IDs through
the fitted Trainset.  The wrapper keeps that Trainset alongside the fitted algorithm
so recommendation/exclusion behavior survives serialization.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.accuracy import mae, rmse
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """Matrix-factorization recommender backed by Surprise SVD."""

    SERIALIZATION_VERSION = 2

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
        rating_scale: Tuple[float, float] = (0.5, 5.0),
    ):
        self.n_factors = max(1, int(n_factors))
        self.n_epochs = max(1, int(n_epochs))
        self.lr_all = float(lr_all)
        self.reg_all = float(reg_all)
        self.random_state = int(random_state)
        self.rating_scale = (float(rating_scale[0]), float(rating_scale[1]))

        self.model = self._new_model()
        self.movies_df: Optional[pd.DataFrame] = None
        self.trainset = None
        self.testset = None
        self.user_ids: set[int] = set()
        self.movie_ids: set[int] = set()
        self.rated_by_user: Dict[int, set[int]] = {}
        self.evaluation_metrics: Optional[Dict[str, float]] = None

    def _new_model(self):
        return SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )

    def _prepare_ratings(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        required = {"userId", "movieId", "rating"}
        missing = sorted(required.difference(ratings_df.columns))
        if missing:
            raise ValueError(f"Missing required columns in ratings DataFrame: {missing}")
        if movies_df is None or "movieId" not in movies_df.columns:
            raise ValueError("movies_df must contain movieId")

        frame = ratings_df[["userId", "movieId", "rating"]].dropna().copy()
        if frame.empty:
            raise ValueError("ratings_df contains no usable ratings")

        frame["userId"] = frame["userId"].astype(int)
        frame["movieId"] = frame["movieId"].astype(int)
        frame["rating"] = frame["rating"].astype(float)
        catalog_ids = set(movies_df["movieId"].astype(int))
        frame = frame[frame["movieId"].isin(catalog_ids)].copy()
        if frame.empty:
            raise ValueError("No ratings refer to movies in the catalog")

        low, high = self.rating_scale
        if ((frame["rating"] < low) | (frame["rating"] > high)).any():
            raise ValueError(f"Ratings must be between {low} and {high}")

        # Keep the most recent/last supplied value if a caller accidentally supplies
        # duplicate user/movie rows. SQLAlchemy persistence already enforces uniqueness.
        return frame.drop_duplicates(["userId", "movieId"], keep="last").reset_index(drop=True)

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> "CollaborativeRecommender":
        """Fit the production model on all supplied ratings.

        When enough data exists, a temporary holdout model is used to calculate RMSE
        and MAE. The production model is then fitted again on the complete Trainset so
        recommendations do not discard holdout interactions.
        """
        frame = self._prepare_ratings(ratings_df, movies_df)
        self.movies_df = movies_df.copy()
        self.movies_df["movieId"] = self.movies_df["movieId"].astype(int)
        self.user_ids = set(frame["userId"].astype(int))
        self.movie_ids = set(frame["movieId"].astype(int))
        self.rated_by_user = {
            int(user_id): set(group["movieId"].astype(int))
            for user_id, group in frame.groupby("userId")
        }

        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(frame[["userId", "movieId", "rating"]], reader)
        self.evaluation_metrics = None
        self.testset = None

        # Holdout evaluation is meaningful only when there is enough data to leave
        # observations on both sides of the split.
        if len(frame) >= 5 and 0 < float(test_size) < 1:
            try:
                eval_trainset, self.testset = train_test_split(
                    data,
                    test_size=float(test_size),
                    random_state=self.random_state,
                )
                eval_model = self._new_model()
                eval_model.fit(eval_trainset)
                predictions = eval_model.test(self.testset)
                self.evaluation_metrics = {
                    "rmse": float(rmse(predictions, verbose=False)),
                    "mae": float(mae(predictions, verbose=False)),
                }
            except Exception:
                logger.exception("Collaborative holdout evaluation failed; continuing with full-data fit")
                self.testset = None
                self.evaluation_metrics = None

        self.trainset = data.build_full_trainset()
        self.model = self._new_model()
        self.model.fit(self.trainset)
        logger.info(
            "Fitted collaborative model on %s ratings from %s users across %s movies",
            len(frame),
            len(self.user_ids),
            len(self.movie_ids),
        )
        return self

    def evaluate(self) -> Dict[str, float]:
        """Return holdout metrics calculated during the latest fit."""
        if self.evaluation_metrics is None:
            raise ValueError("No holdout evaluation is available for this fitted model")
        return dict(self.evaluation_metrics)

    def knows_user(self, user_id: int) -> bool:
        if self.trainset is None:
            return False
        try:
            self.trainset.to_inner_uid(int(user_id))
            return True
        except ValueError:
            return False

    def knows_movie(self, movie_id: int) -> bool:
        if self.trainset is None:
            return False
        try:
            self.trainset.to_inner_iid(int(movie_id))
            return True
        except ValueError:
            return False

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a rating using the same integer raw IDs used to fit Surprise."""
        if self.trainset is None or self.model is None:
            raise ValueError("Model not fitted yet")
        if not self.knows_user(user_id) or not self.knows_movie(movie_id):
            raise ValueError("User or movie is not part of the collaborative Trainset")
        return float(self.model.predict(int(user_id), int(movie_id)).est)

    def get_top_n_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Score a bounded candidate set and return highest predicted ratings.

        If ``candidate_ids`` is omitted, only items already represented in the
        collaborative Trainset are scored. This avoids treating Surprise's unknown-item
        default prediction as a genuine collaborative signal.
        """
        if self.trainset is None or self.movies_df is None:
            raise ValueError("Model not fitted yet")
        user_id = int(user_id)
        if not self.knows_user(user_id):
            return []

        rated = self.rated_by_user.get(user_id, set()) if exclude_rated else set()
        if candidate_ids is None:
            candidates = set(self.movie_ids)
        else:
            candidates = {int(movie_id) for movie_id in candidate_ids}
            candidates.intersection_update(self.movie_ids)
        candidates.difference_update(rated)

        scored = [
            (movie_id, self.predict_rating(user_id, movie_id))
            for movie_id in candidates
        ]
        scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        return scored[: max(1, int(n))]

    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict]:
        """Return collaborative recommendations in the canonical service shape."""
        recommendations = []
        for movie_id, predicted_rating in self.get_top_n_recommendations(
            user_id,
            n=n,
            exclude_rated=exclude_rated,
            candidate_ids=candidate_ids,
        ):
            match = self.movies_df[self.movies_df["movieId"] == int(movie_id)]
            if match.empty:
                continue
            row = match.iloc[0]
            movie = {
                "movieId": int(row["movieId"]),
                "title": row.get("title", ""),
                "genres": row.get("genres", ""),
            }
            for field in ["year", "overview", "poster_url"]:
                if field in row.index and pd.notna(row[field]) and row[field] != "":
                    movie[field] = row[field]
            recommendations.append(
                {
                    "movie": movie,
                    "score": float(predicted_rating),
                    "reason": (
                        "Collaborative signal: users' rating patterns predict "
                        f"about {predicted_rating:.1f}/5 for you."
                    ),
                }
            )
        return recommendations

    def save(self, path: str) -> None:
        """Atomically serialize the fitted wrapper state, including its Trainset."""
        if self.trainset is None or self.model is None:
            raise ValueError("Cannot save an unfitted collaborative model")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "serialization_version": self.SERIALIZATION_VERSION,
            "params": {
                "n_factors": self.n_factors,
                "n_epochs": self.n_epochs,
                "lr_all": self.lr_all,
                "reg_all": self.reg_all,
                "random_state": self.random_state,
                "rating_scale": self.rating_scale,
            },
            "model": self.model,
            "movies_df": self.movies_df,
            "trainset": self.trainset,
            "testset": self.testset,
            "user_ids": self.user_ids,
            "movie_ids": self.movie_ids,
            "rated_by_user": self.rated_by_user,
            "evaluation_metrics": self.evaluation_metrics,
        }
        temp_path = f"{path}.tmp"
        with open(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)

    @classmethod
    def load(cls, path: str) -> "CollaborativeRecommender":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("serialization_version") != cls.SERIALIZATION_VERSION:
            raise ValueError("Unsupported collaborative model serialization version")

        recommender = cls(**payload["params"])
        recommender.model = payload["model"]
        recommender.movies_df = payload["movies_df"]
        recommender.trainset = payload["trainset"]
        recommender.testset = payload.get("testset")
        recommender.user_ids = {int(value) for value in payload.get("user_ids", set())}
        recommender.movie_ids = {int(value) for value in payload.get("movie_ids", set())}
        recommender.rated_by_user = {
            int(user_id): {int(movie_id) for movie_id in movie_ids}
            for user_id, movie_ids in payload.get("rated_by_user", {}).items()
        }
        recommender.evaluation_metrics = payload.get("evaluation_metrics")

        # Surprise algorithms also retain a trainset internally after fit. Restore that
        # link explicitly so wrapper and algorithm state cannot drift after loading.
        recommender.model.trainset = recommender.trainset
        return recommender
