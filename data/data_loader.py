"""CSV-backed catalog and baseline-rating data loader.

``DataLoader`` owns catalog/baseline data for recommendation training.
Application-user ratings and watchlists are persisted through SQLAlchemy services and
must not be written here.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and expose movie catalog and optional baseline ratings CSV files."""

    def __init__(
        self,
        movies_path: str,
        ratings_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.test_size = test_size
        self.random_state = random_state

        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.train_ratings: Optional[pd.DataFrame] = None
        self.test_ratings: Optional[pd.DataFrame] = None

        self._load_movies()
        if ratings_path:
            try:
                self._load_ratings()
            except FileNotFoundError:
                logger.warning("Ratings file not found: %s; continuing with no baseline ratings", ratings_path)
                self._set_empty_ratings()
        else:
            self._set_empty_ratings()

    def _load_movies(self) -> None:
        if not os.path.exists(self.movies_path):
            raise FileNotFoundError(f"Movies file not found: {self.movies_path}")

        frame = pd.read_csv(self.movies_path)
        required = {"movieId", "title", "genres"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns in movies CSV: {missing}")

        self.movies_df = frame
        self._process_movies()
        logger.info("Loaded %s movies", len(self.movies_df))

    def _load_ratings(self) -> None:
        if not self.ratings_path or not os.path.exists(self.ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {self.ratings_path}")

        frame = pd.read_csv(self.ratings_path)
        required = {"userId", "movieId", "rating"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns in ratings CSV: {missing}")

        self.ratings_df = frame
        self._process_ratings()
        self._split_ratings()
        logger.info(
            "Loaded %s baseline ratings from %s users",
            len(self.ratings_df),
            self.ratings_df["userId"].nunique() if not self.ratings_df.empty else 0,
        )

    def _set_empty_ratings(self) -> None:
        self.ratings_df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        self.ratings_df = self.ratings_df.astype(
            {"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "float64"}
        )
        self.train_ratings = self.ratings_df.copy()
        self.test_ratings = self.ratings_df.copy()

    def _process_movies(self) -> None:
        frame = self.movies_df
        frame["movieId"] = frame["movieId"].astype(int)
        frame["genres"] = frame["genres"].fillna("")
        frame["title"] = frame["title"].fillna("").astype(str)

        if "year" not in frame.columns:
            frame["year"] = pd.to_numeric(
                frame["title"].str.extract(r"\((\d{4})\)$")[0],
                errors="coerce",
            )
        if "clean_title" not in frame.columns:
            frame["clean_title"] = frame["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
        frame["genres_list"] = frame["genres"].apply(lambda value: value.split("|") if value else [])
        if "overview" not in frame.columns:
            frame["overview"] = ""
        if "poster_url" not in frame.columns:
            frame["poster_url"] = ""

    def _process_ratings(self) -> None:
        frame = self.ratings_df.dropna(subset=["userId", "movieId", "rating"]).copy()
        frame["userId"] = frame["userId"].astype(int)
        frame["movieId"] = frame["movieId"].astype(int)
        frame["rating"] = frame["rating"].astype(float)
        if "timestamp" not in frame.columns:
            frame["timestamp"] = 0
        frame = frame[frame["movieId"].isin(self.movies_df["movieId"])]
        self.ratings_df = frame.reset_index(drop=True)

    def _split_ratings(self) -> None:
        if self.ratings_df is None or len(self.ratings_df) < 2:
            self.train_ratings = self.ratings_df.copy() if self.ratings_df is not None else None
            self.test_ratings = self.ratings_df.iloc[0:0].copy() if self.ratings_df is not None else None
            return
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings_df,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def get_movies(self) -> pd.DataFrame:
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        return self.movies_df

    def get_ratings(self) -> pd.DataFrame:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded")
        return self.ratings_df

    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_ratings is None or self.test_ratings is None:
            raise ValueError("Ratings data not loaded or split")
        return self.train_ratings, self.test_ratings

    def get_movie_by_id(self, movie_id: int) -> Optional[pd.Series]:
        movies = self.get_movies()
        match = movies[movies["movieId"] == int(movie_id)]
        return None if match.empty else match.iloc[0]

    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search titles; Phase 4 will replace current regex/truncation behavior."""
        movies = self.get_movies()
        matches = movies[movies["title"].str.contains(query, case=False, na=False)]
        return matches.head(max(1, int(limit)))

    def get_movies_by_genre(self, genre: str, limit: int = 50) -> pd.DataFrame:
        """Filter genres; exact matching/pagination semantics are a Phase 4 task."""
        movies = self.get_movies()
        matches = movies[movies["genres"].str.contains(genre, case=False, na=False)]
        return matches.head(max(1, int(limit)))

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        ratings = self.get_ratings()
        return ratings[ratings["userId"] == int(user_id)]

    def get_movie_ratings(self, movie_id: int) -> pd.DataFrame:
        ratings = self.get_ratings()
        return ratings[ratings["movieId"] == int(movie_id)]

    def add_rating(self, user_id: int, movie_id: int, rating: float) -> None:
        """Legacy baseline-data mutation helper for offline callers only."""
        if self.get_movie_by_id(movie_id) is None:
            raise ValueError(f"Movie {movie_id} does not exist")
        rating = float(rating)
        if not 0.5 <= rating <= 5.0:
            raise ValueError("Rating must be between 0.5 and 5.0")

        ratings = self.get_ratings()
        mask = (ratings["userId"] == int(user_id)) & (ratings["movieId"] == int(movie_id))
        timestamp = int(pd.Timestamp.now().timestamp())
        if mask.any():
            self.ratings_df.loc[mask, ["rating", "timestamp"]] = [rating, timestamp]
        else:
            self.ratings_df = pd.concat(
                [
                    ratings,
                    pd.DataFrame(
                        [{"userId": int(user_id), "movieId": int(movie_id), "rating": rating, "timestamp": timestamp}]
                    ),
                ],
                ignore_index=True,
            )
        self._split_ratings()

    def get_unique_genres(self) -> List[str]:
        all_genres = set()
        for genres in self.get_movies()["genres"].dropna():
            all_genres.update(
                genre
                for genre in str(genres).split("|")
                if genre and genre != "(no genres listed)"
            )
        return sorted(all_genres)

    def get_popular_movies(self, n: int = 10) -> pd.DataFrame:
        movies = self.get_movies()
        ratings = self.get_ratings()
        if ratings.empty:
            return movies.head(max(1, int(n))).copy()

        counts = ratings.groupby("movieId").size().reset_index(name="rating_count")
        result = movies.merge(counts, on="movieId", how="inner")
        return result.sort_values(["rating_count", "title"], ascending=[False, True]).head(max(1, int(n)))

    def get_high_rated_movies(self, min_ratings: int = 10, n: int = 10) -> pd.DataFrame:
        movies = self.get_movies()
        ratings = self.get_ratings()
        if ratings.empty:
            return movies.head(max(1, int(n))).copy()

        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(average_rating="mean", rating_count="count")
            .reset_index()
        )
        qualified = stats[stats["rating_count"] >= max(1, int(min_ratings))]
        if qualified.empty:
            qualified = stats
        result = movies.merge(qualified, on="movieId", how="inner")
        return result.sort_values(
            ["average_rating", "rating_count", "title"],
            ascending=[False, False, True],
        ).head(max(1, int(n)))
