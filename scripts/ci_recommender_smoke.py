"""CI smoke check for the real Sentence Transformers content-model boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from models.content_based import ContentBasedRecommender


def main():
    catalog = pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Orbit (2020)",
                "genres": "Science Fiction|Drama",
                "clean_title": "Orbit",
                "overview": "Astronauts investigate an anomaly near Earth.",
            },
            {
                "movieId": 2,
                "title": "Deep Signal (2021)",
                "genres": "Science Fiction|Thriller",
                "clean_title": "Deep Signal",
                "overview": "A research crew receives a mysterious transmission.",
            },
            {
                "movieId": 3,
                "title": "Sunday Table (2019)",
                "genres": "Comedy|Drama",
                "clean_title": "Sunday Table",
                "overview": "A family reconnects over a chaotic weekend meal.",
            },
            {
                "movieId": 4,
                "title": "Night Run (2022)",
                "genres": "Action|Thriller",
                "clean_title": "Night Run",
                "overview": "A courier races across a city before sunrise.",
            },
        ]
    )

    with TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "embeddings.pkl"
        recommender = ContentBasedRecommender(
            transformer_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        recommender.fit(
            catalog,
            force_rebuild=True,
            cache_path=str(cache_path),
            batch_size=2,
        )
        recommendations = recommender.get_recommendations(1, top_n=2)

        if len(recommender.movie_embeddings) != len(catalog):
            raise RuntimeError("Transformer smoke test did not embed the complete test catalog")
        if not cache_path.exists():
            raise RuntimeError("Transformer smoke test did not write its embedding cache")
        if len(recommendations) != 2:
            raise RuntimeError("Transformer smoke test did not return the requested candidates")
        if any(item["movie"]["movieId"] == 1 for item in recommendations):
            raise RuntimeError("Transformer smoke test recommended the source movie itself")

    print("Sentence Transformers recommender smoke test passed")


if __name__ == "__main__":
    main()
