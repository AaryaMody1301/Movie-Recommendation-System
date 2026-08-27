"""Pre-generate the full movie-catalog embedding cache."""

import argparse
import logging

from data.data_loader import DataLoader
from models.content_based import ContentBasedRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the full-catalog embedding cache for the recommendation system"
    )
    parser.add_argument("--movies-csv", default="data/movies.csv", help="Path to movies CSV file")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore a compatible cache and regenerate every embedding",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model to use",
    )
    parser.add_argument(
        "--output",
        default="instance/embeddings_cache.pkl",
        help="Fingerprint-validated embedding cache path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    args = parser.parse_args()

    logger.info("Loading movie catalog from %s", args.movies_csv)
    data_loader = DataLoader(movies_path=args.movies_csv)
    movies_df = data_loader.get_movies()
    logger.info("Loaded %s movies", len(movies_df))

    recommender = ContentBasedRecommender(transformer_model=args.model)
    recommender.fit(
        movies_df,
        force_rebuild=args.force_rebuild,
        cache_path=args.output,
        batch_size=max(1, args.batch_size),
    )
    logger.info(
        "Full-catalog embeddings ready: %s movies at %s",
        len(recommender.movies_df),
        args.output,
    )


if __name__ == "__main__":
    main()
