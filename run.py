"""Development server script for Movie Recommendation System."""

import argparse
import logging
import sys

from app import create_app


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    for logger_name in [
        "",
        "models.content_based",
        "data.data_loader",
        "services.movie_service",
        "services.recommendation_service",
        "services.tmdb_service",
        "app",
        "__main__",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Force rebuild of the full-catalog recommendation embeddings",
    )
    pre_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Override the configured embedding batch size for this run",
    )

    embedding_args, remaining_argv = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    flask_parser = argparse.ArgumentParser(description="Run the movie recommendation app")
    flask_parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    flask_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    flask_parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    flask_args = flask_parser.parse_args(remaining_argv)

    embedding_args_dict = {
        "rebuild_embeddings": embedding_args.rebuild_embeddings,
        "batch_size": embedding_args.embedding_batch_size,
    }
    app = create_app(embedding_args=embedding_args_dict)

    print(f"\nMovie Recommendation System running at: http://{flask_args.host}:{flask_args.port}\n")
    app.run(
        host=flask_args.host,
        port=flask_args.port,
        debug=not flask_args.no_debug,
    )
