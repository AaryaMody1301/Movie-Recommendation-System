"""
Utility script to pre-generate movie embeddings.

This script loads the movie data and generates embeddings for the recommendation system
without starting the full application.
"""
import os
import argparse
import logging
from data.data_loader import DataLoader
from models.content_based import ContentBasedRecommender

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate movie embeddings for recommendation system')
    parser.add_argument('--movies-csv', default='data/movies.csv', help='Path to movies CSV file')
    parser.add_argument('--max-movies', type=int, default=2000, help='Maximum number of movies to process')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild of embeddings')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='Transformer model to use')
    parser.add_argument('--output', default='instance/embeddings_cache.pkl', help='Output file for embeddings')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding generation')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    logger.info("Loading movie data...")
    try:
        data_loader = DataLoader(movies_path=args.movies_csv)
        movies_df = data_loader.get_movies()
        logger.info(f"Loaded {len(movies_df)} movies from {args.movies_csv}")
    except Exception as e:
        logger.error(f"Error loading movies: {e}")
        return
    
    # Create and fit recommender
    logger.info(f"Initializing recommender with model: {args.model}")
    recommender = ContentBasedRecommender(transformer_model=args.model)
    
    # Override batch size for this script
    original_batch_size = recommender._generate_embeddings.__defaults__[0]
    recommender._generate_embeddings.__defaults__ = (args.batch_size,)
    
    # Fit the model
    logger.info(f"Generating embeddings for up to {args.max_movies} movies...")
    recommender.fit(movies_df, max_items=args.max_movies, force_rebuild=args.force_rebuild)
    
    # Save to specified output file
    try:
        import pickle
        cache_data = {
            'movies_df': recommender.movies_df,
            'embeddings': recommender.movie_embeddings,
            'id_to_index': recommender.id_to_index,
            'index_to_id': recommender.index_to_id
        }
        with open(args.output, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Embeddings saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 