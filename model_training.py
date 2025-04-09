"""
Script for training and saving recommendation models.

This script trains and saves the following models:
1. Content-based recommender using Sentence Transformers
2. Collaborative filtering recommender using Matrix Factorization
3. Hybrid recommender combining both approaches

Usage:
    python model_training.py [--movies_path MOVIES_PATH] [--ratings_path RATINGS_PATH]
"""
import os
import sys
import logging
import argparse
import pickle
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from data.data_loader import DataLoader
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeRecommender
from models.hybrid_recommender import HybridRecommender
from models.evaluation import generate_evaluation_report
from config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train recommendation models')
    
    config = get_config()
    
    parser.add_argument('--movies_path', type=str, default=config.MOVIES_CSV,
                      help=f'Path to movies CSV file (default: {config.MOVIES_CSV})')
    parser.add_argument('--ratings_path', type=str, default=config.RATINGS_CSV,
                      help=f'Path to ratings CSV file (default: {config.RATINGS_CSV})')
    parser.add_argument('--content_model_path', type=str, default=config.CONTENT_MODEL_PATH,
                      help=f'Path to save content-based model (default: {config.CONTENT_MODEL_PATH})')
    parser.add_argument('--collab_model_path', type=str, default=config.COLLAB_MODEL_PATH,
                      help=f'Path to save collaborative model (default: {config.COLLAB_MODEL_PATH})')
    parser.add_argument('--n_factors', type=int, default=config.N_FACTORS,
                      help=f'Number of factors for matrix factorization (default: {config.N_FACTORS})')
    parser.add_argument('--test_size', type=float, default=config.TEST_SIZE,
                      help=f'Test size for evaluation (default: {config.TEST_SIZE})')
    parser.add_argument('--random_state', type=int, default=config.RANDOM_STATE,
                      help=f'Random state for reproducibility (default: {config.RANDOM_STATE})')
    parser.add_argument('--transformer_model', type=str, default=config.TRANSFORMER_MODEL,
                      help=f'Sentence transformer model (default: {config.TRANSFORMER_MODEL})')
    parser.add_argument('--content_weight', type=float, default=config.CONTENT_WEIGHT,
                      help=f'Weight for content-based recommendations (default: {config.CONTENT_WEIGHT})')
    parser.add_argument('--skip_content', action='store_true',
                      help='Skip training content-based model')
    parser.add_argument('--skip_collab', action='store_true',
                      help='Skip training collaborative model')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate models after training')
    parser.add_argument('--sample_size', type=int, default=None,
                      help='Sample size for training (default: use all data)')
    
    return parser.parse_args()


def train_content_based_model(data_loader: DataLoader, 
                             args) -> Optional[ContentBasedRecommender]:
    """
    Train and save content-based recommendation model.
    
    Args:
        data_loader: Data loader with movies data.
        args: Command line arguments.
        
    Returns:
        Trained content-based recommender, or None if training is skipped.
    """
    if args.skip_content:
        logger.info("Skipping content-based model training")
        return None
        
    logger.info("Training content-based recommendation model")
    
    # Get movies data
    movies_df = data_loader.get_movies()
    
    # Initialize and train content-based recommender
    try:
        start_time = datetime.now()
        
        recommender = ContentBasedRecommender(args.transformer_model)
        recommender.fit(movies_df)
        
        # Log training time
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Content-based model training completed in {training_time:.2f} seconds")
        
        # Save model
        recommender.save(args.content_model_path)
        logger.info(f"Content-based model saved to {args.content_model_path}")
        
        return recommender
        
    except Exception as e:
        logger.error(f"Error training content-based model: {str(e)}")
        return None


def train_collaborative_model(data_loader: DataLoader, 
                             args) -> Optional[CollaborativeRecommender]:
    """
    Train and save collaborative filtering recommendation model.
    
    Args:
        data_loader: Data loader with movies and ratings data.
        args: Command line arguments.
        
    Returns:
        Trained collaborative recommender, or None if training is skipped.
    """
    if args.skip_collab:
        logger.info("Skipping collaborative model training")
        return None
        
    logger.info("Training collaborative filtering recommendation model")
    
    try:
        # Get movies and ratings data
        movies_df = data_loader.get_movies()
        ratings_df = data_loader.get_ratings()
        
        if ratings_df is None or len(ratings_df) == 0:
            logger.error("No ratings data available for collaborative filtering")
            return None
            
        # Initialize and train collaborative recommender
        start_time = datetime.now()
        
        recommender = CollaborativeRecommender(
            n_factors=args.n_factors,
            random_state=args.random_state
        )
        
        recommender.fit(
            ratings_df=ratings_df,
            movies_df=movies_df,
            test_size=args.test_size
        )
        
        # Log training time
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Collaborative model training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        metrics = recommender.evaluate()
        logger.info(f"Collaborative model evaluation: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        
        # Save model
        recommender.save(args.collab_model_path)
        logger.info(f"Collaborative model saved to {args.collab_model_path}")
        
        return recommender
        
    except Exception as e:
        logger.error(f"Error training collaborative model: {str(e)}")
        return None


def train_hybrid_model(content_recommender: Optional[ContentBasedRecommender],
                      collab_recommender: Optional[CollaborativeRecommender],
                      data_loader: DataLoader,
                      args) -> Optional[HybridRecommender]:
    """
    Create and save hybrid recommendation model.
    
    Args:
        content_recommender: Trained content-based recommender.
        collab_recommender: Trained collaborative recommender.
        data_loader: Data loader with movies data.
        args: Command line arguments.
        
    Returns:
        Hybrid recommender, or None if both component models are missing.
    """
    if content_recommender is None and collab_recommender is None:
        logger.warning("Cannot create hybrid model: both component models are missing")
        return None
        
    logger.info("Creating hybrid recommendation model")
    
    try:
        # Get movies data
        movies_df = data_loader.get_movies()
        
        # Initialize hybrid recommender
        recommender = HybridRecommender(
            content_recommender=content_recommender,
            collaborative_recommender=collab_recommender,
            content_weight=args.content_weight
        )
        
        # Set the movies DataFrame
        recommender.movies_df = movies_df
        
        # Save hybrid model configuration
        hybrid_model_path = os.path.join(os.path.dirname(args.content_model_path), 'hybrid_model.pkl')
        recommender.save(hybrid_model_path)
        logger.info(f"Hybrid model saved to {hybrid_model_path}")
        
        return recommender
        
    except Exception as e:
        logger.error(f"Error creating hybrid model: {str(e)}")
        return None


def evaluate_models(content_recommender: Optional[ContentBasedRecommender],
                   collab_recommender: Optional[CollaborativeRecommender],
                   hybrid_recommender: Optional[HybridRecommender],
                   data_loader: DataLoader):
    """
    Evaluate trained recommendation models.
    
    Args:
        content_recommender: Trained content-based recommender.
        collab_recommender: Trained collaborative recommender.
        hybrid_recommender: Trained hybrid recommender.
        data_loader: Data loader with movies and ratings data.
    """
    logger.info("Evaluating recommendation models")
    
    # Evaluate content-based model
    if content_recommender is not None:
        try:
            logger.info("Evaluating content-based model")
            content_metrics = generate_evaluation_report(content_recommender, data_loader)
            logger.info(f"Content-based model evaluation: {content_metrics}")
        except Exception as e:
            logger.error(f"Error evaluating content-based model: {str(e)}")
    
    # Evaluate collaborative model
    if collab_recommender is not None:
        try:
            logger.info("Evaluating collaborative model")
            collab_metrics = generate_evaluation_report(collab_recommender, data_loader)
            logger.info(f"Collaborative model evaluation: {collab_metrics}")
        except Exception as e:
            logger.error(f"Error evaluating collaborative model: {str(e)}")
    
    # Evaluate hybrid model
    if hybrid_recommender is not None:
        try:
            logger.info("Evaluating hybrid model")
            hybrid_metrics = generate_evaluation_report(hybrid_recommender, data_loader)
            logger.info(f"Hybrid model evaluation: {hybrid_metrics}")
        except Exception as e:
            logger.error(f"Error evaluating hybrid model: {str(e)}")


def main():
    """Main function to train and evaluate recommendation models."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Check if data files exist
        if not os.path.exists(args.movies_path):
            logger.error(f"Movies file not found: {args.movies_path}")
            return
            
        if not args.skip_collab and not os.path.exists(args.ratings_path):
            logger.error(f"Ratings file not found: {args.ratings_path}")
            return
        
        # Create directories for model files if they don't exist
        os.makedirs(os.path.dirname(args.content_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.collab_model_path), exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {args.movies_path} and {args.ratings_path}")
        data_loader = DataLoader(
            movies_path=args.movies_path,
            ratings_path=args.ratings_path if not args.skip_collab else None,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Sample data if specified
        if args.sample_size is not None:
            logger.info(f"Using sample of {args.sample_size} movies")
            movies_df = data_loader.get_movies().sample(min(args.sample_size, len(data_loader.get_movies())), 
                                                      random_state=args.random_state)
            data_loader.movies_df = movies_df
            
            if not args.skip_collab:
                ratings_df = data_loader.get_ratings()
                # Filter ratings to only include sampled movies
                ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])]
                data_loader.ratings_df = ratings_df
                # Re-split ratings
                data_loader._split_ratings()
        
        # Train content-based model
        content_recommender = train_content_based_model(data_loader, args)
        
        # Train collaborative model
        collab_recommender = train_collaborative_model(data_loader, args)
        
        # Create hybrid model
        hybrid_recommender = train_hybrid_model(content_recommender, collab_recommender, data_loader, args)
        
        # Evaluate models
        if args.evaluate:
            evaluate_models(content_recommender, collab_recommender, hybrid_recommender, data_loader)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 