"""
Script to download a subset of MovieLens ratings data for the recommender system.
"""
import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_ratings(movies_df, output_file, num_users=100, ratings_per_user=20):
    """
    Create sample ratings data based on movies_df.
    
    Args:
        movies_df: DataFrame containing movie data
        output_file: Path to save the ratings file
        num_users: Number of users to simulate
        ratings_per_user: Number of ratings per user
    """
    try:
        # Create user IDs
        user_ids = list(range(1, num_users + 1))
        
        # Sample movies for each user
        all_ratings = []
        valid_movie_ids = movies_df['movieId'].values
        
        for user_id in user_ids:
            # Sample movies for this user
            sample_size = min(ratings_per_user, len(valid_movie_ids))
            movie_sample = random.sample(list(valid_movie_ids), sample_size)
            
            # Generate ratings (biased towards higher ratings)
            for movie_id in movie_sample:
                # Simulate rating behavior (people tend to rate movies they like)
                rating = round(max(2.5, random.gauss(3.8, 1.0)), 1)
                # Ensure rating is in correct range
                rating = min(max(rating, 0.5), 5.0)
                
                # Add timestamp (random time in last 2 years)
                timestamp = int(pd.Timestamp.now().timestamp() - random.randint(0, 63072000))
                
                all_ratings.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        # Create DataFrame and save
        ratings_df = pd.DataFrame(all_ratings)
        ratings_df.to_csv(output_file, index=False)
        logger.info(f"Created {len(ratings_df)} sample ratings from {num_users} users")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample ratings: {str(e)}")
        return False

def main():
    # Ensure data directory exists
    data_dir = Path('data')
    output_file = data_dir / 'ratings.csv'
    
    # Check if ratings file already exists
    if os.path.exists(output_file):
        logger.info(f"Ratings file already exists at {output_file}")
        return
    
    # Check if movies file exists
    movies_file = data_dir / 'movies.csv'
    if not os.path.exists(movies_file):
        logger.error(f"Movies file not found at {movies_file}. Cannot create ratings.")
        return
    
    # Load movies data
    try:
        logger.info(f"Loading movies from {movies_file}")
        movies_df = pd.read_csv(movies_file)
        logger.info(f"Loaded {len(movies_df)} movies")
        
        # Create ratings
        logger.info("Generating sample ratings data...")
        if create_sample_ratings(movies_df, output_file, num_users=200, ratings_per_user=50):
            logger.info(f"Successfully created ratings data at {output_file}")
        else:
            logger.error("Failed to create ratings data")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main() 