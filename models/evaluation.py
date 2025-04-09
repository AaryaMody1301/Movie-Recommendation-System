"""
Evaluation module for recommendation models.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise.accuracy import rmse, mae
from .collaborative_filtering import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)


def evaluate_collaborative_model(model: CollaborativeRecommender, 
                                 testset: List[Tuple[str, str, float]]) -> Dict[str, float]:
    """
    Evaluate a collaborative filtering model.
    
    Args:
        model: Trained collaborative filtering model.
        testset: Test set in Surprise format.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    # Make predictions
    predictions = model.model.test(testset)
    
    # Calculate metrics
    rmse_score = rmse(predictions)
    mae_score = mae(predictions)
    
    return {
        'rmse': rmse_score,
        'mae': mae_score
    }


def evaluate_recommendation_precision_recall(recommender: Union[ContentBasedRecommender, CollaborativeRecommender, HybridRecommender],
                                            data_loader,
                                            k_values: List[int] = [5, 10, 20],
                                            user_sample: int = 100) -> Dict[str, Dict[int, float]]:
    """
    Evaluate precision and recall at k for recommendations.
    
    Args:
        recommender: Trained recommender model.
        data_loader: Data loader with train and test data.
        k_values: List of k values for precision/recall calculation.
        user_sample: Number of users to sample for evaluation.
        
    Returns:
        Dictionary with precision@k and recall@k metrics.
    """
    logger.info(f"Evaluating recommendation precision/recall@k for k={k_values}")
    
    # Get train and test data
    train, test = data_loader.get_train_test_split()
    
    # Sample users (if needed)
    user_ids = test['userId'].unique()
    if len(user_ids) > user_sample:
        np.random.seed(42)
        user_ids = np.random.choice(user_ids, user_sample, replace=False)
    
    # Initialize metrics
    precision = {k: [] for k in k_values}
    recall = {k: [] for k in k_values}
    
    # Evaluate for each user
    for user_id in user_ids:
        # Get ground truth movies from test set
        user_test = test[test['userId'] == user_id]
        ground_truth = set(user_test['movieId'])
        
        if len(ground_truth) == 0:
            continue
        
        # Get recommendations for user
        try:
            if isinstance(recommender, HybridRecommender):
                # Use the largest k
                recs = recommender.get_recommendations_for_user(user_id, n=max(k_values))
            elif isinstance(recommender, CollaborativeRecommender):
                # Use the largest k
                recs = recommender.get_recommendations(user_id, n=max(k_values))
            elif isinstance(recommender, ContentBasedRecommender):
                # For content-based, we need to find movies the user has rated in the training set
                user_train = train[train['userId'] == user_id]
                if len(user_train) == 0:
                    continue
                
                # Use the highest rated movie as seed
                seed_movie = user_train.sort_values('rating', ascending=False).iloc[0]['movieId']
                recs = recommender.get_recommendations(seed_movie, top_n=max(k_values))
            else:
                logger.warning(f"Unknown recommender type: {type(recommender)}")
                continue
                
            # Extract movie IDs from recommendations
            rec_ids = [rec['movie']['movieId'] for rec in recs]
            
            # Calculate precision and recall for each k
            for k in k_values:
                rec_at_k = set(rec_ids[:k])
                num_relevant = len(rec_at_k.intersection(ground_truth))
                
                # Precision@k = (# of recommended items @k that are relevant) / k
                precision[k].append(num_relevant / k if k > 0 else 0)
                
                # Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
                recall[k].append(num_relevant / len(ground_truth) if len(ground_truth) > 0 else 0)
                
        except Exception as e:
            logger.warning(f"Error evaluating user {user_id}: {str(e)}")
    
    # Calculate average precision and recall for each k
    avg_precision = {k: np.mean(vals) if vals else 0 for k, vals in precision.items()}
    avg_recall = {k: np.mean(vals) if vals else 0 for k, vals in recall.items()}
    
    # Calculate MAP (Mean Average Precision)
    map_score = np.mean([avg_precision[k] for k in k_values])
    
    metrics = {
        'precision@k': avg_precision,
        'recall@k': avg_recall,
        'map': map_score
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def evaluate_diversity(recommendations: List[Dict], movies_df: pd.DataFrame) -> float:
    """
    Calculate diversity of recommendations based on genre overlap.
    
    Args:
        recommendations: List of recommendation dictionaries.
        movies_df: DataFrame with movie metadata.
        
    Returns:
        Diversity score (0-1, higher is more diverse).
    """
    if not recommendations:
        return 0.0
    
    # Extract movie IDs
    movie_ids = [rec['movie']['movieId'] for rec in recommendations]
    
    # Get genres for each movie
    genres_list = []
    for movie_id in movie_ids:
        movie = movies_df[movies_df['movieId'] == movie_id]
        if len(movie) > 0:
            genres = movie.iloc[0]['genres'].split('|')
            genres_list.append(set(genres))
        else:
            genres_list.append(set())
    
    # Calculate pairwise Jaccard distance between genre sets
    n = len(genres_list)
    if n <= 1:
        return 0.0
    
    total_distance = 0.0
    pair_count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            # Jaccard distance = 1 - (|A ∩ B| / |A ∪ B|)
            intersection = len(genres_list[i].intersection(genres_list[j]))
            union = len(genres_list[i].union(genres_list[j]))
            
            if union > 0:
                distance = 1.0 - (intersection / union)
                total_distance += distance
                pair_count += 1
    
    # Average distance
    diversity = total_distance / pair_count if pair_count > 0 else 0.0
    return diversity


def evaluate_coverage(recommender: Union[ContentBasedRecommender, CollaborativeRecommender, HybridRecommender],
                      data_loader,
                      num_users: int = 100,
                      k: int = 10) -> float:
    """
    Calculate catalog coverage of recommendations.
    
    Args:
        recommender: Trained recommender model.
        data_loader: Data loader with movie catalog.
        num_users: Number of users to sample for evaluation.
        k: Number of recommendations per user.
        
    Returns:
        Coverage score (0-1, higher is better).
    """
    # Get movie catalog
    movies_df = data_loader.get_movies()
    all_movie_ids = set(movies_df['movieId'])
    
    # Get user IDs from ratings
    try:
        ratings_df = data_loader.get_ratings()
        user_ids = ratings_df['userId'].unique()
        
        # Sample users
        if len(user_ids) > num_users:
            np.random.seed(42)
            user_ids = np.random.choice(user_ids, num_users, replace=False)
        
        # Get recommendations for each user
        recommended_movies = set()
        
        for user_id in user_ids:
            try:
                if isinstance(recommender, HybridRecommender):
                    recs = recommender.get_recommendations_for_user(user_id, n=k)
                elif isinstance(recommender, CollaborativeRecommender):
                    recs = recommender.get_recommendations(user_id, n=k)
                else:
                    # For content-based, use a popular movie as seed
                    # Get user's ratings and use highest rated movie
                    user_ratings = ratings_df[ratings_df['userId'] == user_id]
                    if len(user_ratings) == 0:
                        continue
                        
                    seed_movie = user_ratings.sort_values('rating', ascending=False).iloc[0]['movieId']
                    recs = recommender.get_recommendations(seed_movie, top_n=k)
                
                # Add movie IDs to set of recommended movies
                for rec in recs:
                    recommended_movies.add(rec['movie']['movieId'])
                    
            except Exception as e:
                logger.warning(f"Error getting recommendations for user {user_id}: {str(e)}")
        
        # Calculate coverage
        coverage = len(recommended_movies) / len(all_movie_ids) if all_movie_ids else 0.0
        return coverage
        
    except Exception as e:
        logger.warning(f"Error calculating coverage: {str(e)}")
        return 0.0


def generate_evaluation_report(recommender: Union[ContentBasedRecommender, CollaborativeRecommender, HybridRecommender],
                              data_loader,
                              k_values: List[int] = [5, 10, 20],
                              user_sample: int = 100) -> Dict:
    """
    Generate a comprehensive evaluation report for a recommender.
    
    Args:
        recommender: Trained recommender model.
        data_loader: Data loader with data.
        k_values: List of k values for precision/recall calculation.
        user_sample: Number of users to sample for evaluation.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info(f"Generating evaluation report for {type(recommender).__name__}")
    
    report = {}
    
    # Evaluate precision/recall
    try:
        pr_metrics = evaluate_recommendation_precision_recall(
            recommender, data_loader, k_values, user_sample
        )
        report.update(pr_metrics)
    except Exception as e:
        logger.warning(f"Error evaluating precision/recall: {str(e)}")
    
    # Evaluate coverage
    try:
        coverage = evaluate_coverage(recommender, data_loader, user_sample, k=max(k_values))
        report['coverage'] = coverage
    except Exception as e:
        logger.warning(f"Error evaluating coverage: {str(e)}")
    
    # Evaluate RMSE/MAE for collaborative filtering
    if isinstance(recommender, CollaborativeRecommender) and hasattr(recommender, 'testset'):
        try:
            error_metrics = evaluate_collaborative_model(recommender, recommender.testset)
            report.update(error_metrics)
        except Exception as e:
            logger.warning(f"Error evaluating RMSE/MAE: {str(e)}")
    
    # Evaluate diversity for a sample user
    try:
        ratings_df = data_loader.get_ratings()
        user_ids = ratings_df['userId'].unique()
        
        if len(user_ids) > 0:
            # Sample a user
            user_id = user_ids[0]
            
            # Get recommendations
            if isinstance(recommender, HybridRecommender):
                recs = recommender.get_recommendations_for_user(user_id, n=10)
            elif isinstance(recommender, CollaborativeRecommender):
                recs = recommender.get_recommendations(user_id, n=10)
            else:
                # For content-based, use a popular movie as seed
                popular_movies = data_loader.get_popular_movies(1)
                if len(popular_movies) > 0:
                    seed_movie = popular_movies.iloc[0]['movieId']
                    recs = recommender.get_recommendations(seed_movie, top_n=10)
                else:
                    recs = []
            
            # Calculate diversity
            movies_df = data_loader.get_movies()
            diversity = evaluate_diversity(recs, movies_df)
            report['diversity'] = diversity
            
    except Exception as e:
        logger.warning(f"Error evaluating diversity: {str(e)}")
    
    logger.info(f"Evaluation report: {report}")
    return report 