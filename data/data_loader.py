"""
Data loader module for CSV files.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for CSV files."""
    
    def __init__(self, movies_path: str, ratings_path: Optional[str] = None, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the data loader.
        
        Args:
            movies_path: Path to the movies CSV file.
            ratings_path: Path to the ratings CSV file (optional).
            test_size: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.
        """
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.test_size = test_size
        self.random_state = random_state
        
        # Data containers
        self.movies_df = None
        self.ratings_df = None
        self.train_ratings = None
        self.test_ratings = None
        
        # Load data
        self._load_movies()
        if ratings_path:
            try:
                self._load_ratings()
            except FileNotFoundError:
                logger.warning(f"Ratings file not found: {ratings_path}. Using dummy ratings data.")
                self._create_dummy_ratings()
    
    def _load_movies(self) -> None:
        """Load movies data from CSV."""
        try:
            logger.info(f"Loading movies from {self.movies_path}")
            
            if not os.path.exists(self.movies_path):
                raise FileNotFoundError(f"Movies file not found: {self.movies_path}")
            
            self.movies_df = pd.read_csv(self.movies_path)
            
            # Check required columns
            required_cols = ['movieId', 'title', 'genres']
            missing_cols = [col for col in required_cols if col not in self.movies_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in movies CSV: {missing_cols}")
            
            # Process data
            self._process_movies()
            
            logger.info(f"Loaded {len(self.movies_df)} movies")
        
        except Exception as e:
            logger.error(f"Error loading movies: {str(e)}")
            raise
    
    def _create_dummy_ratings(self) -> None:
        """Create dummy ratings data when ratings file is missing."""
        logger.info("Creating dummy ratings data")
        
        # Take a sample of movies for dummy ratings
        sample_movies = self.movies_df.sample(min(1000, len(self.movies_df)), random_state=self.random_state)
        
        # Create dummy user IDs (1-10)
        user_ids = list(range(1, 11))
        
        # Create a list to hold the dummy ratings
        dummy_ratings = []
        
        # For each user, rate a subset of movies
        for user_id in user_ids:
            # Each user rates a random subset of movies
            user_movies = sample_movies.sample(min(50, len(sample_movies)), random_state=user_id)
            
            for _, movie in user_movies.iterrows():
                # Generate a random rating between 1 and 5
                rating = np.random.randint(1, 6)
                timestamp = 1600000000  # Dummy timestamp
                
                dummy_ratings.append({
                    'userId': user_id,
                    'movieId': movie['movieId'],
                    'rating': float(rating),
                    'timestamp': timestamp
                })
        
        # Create DataFrame from dummy ratings
        self.ratings_df = pd.DataFrame(dummy_ratings)
        
        # Process data
        self._process_ratings()
        
        # Split into train and test sets
        self._split_ratings()
        
        logger.info(f"Created {len(self.ratings_df)} dummy ratings from {len(user_ids)} dummy users")
    
    def _load_ratings(self) -> None:
        """Load ratings data from CSV."""
        try:
            logger.info(f"Loading ratings from {self.ratings_path}")
            
            if not os.path.exists(self.ratings_path):
                raise FileNotFoundError(f"Ratings file not found: {self.ratings_path}")
            
            self.ratings_df = pd.read_csv(self.ratings_path)
            
            # Check required columns
            required_cols = ['userId', 'movieId', 'rating']
            missing_cols = [col for col in required_cols if col not in self.ratings_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in ratings CSV: {missing_cols}")
            
            # Process data
            self._process_ratings()
            
            # Split into train and test sets
            self._split_ratings()
            
            logger.info(f"Loaded {len(self.ratings_df)} ratings from {self.ratings_df['userId'].nunique()} users")
        
        except FileNotFoundError:
            # Rethrow FileNotFoundError to be handled by the caller
            raise
        except Exception as e:
            logger.error(f"Error loading ratings: {str(e)}")
            raise
    
    def _process_movies(self) -> None:
        """Process movies data."""
        # Handle missing values
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        
        # Extract year from title if not already available
        if 'year' not in self.movies_df.columns:
            # Extract year from title (format: "Movie Title (YYYY)")
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$').astype('float')
            
        # Clean title (remove year)
        if 'clean_title' not in self.movies_df.columns:
            self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
            
        # Create a genres list
        self.movies_df['genres_list'] = self.movies_df['genres'].apply(lambda x: x.split('|') if x else [])
        
        # Add overview/plot column if missing
        if 'overview' not in self.movies_df.columns:
            self.movies_df['overview'] = ''
            
        # Add poster URL column if missing
        if 'poster_url' not in self.movies_df.columns:
            self.movies_df['poster_url'] = ''
    
    def _process_ratings(self) -> None:
        """Process ratings data."""
        # Handle missing values
        self.ratings_df = self.ratings_df.dropna(subset=['userId', 'movieId', 'rating'])
        
        # Ensure user and movie IDs are integers
        self.ratings_df['userId'] = self.ratings_df['userId'].astype(int)
        self.ratings_df['movieId'] = self.ratings_df['movieId'].astype(int)
        
        # Ensure ratings are floats
        self.ratings_df['rating'] = self.ratings_df['rating'].astype(float)
        
        # Filter ratings for movies that exist in the movies dataframe
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(self.movies_df['movieId'])]
    
    def _split_ratings(self) -> None:
        """Split ratings into train and test sets."""
        if self.ratings_df is not None:
            self.train_ratings, self.test_ratings = train_test_split(
                self.ratings_df,
                test_size=self.test_size,
                random_state=self.random_state
            )
            logger.info(f"Split ratings into {len(self.train_ratings)} train and {len(self.test_ratings)} test samples")
    
    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the train and test ratings.
        
        Returns:
            Tuple containing training and testing DataFrames.
        """
        if self.train_ratings is None or self.test_ratings is None:
            raise ValueError("Ratings data not loaded or split")
        return self.train_ratings, self.test_ratings
    
    def get_movies(self) -> pd.DataFrame:
        """
        Get the movies DataFrame.
        
        Returns:
            Movies DataFrame.
        """
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        return self.movies_df
    
    def get_ratings(self) -> pd.DataFrame:
        """
        Get the ratings DataFrame.
        
        Returns:
            Ratings DataFrame.
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded")
        return self.ratings_df
    
    def get_movie_by_id(self, movie_id: int) -> Optional[pd.Series]:
        """
        Get a movie by its ID.
        
        Args:
            movie_id: Movie ID.
            
        Returns:
            Movie as a pandas Series, or None if not found.
        """
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            return None
        return movie.iloc[0]
    
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for movies by title.
        
        Args:
            query: Search query.
            limit: Maximum number of results to return.
            
        Returns:
            DataFrame with matching movies.
        """
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        
        # Case-insensitive search on title
        matching_movies = self.movies_df[self.movies_df['title'].str.contains(query, case=False, na=False)]
        return matching_movies.head(limit)
    
    def get_movies_by_genre(self, genre: str, limit: int = 50) -> pd.DataFrame:
        """
        Get movies by genre.
        
        Args:
            genre: Genre to filter by.
            limit: Maximum number of results to return.
            
        Returns:
            DataFrame with movies matching the genre.
        """
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        
        # Case-insensitive search in the genres string
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, case=False, na=False)]
        return genre_movies.head(limit)
    
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """
        Get ratings for a specific user.
        
        Args:
            user_id: User ID.
            
        Returns:
            DataFrame of user ratings.
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded")
        
        # Filter by user ID
        return self.ratings_df[self.ratings_df['userId'] == user_id]
    
    def get_movie_ratings(self, movie_id: int) -> pd.DataFrame:
        """
        Get ratings for a specific movie.
        
        Args:
            movie_id: Movie ID.
            
        Returns:
            DataFrame of movie ratings.
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded")
        
        # Filter by movie ID
        return self.ratings_df[self.ratings_df['movieId'] == movie_id]
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> None:
        """
        Add a new rating or update an existing one.
        
        Args:
            user_id: User ID.
            movie_id: Movie ID.
            rating: Rating value.
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded")
        
        # Check if movie exists
        if not self.movies_df[self.movies_df['movieId'] == movie_id].any().any():
            raise ValueError(f"Movie {movie_id} does not exist")
        
        # Check if rating already exists
        existing = self.ratings_df[
            (self.ratings_df['userId'] == user_id) & 
            (self.ratings_df['movieId'] == movie_id)
        ]
        
        if len(existing) > 0:
            # Update existing rating
            self.ratings_df.loc[existing.index, 'rating'] = rating
        else:
            # Add new rating
            new_rating = pd.DataFrame({
                'userId': [user_id],
                'movieId': [movie_id],
                'rating': [rating],
                'timestamp': [pd.Timestamp.now().timestamp()]
            })
            self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
        
        # Re-split data
        self._split_ratings()
    
    def get_unique_genres(self) -> List[str]:
        """
        Get a list of unique movie genres from the dataset.
        
        Returns:
            Sorted list of unique genres.
        """
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        
        all_genres = set()
        # Iterate through the 'genres' column, split by '|', and add to the set
        for genres_str in self.movies_df['genres'].dropna():
            genres_list = genres_str.split('|')
            # Filter out empty strings or placeholder genres like '(no genres listed)'
            valid_genres = {genre for genre in genres_list if genre and genre != '(no genres listed)'}
            all_genres.update(valid_genres)
        
        # Return sorted list
        return sorted(list(all_genres))
    
    def get_popular_movies(self, n: int = 10) -> pd.DataFrame:
        """
        Get the most popular movies based on the number of ratings.
        
        Args:
            n: Number of popular movies to return.
            
        Returns:
            DataFrame containing the most popular movies.
        """
        if self.ratings_df is None:
            logger.warning("Ratings data not loaded, cannot determine popular movies by rating count.")
            # Fallback: return top N movies by arbitrary order if no ratings
            if self.movies_df is not None:
                return self.movies_df.head(n)
            else:
                return pd.DataFrame()
        
        # Count ratings per movie
        movie_rating_counts = self.ratings_df.groupby('movieId').size().reset_index(name='rating_count')
        
        # Sort by rating count descending
        popular_movie_ids = movie_rating_counts.sort_values('rating_count', ascending=False).head(n)['movieId']
        
        # Get movie details for these IDs
        popular_movies = self.movies_df[self.movies_df['movieId'].isin(popular_movie_ids)]
        
        # Add rating count to the DataFrame and sort by it
        popular_movies = popular_movies.merge(movie_rating_counts, on='movieId', how='left')
        popular_movies = popular_movies.sort_values('rating_count', ascending=False)
        
        return popular_movies
    
    def get_high_rated_movies(self, min_ratings: int = 10, n: int = 10) -> pd.DataFrame:
        """
        Get the highest-rated movies, considering a minimum number of ratings.
        
        Args:
            min_ratings: Minimum number of ratings required for a movie to be considered.
            n: Number of high-rated movies to return.
            
        Returns:
            DataFrame containing the highest-rated movies.
        """
        if self.ratings_df is None:
            logger.warning("Ratings data not loaded, cannot determine high-rated movies.")
             # Fallback: return top N movies by arbitrary order if no ratings
            if self.movies_df is not None:
                return self.movies_df.head(n)
            else:
                return pd.DataFrame()
        
        # Calculate average rating and rating count per movie
        movie_stats = self.ratings_df.groupby('movieId').agg(
            average_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
            rating_count=pd.NamedAgg(column='rating', aggfunc='count')
        ).reset_index()
        
        # Filter movies with enough ratings
        qualified_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
        
        # Sort by average rating descending
        high_rated_ids = qualified_movies.sort_values('average_rating', ascending=False).head(n)['movieId']
        
        # Get movie details for these IDs
        high_rated_movies = self.movies_df[self.movies_df['movieId'].isin(high_rated_ids)]
        
        # Add average rating and rating count to the DataFrame and sort
        high_rated_movies = high_rated_movies.merge(movie_stats, on='movieId', how='left')
        high_rated_movies = high_rated_movies.sort_values('average_rating', ascending=False)
        
        return high_rated_movies
    
    def describe_data(self) -> Dict:
        """
        Get descriptive statistics about the data.
        
        Returns:
            Dictionary with data statistics.
        """
        stats = {
            'movies': {
                'count': len(self.movies_df) if self.movies_df is not None else 0,
                'genres': self.get_unique_genres() if self.movies_df is not None else [],
                'years': {
                    'min': self.movies_df['year'].min() if self.movies_df is not None and 'year' in self.movies_df.columns else None,
                    'max': self.movies_df['year'].max() if self.movies_df is not None and 'year' in self.movies_df.columns else None
                }
            }
        }
        
        if self.ratings_df is not None:
            stats['ratings'] = {
                'count': len(self.ratings_df),
                'users': self.ratings_df['userId'].nunique(),
                'movies_rated': self.ratings_df['movieId'].nunique(),
                'rating_distribution': self.ratings_df['rating'].value_counts().to_dict(),
                'avg_rating': self.ratings_df['rating'].mean(),
                'median_rating': self.ratings_df['rating'].median()
            }
        
        return stats 