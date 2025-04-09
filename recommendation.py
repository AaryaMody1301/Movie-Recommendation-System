import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import gc  # Garbage collection
import os

class MovieRecommender:
    def __init__(self, movies_csv='movies.csv', sample_size=None, use_tmdb=True):
        """Initialize the movie recommender with movie data.
        
        Args:
            movies_csv (str): Path to the movies CSV file
            sample_size (int, optional): Number of movies to sample for development/testing
            use_tmdb (bool): Whether to use TMDb data to enhance recommendations
        """
        self.movies_df = None
        self.tfidf_matrix = None
        self.indices = None
        self.tmdb_keywords_cache = {}  # Cache for TMDb keywords
        self.use_tmdb = use_tmdb
        self.load_data(movies_csv, sample_size)
        self.preprocess_data()
        self.build_recommendation_matrix()
    
    def load_data(self, movies_csv, sample_size=None):
        """Load movie data from CSV file."""
        try:
            self.movies_df = pd.read_csv(movies_csv)
            
            # For testing, ensure certain popular movies are included
            if sample_size and sample_size < len(self.movies_df):
                # First, include some popular movies
                popular_movies = self.movies_df[self.movies_df['title'].str.contains('Toy Story|Star Wars|Matrix|Jurassic Park', case=False)]
                # Then sample the rest
                remaining = self.movies_df[~self.movies_df.index.isin(popular_movies.index)]
                remaining_sample = remaining.sample(min(sample_size - len(popular_movies), len(remaining)), random_state=42)
                # Combine the popular movies with the sample
                self.movies_df = pd.concat([popular_movies, remaining_sample])
            
            # Reset index to ensure continuity
            self.movies_df = self.movies_df.reset_index(drop=True)
            print(f"Loaded {len(self.movies_df)} movies from {movies_csv}")
        except Exception as e:
            print(f"Error loading {movies_csv}: {e}")
            raise
    
    def preprocess_data(self):
        """Preprocess movie data for content-based recommendation."""
        # Fill missing values
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        
        # Clean movie titles to remove year
        self.movies_df['clean_title'] = self.movies_df['title'].apply(lambda x: x.split('(')[0].strip())
        
        # Create indices for fast lookup
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['clean_title']).drop_duplicates()
    
    def get_tmdb_keywords_for_movie(self, movie_title):
        """Get keywords for a movie from TMDb API."""
        if not self.use_tmdb:
            return ""
        
        # Check if already cached
        if movie_title in self.tmdb_keywords_cache:
            return self.tmdb_keywords_cache[movie_title]
        
        try:
            from services.tmdb_service import find_tmdb_id_for_movie, get_movie_keywords, get_movie_details
            import re
            
            # Extract year from title if present
            year_match = re.search(r'\((\d{4})\)$', movie_title)
            year = None
            if year_match:
                year = int(year_match.group(1))
            
            # Clean title (remove year)
            clean_title = re.sub(r'\s*\(\d{4}\)$', '', movie_title)
            
            # Find TMDb ID
            tmdb_id = find_tmdb_id_for_movie(clean_title, year)
            
            if not tmdb_id:
                self.tmdb_keywords_cache[movie_title] = ""
                return ""
            
            # Get movie keywords
            keywords = get_movie_keywords(tmdb_id)
            
            # Get movie details for additional data
            details = get_movie_details(tmdb_id)
            
            # Initialize enhanced content string
            enhanced_content = ""
            
            # Add keywords
            if keywords:
                keyword_text = " ".join([keyword['name'] for keyword in keywords])
                enhanced_content += keyword_text + " "
            
            # Add director and main cast if available
            if details:
                # Add director
                if details.get('director'):
                    enhanced_content += details['director'].get('name', '') + " "
                
                # Add top cast members
                if details.get('cast'):
                    cast_text = " ".join([actor.get('name', '') for actor in details['cast'][:3]])
                    enhanced_content += cast_text + " "
                
                # Add production companies
                if details.get('production_companies'):
                    companies_text = " ".join([company.get('name', '') for company in details['production_companies'][:3]])
                    enhanced_content += companies_text + " "
            
            # Cache the result
            self.tmdb_keywords_cache[movie_title] = enhanced_content
            
            return enhanced_content
        except Exception as e:
            print(f"Error getting TMDb keywords for {movie_title}: {e}")
            self.tmdb_keywords_cache[movie_title] = ""
            return ""
    
    def build_recommendation_matrix(self):
        """Build the TF-IDF matrix for content-based recommendations."""
        # Create a TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Prepare the content for TF-IDF
        content_data = []
        for idx, row in self.movies_df.iterrows():
            # Start with genres
            content = row['genres'].replace('|', ' ')
            
            # Add TMDb enhanced content if available
            if self.use_tmdb:
                tmdb_content = self.get_tmdb_keywords_for_movie(row['title'])
                if tmdb_content:
                    content += " " + tmdb_content
            
            content_data.append(content)
        
        # Construct the TF-IDF matrix
        self.tfidf_matrix = tfidf.fit_transform(content_data)
        print("TF-IDF matrix built successfully")
    
    def get_movie_recommendations(self, title, top_n=10):
        """Get movie recommendations based on genre similarity."""
        # Get the index of the movie that matches the title
        try:
            idx = self.indices[title]
        except KeyError:
            # If the exact title is not found, find the closest match
            closest_titles = self.movies_df[self.movies_df['clean_title'].str.contains(title, case=False)]
            if len(closest_titles) > 0:
                idx = closest_titles.index[0]
                print(f"Exact title not found. Using closest match: {self.movies_df.iloc[idx]['clean_title']}")
            else:
                print(f"No movie found with title containing '{title}'")
                return pd.DataFrame()
        
        # Get the TF-IDF vector for the selected movie
        movie_vector = self.tfidf_matrix[idx:idx+1]
        
        # Calculate cosine similarity between this movie and all others
        # Use linear_kernel for efficiency instead of cosine_similarity
        cosine_sim = linear_kernel(movie_vector, self.tfidf_matrix).flatten()
        
        # Create a list of (index, similarity score) tuples
        sim_scores = list(enumerate(cosine_sim))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the top_n most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top N most similar movies with similarity scores
        recommendations = self.movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        # Force garbage collection to free memory
        gc.collect()
        
        return recommendations[['movieId', 'title', 'genres', 'similarity_score']]
    
    def get_all_movies(self, limit=1000):
        """Return all movies with optional limit."""
        return self.movies_df.head(limit)
    
    def get_movie_by_id(self, movie_id):
        """Return a movie by its ID."""
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) > 0:
            return movie.iloc[0]
        return None
    
    def get_top_movies_by_genre(self, genre, top_n=10):
        """Return top movies of a specific genre."""
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, case=False)]
        return genre_movies.head(top_n)
    
    def get_unique_genres(self):
        """Extract and return a list of unique genres from the dataset."""
        all_genres = []
        for genres in self.movies_df['genres'].str.split('|'):
            if isinstance(genres, list):
                all_genres.extend(genres)
        return sorted(list(set(all_genres)))
    
    def search_movies(self, query, limit=10):
        """Search for movies by title."""
        matching_movies = self.movies_df[self.movies_df['title'].str.contains(query, case=False)]
        return matching_movies.head(limit)

    def update_tmdb_info_for_movie(self, movie_id, tmdb_data):
        """Update the TMDb information for a movie."""
        if movie_id in self.movies_df['movieId'].values:
            idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
            
            # Extract keywords from TMDb data
            keywords_text = ""
            if 'keywords' in tmdb_data and tmdb_data['keywords']:
                keywords_text = " ".join([kw['name'] for kw in tmdb_data['keywords']])
            
            # Update the cache
            self.tmdb_keywords_cache[self.movies_df.loc[idx, 'title']] = keywords_text
            
            return True
        return False

if __name__ == "__main__":
    # For testing purposes
    # Use a sample for testing to avoid memory issues
    recommender = MovieRecommender(sample_size=5000)
    
    # Show available movies with "Toy" in the title
    print("\nAvailable movies with 'Toy' in the title:")
    toy_movies = recommender.search_movies("Toy")
    for _, row in toy_movies.iterrows():
        print(f"{row['title']} - {row['genres']}")
    
    # Get recommendations for a movie
    movie_title = "Toy Story"
    if len(toy_movies) > 0:
        movie_title = toy_movies.iloc[0]['clean_title']
    
    recommendations = recommender.get_movie_recommendations(movie_title)
    
    print(f"\nRecommendations for '{movie_title}':")
    for _, row in recommendations.iterrows():
        print(f"{row['title']} - {row['genres']} (Score: {row['similarity_score']:.4f})") 