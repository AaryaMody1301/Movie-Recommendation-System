# Movie Recommendation System

A content-based movie recommendation system built with Python, Flask, Pandas, and Scikit-learn, enhanced with The Movie Database (TMDb) API integration.

## Features

- **Content-Based Filtering**: Recommends movies based on genre, keywords, cast, and crew
- **TMDb Integration**: Enriches movie data with details from The Movie Database API
- **Enhanced Movie Details**: View posters, backdrops, cast, director, trailers, and more
- **Watch Providers**: See where to stream, rent, or buy movies (region-specific)
- **Multiple Recommendation Methods**: Toggle between content-based and TMDb similar movies
- **Web Interface**: Modern and responsive UI for browsing movies and recommendations
- **Search Functionality**: Search for movies by title
- **Genre Filtering**: Browse movies by genre
- **Detailed Movie Pages**: View movie details and similar recommendations

## Requirements

- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Werkzeug
- Flask-WTF

## Dataset

The system primarily uses a `movies.csv` dataset with the following structure:

- `movieId`: Unique identifier for each movie
- `title`: Movie title (with year)
- `genres`: Pipe-separated list of genres for each movie

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure your `movies.csv` file is in the project root directory

4. Create a `.env` file based on `.env.example` and add your TMDb API key:
   ```
   TMDB_API_KEY=your_tmdb_api_key_here
   ```
   You can obtain a TMDb API key by creating an account at [https://www.themoviedb.org/](https://www.themoviedb.org/) and requesting an API key from your account settings.

## Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## How It Works

1. **Data Loading**: The system loads movie data from `movies.csv` when the application starts
2. **TMDb Enrichment**: Movie data is enriched with information from The Movie Database API
3. **Enhanced TF-IDF Vectorization**: Movie genres, keywords, cast, and crew are converted to TF-IDF vectors
4. **Similarity Calculation**: Cosine similarity is calculated between movie vectors
5. **Multiple Recommendation Sources**: The system provides both content-based recommendations and TMDb API similar movie suggestions
6. **Caching**: TMDb API responses are cached to improve performance and respect rate limits

## TMDb Integration Features

- **Movie Posters and Backdrops**: High-quality images for each movie
- **Detailed Information**: Overview, release date, runtime, rating, etc.
- **Cast and Crew**: Main actors and director for each movie
- **Keywords**: Associated keywords for better recommendation
- **Trailers**: YouTube trailers when available
- **Watch Providers**: Where to stream, rent, or buy (region-specific)
- **Similar Movies**: TMDb's algorithmic similar movie recommendations

## Project Structure

- `app.py`: Main Flask application
- `recommendation.py`: Core recommendation engine
- `services/tmdb_service.py`: TMDb API integration service
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript, images)
- `requirements.txt`: Required Python packages
- `.env.example`: Example environment variables file
- `README.md`: Project documentation

## Future Enhancements

- Add collaborative filtering for user-based recommendations
- Incorporate additional movie metadata (directors, actors, plot)
- Implement user accounts and personalized recommendations
- Add movie ratings and reviews 