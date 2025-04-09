# Movie Recommendation System

A movie recommendation system built with Python, Flask, and machine learning techniques. The system utilizes content-based filtering and integrates with The Movie Database (TMDb) API for enhanced movie details.

## Features

- **Content-Based Filtering**: Recommends movies based on genre similarity
- **TMDb Integration**: Enriches movie data with details from The Movie Database API
- **Movie Details**: View posters, backdrops, cast, director, and more
- **Watch Providers**: See where to stream, rent, or buy movies (region-specific)
- **Web Interface**: Responsive UI for browsing movies and recommendations
- **Search Functionality**: Search for movies by title
- **Genre Filtering**: Browse movies by genre
- **Performance Optimized**: Caching and efficient data loading

## Requirements

- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Flask-SQLAlchemy
- Flask-Login
- Python-dotenv
- Requests

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example` and add your TMDb API key:
   ```
   TMDB_API_KEY=your_tmdb_api_key_here
   ```
   You can obtain a TMDb API key by creating an account at [https://www.themoviedb.org/](https://www.themoviedb.org/) and requesting an API key.

5. Place your dataset files in the data directory:
   - `data/movies.csv`: Movie information
   - `data/ratings.csv`: User ratings (optional)

## Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Project Structure

```
movie-recommendation-system/
│
├── app/                             # Application package
│   ├── __init__.py                  # App initialization
│   ├── models/                      # Database models
│   ├── blueprints/                  # Flask route blueprints
│   ├── services/                    # Business logic services
│   ├── database/                    # Database configuration
│   ├── static/                      # Static assets
│   └── templates/                   # Jinja2 templates
│
├── data/                            # Data files and processing
│   ├── movies.csv                   # Movie dataset
│   ├── ratings.csv                  # Ratings dataset
│   └── data_loader.py               # Data loading utilities
│
├── ml/                              # Machine learning components
│   ├── content_based.py             # Content-based filtering
│   ├── collaborative.py             # Collaborative filtering (future)
│   └── utils.py                     # ML utilities
│
├── config.py                        # Configuration settings
├── run.py                           # Development server script
├── wsgi.py                          # WSGI entry point for production
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
├── LICENSE                          # License file
└── .env.example                     # Example environment variables
```

## How It Works

1. **Data Loading**: The system loads movie data from CSV files
2. **TF-IDF Vectorization**: Movie genres are converted to TF-IDF vectors
3. **Similarity Calculation**: Cosine similarity finds movies with similar features
4. **TMDb Enhancement**: Data is enriched with TMDb API information
5. **Caching**: API responses are cached to improve performance

## Future Enhancements

- Add collaborative filtering for user-based recommendations
- Implement user accounts and personalized recommendations
- Add movie ratings and reviews functionality
- Improve recommendation algorithms with deep learning
- Add API endpoints for mobile apps

## License

MIT 