# Movie Recommendation System - Project Structure

```
movie-recommendation-system/
│
├── app/                             # Application package
│   ├── __init__.py                  # Flask app initialization
│   │
│   ├── models/                      # Database models
│   │   ├── __init__.py
│   │   └── user.py                  # User model definitions
│   │
│   ├── blueprints/                  # Flask route blueprints
│   │   ├── __init__.py
│   │   ├── auth.py                  # Authentication routes
│   │   ├── main.py                  # Main routes (home, about)
│   │   ├── movies.py                # Movie browsing routes
│   │   └── user.py                  # User profile routes
│   │
│   ├── services/                    # Business logic services
│   │   ├── __init__.py
│   │   ├── auth_service.py          # Authentication services
│   │   ├── movie_service.py         # Movie-related services
│   │   ├── tmdb_service.py          # TMDB API integration
│   │   └── user_service.py          # User profile services
│   │
│   ├── database/                    # Database configuration
│   │   ├── __init__.py
│   │   ├── db.py                    # Database connection
│   │   ├── models.py                # SQLAlchemy models
│   │   └── schema.sql               # SQL schema definition
│   │
│   ├── static/                      # Static assets
│   │   ├── css/                     # Stylesheets
│   │   │   └── styles.css           # Main CSS
│   │   ├── js/                      # JavaScript files
│   │   │   └── main.js              # Main JavaScript
│   │   └── img/                     # Images
│   │       └── placeholder.jpg      # Placeholder image
│   │
│   └── templates/                   # Jinja2 templates
│       ├── base.html                # Base template with layout
│       ├── index.html               # Homepage
│       ├── 404.html                 # Error page
│       ├── 500.html                 # Server error page
│       ├── search.html              # Search results
│       ├── movie.html               # Movie details
│       └── genre.html               # Genre listing
│
├── data/                            # Data files and processing
│   ├── movies.csv                   # Movie dataset
│   ├── ratings.csv                  # User ratings dataset
│   └── data_loader.py               # Data loading utilities
│
├── ml/                              # Machine learning components
│   ├── __init__.py
│   ├── content_based.py             # Content-based filtering
│   ├── collaborative.py             # Collaborative filtering (future)
│   └── utils.py                     # ML utilities
│
├── config.py                        # Configuration settings
├── run.py                           # Development server script
├── wsgi.py                          # WSGI entry point for production
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_app.py                  # Application tests
│   ├── test_recommendations.py      # Recommendation tests
│   └── conftest.py                  # Test fixtures
│
├── scripts/                         # Utility scripts
│   ├── analyze_data.py              # Data analysis script
│   └── model_training.py            # Training script
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
├── LICENSE                          # License file
├── .env.example                     # Example environment variables
└── .gitignore                       # Git ignore file
``` 