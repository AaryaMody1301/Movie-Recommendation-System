# Enhanced Movie Recommendation System - Project Structure

```
movie_recommendation_system/
│
├── app.py                           # Main Flask application entry point
├── config.py                        # Configuration settings
├── models/                          # Database models and ML models
│   ├── __init__.py
│   ├── collaborative_filtering.py   # Matrix Factorization with Surprise
│   ├── content_based.py             # Enhanced content-based filtering with sentence-transformers
│   ├── hybrid_recommender.py        # Combines both recommendation approaches
│   └── evaluation.py                # Model evaluation metrics and functions
│
├── blueprints/                      # Flask blueprints for route organization
│   ├── __init__.py
│   ├── auth.py                      # Authentication routes (login, register)
│   ├── main.py                      # Main routes (home, about)
│   ├── movies.py                    # Movie browsing, search, and filtering
│   ├── recommendations.py           # Recommendation routes
│   └── user.py                      # User profile and watchlist management
│
├── data/                            # Data files and processing
│   ├── movies.csv                   # Movie metadata
│   ├── ratings.csv                  # User ratings
│   └── data_loader.py               # Data loading and preprocessing utilities
│
├── services/                        # Business logic
│   ├── __init__.py
│   ├── auth_service.py              # Authentication services
│   ├── movie_service.py             # Movie-related services
│   ├── recommendation_service.py    # Recommendation logic
│   └── user_service.py              # User profile and watchlist services
│
├── database/                        # Database utilities
│   ├── __init__.py
│   ├── db.py                        # Database connection and session management
│   ├── models.py                    # SQLAlchemy models for user data
│   └── schemas.py                   # Marshmallow schemas for serialization
│
├── static/                          # Static assets
│   ├── css/
│   │   ├── styles.css               # Main stylesheet
│   │   └── bootstrap.min.css        # Bootstrap CSS
│   ├── js/
│   │   ├── main.js                  # Main JavaScript file
│   │   └── recommendation.js        # Recommendation-specific JavaScript
│   └── img/
│       └── placeholder.jpg          # Placeholder for missing movie posters
│
├── templates/                       # Jinja2 templates
│   ├── base.html                    # Base template
│   ├── index.html                   # Homepage
│   ├── auth/
│   │   ├── login.html               # Login page
│   │   └── register.html            # Registration page
│   ├── movies/
│   │   ├── browse.html              # Movie browsing page
│   │   ├── detail.html              # Movie detail page
│   │   └── search.html              # Search results page
│   ├── recommendations/
│   │   ├── personal.html            # Personalized recommendations
│   │   ├── similar.html             # Similar movies
│   │   └── explanation.html         # Why recommended page
│   └── user/
│       ├── profile.html             # User profile page
│       └── watchlist.html           # User watchlist page
│
├── instance/                        # Instance-specific files (ignored by git)
│   └── movie_recommender.db         # SQLite database
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
├── .env.example                     # Example environment variables file
├── .gitignore                       # Git ignore file
└── model_training.py                # Script to train and save recommendation models
``` 