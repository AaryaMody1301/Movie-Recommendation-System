# Movie Recommendation System - Project Summary

This project implements a movie recommendation system using Flask, Pandas, and Scikit-learn, focusing on content-based filtering using movie genres.

## Key Components

### 1. Recommendation Engine (`recommendation.py`)
- Uses TF-IDF vectorization to convert movie genres into numerical vectors
- Calculates similarity between movies using cosine similarity
- Implements memory-efficient batch processing for larger datasets
- Provides movie search and filtering by genres

### 2. Web Application (`app.py`)
- Implements a Flask server with routes for:
  - Home page with featured movies
  - Movie detail pages with recommendations
  - Search functionality
  - Genre-based browsing

### 3. User Interface (HTML/CSS/JS)
- Responsive Bootstrap-based design
- Dynamic search functionality with AJAX
- Movie cards with placeholders for images
- Genre filtering and navigation

## Technical Implementation Details

### Content-Based Filtering
The system analyzes movie genres using TF-IDF (Term Frequency-Inverse Document Frequency) to create vector representations of each movie. When a user views a movie, the system calculates the cosine similarity between that movie's vector and all other movies to find the most similar ones.

### Memory Optimization
The implementation uses techniques like:
- Linear kernel instead of full cosine similarity matrix
- On-demand similarity calculation
- Optional sampling for development/testing
- Garbage collection to free memory

### Data Handling
- Loads movie data from CSV files
- Error handling for missing or malformed data
- Flexible schema to accommodate different CSV structures

## Possible Enhancements

1. **Collaborative Filtering**: Add user ratings and implement user-based recommendations
2. **Improved Content Analysis**: Incorporate additional movie metadata like plot summaries, directors, actors
3. **Database Integration**: Replace CSV files with a proper database for better performance
4. **User Accounts**: Add authentication and user profiles for personalized recommendations
5. **Visual Improvements**: Add real movie posters from an API like TMDB

## How to Run the System

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000/ in your web browser. 

# TMDb API Integration Summary

## Overview
This document summarizes the enhancement of our Movie Recommendation System with The Movie Database (TMDb) API integration. The integration greatly enriches the movie data, improves recommendation quality, and provides a more engaging and informative user interface.

## Implemented Features

### 1. TMDb Service Module (`services/tmdb_service.py`)
- Created a dedicated module for TMDb API interactions
- Implemented comprehensive functions for movie data retrieval:
  - `search_movie_by_title()`: Finds potential matching movies in TMDb
  - `get_movie_details()`: Fetches comprehensive details for a specific movie
  - `get_watch_providers()`: Retrieves streaming/rental/purchase options (region-specific)
  - `get_similar_movies()`: Obtains TMDb-recommended similar movies
  - `find_tmdb_id_for_movie()`: Matches local movie entries with TMDb IDs
- Implemented robust caching using Python's dictionary and LRU cache to minimize API requests
- Added error handling for network issues, rate limits, and other potential API problems

### 2. Movie Service Integration (`services/movie_service.py`)
- Enhanced the existing movie service to incorporate TMDb data
- Added functions to associate local movie data with TMDb IDs
- Implemented `enrich_movie_with_tmdb()` to combine local and TMDb data
- Added support for TMDb similar movies recommendations

### 3. Enhanced Recommendation Engine (`recommendation.py`)
- Improved content-based filtering by incorporating TMDb keywords, cast, and crew data
- Implemented caching for TMDb keyword data
- Created a more sophisticated TF-IDF matrix that includes enhanced movie metadata
- Added support for toggling between traditional content-based and TMDb recommendations

### 4. Flask Route Updates (`app.py`)
- Enhanced the movie detail route to include TMDb data
- Added a new route for viewing movies directly from TMDb IDs
- Implemented a recommendation method toggle feature (content-based vs TMDb)

### 5. User Interface Enhancements (`templates/movie.html`)
- Completely redesigned the movie detail page to showcase TMDb data:
  - Added movie poster and backdrop images
  - Displayed comprehensive movie information (title, overview, release date, runtime, etc.)
  - Added TMDb rating with visual indicator
  - Included cast and director information
  - Added a "Where to Watch" section with streaming/rental/purchase options
  - Embedded YouTube trailers
  - Listed keywords associated with the movie
  - Enhanced the recommendation display with movie posters and additional information
- Implemented a toggle switch to choose between recommendation methods

### 6. Configuration and Security
- Updated the example environment file (`.env.example`) to include TMDb API key configuration
- Added necessary dependencies to `requirements.txt`
- Ensured the TMDb API key is loaded securely from environment variables

## Technical Improvements

### 1. Data Enrichment
The integration significantly enhances the movie data available in the system. For each movie, we now have:
- High-quality poster and backdrop images
- Detailed movie information (overview, release date, runtime, etc.)
- Cast and director information
- Associated keywords
- Trailers
- Watch providers (streaming services, rental options, etc.)

### 2. Improved Recommendations
The recommendation quality has been improved in two ways:
- **Enhanced Content-Based Filtering**: By incorporating TMDb keywords, cast, and crew data into the TF-IDF matrix
- **TMDb Similar Movies**: Offering an alternative recommendation source from TMDb's own algorithm

### 3. Caching and Performance
To ensure good performance and respect API rate limits:
- Implemented dictionary-based caching for API responses
- Used Python's LRU cache for frequently accessed functions
- Set appropriate cache expiration times (24 hours for most data)

### 4. Error Handling
Robust error handling has been implemented throughout the TMDb integration:
- Network error handling for API requests
- Graceful fallbacks when TMDb data is unavailable
- Appropriate user feedback when errors occur

## User Experience Improvements
The TMDb integration significantly enhances the user experience:
- Visually appealing movie detail pages with posters and backdrops
- Rich movie information beyond basic metadata
- Video content (trailers) embedded directly in the page
- Information about where to watch movies
- Multiple recommendation options
- Better recognition of movies through official artwork

## Future Enhancements
Potential future improvements to the TMDb integration:
- Periodic background updating of TMDb data
- More sophisticated matching between local movie database and TMDb
- Expanded use of TMDb endpoints (reviews, popular movies, etc.)
- User preferences for default recommendation method 