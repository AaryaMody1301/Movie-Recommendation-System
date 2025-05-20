"""
Development server script for Movie Recommendation System.

This script creates and runs the Flask application in development mode.
"""
import sys
import os
import argparse
import logging
from app import create_app

if __name__ == '__main__':
    # Set logging to WARNING for all modules except werkzeug (Flask server address)
    logging.basicConfig(level=logging.WARNING)
    for logger_name in [
        '',  # root logger
        'models.content_based',
        'data.data_loader',
        'services.movie_service',
        'services.recommendation_service',
        'services.tmdb_service',
        'app',
        '__main__',
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    # Werkzeug logger: keep default (shows address and errors)
    
    # First parse only our custom arguments
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--rebuild-embeddings', action='store_true', 
                        help='Force rebuild of recommendation model embeddings')
    pre_parser.add_argument('--max-movies', type=int, default=None,
                        help='Maximum number of movies to process (default: all)')
    
    # Parse only our known args and leave the rest for Flask
    embedding_args, remaining_argv = pre_parser.parse_known_args()
    
    # Update sys.argv to only contain Flask args
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # Simple parser for Flask options (better error messages)
    flask_parser = argparse.ArgumentParser(description='Run the movie recommendation app')
    flask_parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    flask_parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    flask_parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    flask_args = flask_parser.parse_args(remaining_argv)
    
    # Convert embedding_args to dictionary
    embedding_args_dict = vars(embedding_args)
    
    # Create the application with our parsed embedding args
    app = create_app(embedding_args=embedding_args_dict)
    
    # Print the address and port before running the server
    print(f"\nMovie Recommendation System running at: http://{flask_args.host}:{flask_args.port}\n")
    # Run the application with the remaining Flask args
    app.run(
        host=flask_args.host, 
        port=flask_args.port, 
        debug=not flask_args.no_debug
    )