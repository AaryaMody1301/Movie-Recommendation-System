"""
Development server script for Movie Recommendation System.

This script creates and runs the Flask application in development mode.
"""
from app import create_app

# Create the Flask app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 