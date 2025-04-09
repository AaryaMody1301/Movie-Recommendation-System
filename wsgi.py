"""
WSGI entry point for production deployment of Movie Recommendation System.

This file serves as the entry point for WSGI servers like Gunicorn.
"""
from app import create_app

app = create_app() 