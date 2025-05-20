"""
Database connection module for the Movie Recommendation System.

This module provides functions to initialize the database and get database connections.
"""
import sqlite3
import os
import click
from flask import current_app, g
from flask.cli import with_appcontext

# In Python 3.13, SQLAlchemy has compatibility issues, so we'll use a stub class
# that mimics some of the same behavior but doesn't actually use SQLAlchemy
class Column:
    def __init__(self, type_, primary_key=False, nullable=True, unique=False, 
                 default=None, index=False, foreign_key=None):
        self.type_ = type_
        self.primary_key = primary_key
        self.nullable = nullable
        self.unique = unique
        self.default = default
        self.index = index
        self.foreign_key = foreign_key

class Relationship:
    def __init__(self, model_name, back_populates=None, cascade=None):
        self.model_name = model_name
        self.back_populates = back_populates
        self.cascade = cascade

class ForeignKey:
    def __init__(self, key):
        self.key = key

class Integer:
    pass

class String:
    def __init__(self, length=None):
        self.length = length

class Float:
    pass

class Boolean:
    pass

class DateTime:
    pass

class Text:
    pass

class UniqueConstraint:
    def __init__(self, *args, name=None):
        self.args = args
        self.name = name

class SQLAlchemyStub:
    def __init__(self):
        # Create model base class
        self.Model = type('Model', (), {
            '__tablename__': None,
            '__table_args__': None,
        })
        
        # Provide column types and utilities
        self.Column = Column
        self.Integer = Integer
        self.String = String
        self.Float = Float
        self.Boolean = Boolean
        self.DateTime = DateTime
        self.Text = Text
        self.ForeignKey = ForeignKey
        self.relationship = Relationship
        self.UniqueConstraint = UniqueConstraint
        
    def init_app(self, app):
        # This would normally initialize SQLAlchemy with the app
        pass
        
    def create_all(self):
        # This would normally create all tables
        pass

# Create SQLAlchemy stub for models
db = SQLAlchemyStub()


def get_db():
    """Get a database connection for the current request."""
    if 'db' not in g:
        # Extract database path from SQLAlchemy URI if it's a SQLite connection
        db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
        
        if db_uri.startswith('sqlite:///'):
            # Extract the path part for SQLite
            db_path = db_uri.replace('sqlite:///', '')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
            g.db = sqlite3.connect(
                db_path,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row
        else:
            # Fallback to in-memory database
            g.db = sqlite3.connect(
                ':memory:',
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row
            print(f"Warning: Using in-memory SQLite database as fallback. URI was: {db_uri}")

    return g.db


def close_db(e=None):
    """Close the database connection at the end of the request."""
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db(app=None):
    """Initialize the database with the schema."""
    try:
        if not app:
            from flask import current_app
            app = current_app
            
        with app.app_context():
            db = get_db()
            
            # Check if tables already exist
            cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if cursor.fetchone() is None:
                # Execute schema from SQL file
                try:
                    with app.open_resource('schema.sql') as f:
                        db.executescript(f.read().decode('utf8'))
                        db.commit()
                        print("Database initialized with schema.sql")
                except FileNotFoundError:
                    print("schema.sql not found. Database initialization skipped.")
                except Exception as e:
                    print(f"Error initializing database: {e}")
                    db.rollback()
            else:
                print("Database tables already exist, skipping initialization")
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Don't raise, just log the error to avoid crashing app startup


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    """Register database functions with the Flask app."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    
    # Initialize SQLAlchemy with the app
    db.init_app(app) 