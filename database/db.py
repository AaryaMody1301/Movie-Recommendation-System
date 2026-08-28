"""Database extension and migration initialization helpers."""

import sqlite3

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)
db = SQLAlchemy(metadata=metadata)
migrate = Migrate(compare_type=True, render_as_batch=True)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Enable SQLite foreign-key enforcement for every SQLite connection."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_db():
    """Compatibility shim for legacy services; prefer ``db.session`` directly."""
    return db.session


def init_app(app):
    """Attach SQLAlchemy and Flask-Migrate to a Flask application."""
    db.init_app(app)
    migrate.init_app(app, db)
