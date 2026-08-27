"""Database extension and initialization helpers."""

import sqlite3

import click
from flask.cli import with_appcontext
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


def init_db():
    """Create any missing tables from the SQLAlchemy model metadata."""
    # Import models before create_all so every mapped table is registered.
    from database import models  # noqa: F401

    db.create_all()


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Create any missing database tables."""
    init_db()
    click.echo("Initialized the database.")


def init_app(app):
    """Attach SQLAlchemy and database CLI commands to a Flask app."""
    db.init_app(app)
    app.cli.add_command(init_db_command)

    # The project does not have Alembic migrations yet. Until Phase 7 adds them,
    # create missing tables without dropping or rewriting existing user data.
    with app.app_context():
        init_db()
