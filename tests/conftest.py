"""Shared pytest lifecycle cleanup for SQLAlchemy-backed tests."""

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

_test_engines = set()


@event.listens_for(Engine, "engine_connect")
def _track_test_engine(connection):
    _test_engines.add(connection.engine)


@pytest.fixture(autouse=True)
def _dispose_test_engines():
    """Close pooled DBAPI connections after every isolated test application."""
    yield
    for engine in tuple(_test_engines):
        engine.dispose()
    _test_engines.clear()
