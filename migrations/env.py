from logging.config import fileConfig

from alembic import context

from database import models  # noqa: F401
from database.db import db

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


target_metadata = db.metadata


def run_migrations_offline():
    """Run migrations without creating an Engine connection."""
    url = str(db.engine.url).replace("%", "%%")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations using the Flask-SQLAlchemy Engine."""
    with db.engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
