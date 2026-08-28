"""Establish the current application schema.

Revision ID: 0001_initial
Revises: None
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    op.create_table(
        "movie_tmdb_mappings",
        sa.Column("local_movie_id", sa.Integer(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("catalog_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("matched_by", sa.String(length=32), nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("local_movie_id", name=op.f("pk_movie_tmdb_mappings")),
    )
    op.create_index(
        op.f("ix_movie_tmdb_mappings_expires_at"),
        "movie_tmdb_mappings",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_movie_tmdb_mappings_status"),
        "movie_tmdb_mappings",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_movie_tmdb_mappings_tmdb_id"),
        "movie_tmdb_mappings",
        ["tmdb_id"],
        unique=False,
    )

    op.create_table(
        "tmdb_enrichment_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("region", sa.String(length=8), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tmdb_enrichment_cache")),
        sa.UniqueConstraint(
            "tmdb_id",
            "language",
            "region",
            name="tmdb_enrichment_locale",
        ),
    )
    op.create_index(
        op.f("ix_tmdb_enrichment_cache_expires_at"),
        "tmdb_enrichment_cache",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tmdb_enrichment_cache_tmdb_id"),
        "tmdb_enrichment_cache",
        ["tmdb_id"],
        unique=False,
    )

    op.create_table(
        "ratings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "rating >= 0.5 AND rating <= 5.0",
            name=op.f("ck_ratings_rating_range"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_ratings_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ratings")),
        sa.UniqueConstraint("user_id", "movie_id", name="user_movie_rating"),
    )
    op.create_index(op.f("ix_ratings_movie_id"), "ratings", ["movie_id"], unique=False)
    op.create_index(op.f("ix_ratings_timestamp"), "ratings", ["timestamp"], unique=False)
    op.create_index(op.f("ix_ratings_user_id"), "ratings", ["user_id"], unique=False)

    op.create_table(
        "watchlist",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_watchlist_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_watchlist")),
        sa.UniqueConstraint("user_id", "movie_id", name="user_movie_watchlist"),
    )
    op.create_index(op.f("ix_watchlist_added_at"), "watchlist", ["added_at"], unique=False)
    op.create_index(op.f("ix_watchlist_movie_id"), "watchlist", ["movie_id"], unique=False)
    op.create_index(op.f("ix_watchlist_user_id"), "watchlist", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_watchlist_user_id"), table_name="watchlist")
    op.drop_index(op.f("ix_watchlist_movie_id"), table_name="watchlist")
    op.drop_index(op.f("ix_watchlist_added_at"), table_name="watchlist")
    op.drop_table("watchlist")

    op.drop_index(op.f("ix_ratings_user_id"), table_name="ratings")
    op.drop_index(op.f("ix_ratings_timestamp"), table_name="ratings")
    op.drop_index(op.f("ix_ratings_movie_id"), table_name="ratings")
    op.drop_table("ratings")

    op.drop_index(op.f("ix_tmdb_enrichment_cache_tmdb_id"), table_name="tmdb_enrichment_cache")
    op.drop_index(op.f("ix_tmdb_enrichment_cache_expires_at"), table_name="tmdb_enrichment_cache")
    op.drop_table("tmdb_enrichment_cache")

    op.drop_index(op.f("ix_movie_tmdb_mappings_tmdb_id"), table_name="movie_tmdb_mappings")
    op.drop_index(op.f("ix_movie_tmdb_mappings_status"), table_name="movie_tmdb_mappings")
    op.drop_index(op.f("ix_movie_tmdb_mappings_expires_at"), table_name="movie_tmdb_mappings")
    op.drop_table("movie_tmdb_mappings")

    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
