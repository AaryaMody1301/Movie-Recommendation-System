# Movie Recommendation System — Current Project Summary

## Status

The repository has been repaired and production-hardened into one coherent Flask application. The active architecture uses a single application factory, blueprint routes, SQLAlchemy persistence with versioned Flask-Migrate/Alembic schema history, one canonical catalog loader, modern content embeddings, persisted-user collaborative signals, bounded hybrid fusion, optional TMDb enrichment, and GitHub Actions validation.

This document describes what is implemented **now**. Historical repair phases are tracked in `ROADMAP.md`.

## Implemented Application

### Flask application and production configuration

- `app.create_app` is the canonical application factory.
- `wsgi:app` is the production WSGI target and requires `FLASK_ENV=production`.
- Unknown `FLASK_ENV` values fail closed rather than silently selecting development configuration.
- Production refuses a missing secret key or the public development fallback value.
- `run.py` remains the development entry point and supports embedding rebuild/batch options.
- Routes are organized under `blueprints/` for top-level pages, authentication, movie browsing/search/details, recommendations, user interactions, and health checks.
- WTForms/CSRF protection and Flask-Login are integrated into the active application.

### Persistent user data and migrations

SQLAlchemy-backed models persist registered users, ratings, watchlist entries, durable TMDb mappings, and normalized TMDb enrichment cache rows. Ratings and watchlist entries are subject to database uniqueness/relationship constraints.

The default local database is SQLite under Flask's `instance/` directory, with `DATABASE_URI` available for an alternate SQLAlchemy database.

Schema state is versioned under `migrations/` with Flask-Migrate/Alembic. Runtime application startup no longer calls `db.create_all()`. Fresh databases are created with `flask --app app db upgrade`, and CI verifies both migration application and migration/ORM drift with `flask --app app db check`.

### Canonical catalog and baseline ratings

`data/movies.csv` is the single repository copy of the local movie catalog and is loaded through the application-owned `DataLoader`.

`data/ratings.csv` is baseline/offline data for data/model tooling. It is deliberately **not** treated as registered-user behavior by the online personalization service. Collaborative personalization is trained from ratings persisted by actual application users.

### Search and browsing

The active catalog layer supports literal-safe title search, token-aware genre matching, pagination across complete result sets, configurable sorting, and catalog fallbacks when external enrichment is unavailable.

### Content-based recommendations

`models/content_based.py` is the active content recommender. It uses Sentence Transformers embeddings over the complete indexed catalog.

The content cache is validated using version/model/catalog fingerprints. Compatible caches can be loaded without constructing the transformer; incompatible caches rebuild instead of silently returning stale recommendations. Similarity is calculated on demand rather than storing a dense all-pairs similarity matrix.

CI includes a small real-model smoke test using `sentence-transformers/all-MiniLM-L6-v2`, which verifies the actual Torch/Transformers/Sentence-Transformers compatibility boundary without rebuilding the full catalog.

### Collaborative and hybrid recommendations

`models/collaborative_filtering.py` implements matrix-factorization recommendations with corrected raw user/movie ID handling and serialization support.

For the online application, `services/recommendation_service.py` builds the collaborative model lazily from SQLAlchemy `Rating` rows only after configurable minimum interaction/user/item thresholds are met. Hybrid fusion supports normalized weighted-score and reciprocal-rank strategies over bounded candidate lists, with exclusion of already-known items and recommendation reasons based on contributing signals.

Cold-start behavior falls back to available content similarity, popularity from persisted application-user ratings, or deterministic catalog candidates.

### TMDb enrichment

`services/tmdb_service.py` and `services/movie_service.py` provide optional external enrichment without making TMDb a hard dependency for local catalog availability.

Implemented behavior includes local-title/year matching, persisted mappings and enrichment, bounded request timeouts, retry/backoff for transient failures, process-local HTTP caching, configurable TTLs, stale-cache fallback, configurable watch-provider region, posters/backdrops/details, and TMDb similar titles.

Without a TMDb API key, the local application continues to operate with reduced enrichment.

### Observability and deployment

The application exposes:

- `GET /health/live` for liveness;
- `GET /health/ready` for database + catalog readiness;
- text logging for local development;
- structured JSON logging for production;
- per-request method/path/status/duration telemetry.

The recommender is intentionally a degradable rather than critical readiness dependency because the application retains catalog/fallback behavior when it is disabled or unavailable.

Production startup is documented around migrations followed by `wsgi:app` and is validated with Gunicorn in CI. See `DEPLOYMENT.md`.

## Testing and CI

GitHub Actions runs on pull requests and pushes to `main` and covers:

- Python 3.10 and Python 3.13 test boundaries;
- pytest with branch coverage and an enforced 55% project floor;
- `ResourceWarning` as a test error plus explicit SQLAlchemy engine disposal between isolated tests;
- Python compilation checks;
- fatal Ruff correctness rules;
- Bandit medium/high severity scanning;
- an explicit allowlist guard for trusted local pickle deserialization sites;
- `pip-audit` runtime dependency auditing;
- CPU-only PyTorch installation on CPU runners to avoid unused CUDA packages;
- a real Sentence Transformers content-recommender smoke test;
- versioned database upgrade and migration-drift checks;
- Gunicorn configuration validation, production boot, and liveness/readiness probes.

The CI workflows use read-only repository permissions and do not require application secrets or live TMDb credentials.

## Repository Boundaries

- `app.py` — application factory and extension initialization.
- `config.py` — fail-closed environment configuration.
- `blueprints/` — HTTP route layer.
- `services/` — application/business logic.
- `models/` — recommendation algorithms and evaluation helpers.
- `data/` — canonical catalog, offline ratings, and `DataLoader`.
- `database/` — Flask-SQLAlchemy and Flask-Migrate extension setup plus ORM models.
- `migrations/` — committed schema history.
- `forms/` — authentication forms.
- `templates/`, `static/` — web UI.
- `scripts/` — CI/security/recommender smoke helpers.
- `tests/` — regression and production-hardening suite.
- `.github/workflows/` — CI and deployment smoke workflows.

See `README.md` for setup/usage and `CONTRIBUTING.md` for the developer contract.

## Security Posture

The repository avoids committed runtime secrets, fails closed for unsafe production session configuration, loads TMDb credentials from configuration, uses CSRF/session protections, bounds outbound requests, audits runtime dependencies, scans application code, constrains pickle deserialization to reviewed local artifacts, and validates production startup in CI.

Security reporting guidance is in `SECURITY.md`.

## Optional Future Enhancements

The core repair/hardening work is complete. Reasonable future product or infrastructure enhancements include:

- an external/shared cache and managed production database for multi-instance deployments;
- scheduled/background TMDb refresh instead of refresh-on-access behavior;
- container/image packaging and an opinionated deployment manifest;
- broader browser/end-to-end accessibility and UI tests;
- expanded offline recommendation evaluation/monitoring and model-quality benchmarks;
- versioned release/changelog automation.

These are optional enhancements, not unresolved correctness defects in the completed repair roadmap.
