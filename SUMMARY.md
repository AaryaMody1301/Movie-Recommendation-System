# Movie Recommendation System — Current Project Summary

## Status

The repository has been repaired from a collection of partially overlapping implementations into one coherent Flask application. The active architecture uses a single application factory, blueprint routes, SQLAlchemy persistence, one canonical catalog loader, modern content embeddings, persisted-user collaborative signals, bounded hybrid fusion, optional TMDb enrichment, and GitHub Actions validation.

This document describes what is implemented **now**. Historical goals and completed repair phases are tracked in `ROADMAP.md`.

## Implemented Application

### Flask application and routes

- `app.create_app` is the canonical application factory.
- `wsgi:app` is the production WSGI target.
- `run.py` is the development entry point and supports embedding rebuild/batch options.
- Routes are organized under `blueprints/` for top-level pages, authentication, movie browsing/search/details, recommendations, user interactions, and health checks.
- WTForms/CSRF protection and Flask-Login are integrated into the active application.

### Persistent user data

SQLAlchemy-backed models persist registered users and application interactions. Ratings and watchlist entries are not in-memory placeholders and are subject to database uniqueness/relationship constraints.

The default local database is SQLite under Flask's `instance/` directory, with `DATABASE_URI` available for an alternate SQLAlchemy database.

### Canonical catalog and baseline ratings

`data/movies.csv` is the single repository copy of the local movie catalog and is loaded through the application-owned `DataLoader`.

`data/ratings.csv` is baseline/offline data for data/model tooling. It is deliberately **not** treated as registered-user behavior by the online personalization service. Collaborative personalization is trained from ratings persisted by actual application users.

### Search and browsing

The active catalog layer supports:

- literal-safe title search;
- token-aware genre matching;
- pagination across complete result sets rather than pre-truncated samples;
- configurable sorting;
- catalog fallbacks when external enrichment is unavailable.

### Content-based recommendations

`models/content_based.py` is the active content recommender. It uses Sentence Transformers embeddings over the complete indexed catalog rather than the removed historical standalone TF-IDF implementation.

The content cache is validated using version/model/catalog fingerprints. Compatible caches can be loaded without constructing the transformer; incompatible caches rebuild instead of silently returning stale recommendations. Similarity is calculated on demand for the requested movie rather than storing a dense all-pairs similarity matrix.

### Collaborative recommendations

`models/collaborative_filtering.py` implements matrix-factorization recommendations with corrected raw user/movie ID handling and serialization support.

For the online application, `services/recommendation_service.py` builds the collaborative model lazily from SQLAlchemy `Rating` rows only after configurable minimum interaction/user/item thresholds are met. Changes to persisted ratings invalidate the app-scoped collaborative state.

### Hybrid personalization

`models/hybrid_recommender.py` combines bounded content and collaborative candidate lists. It supports:

- normalized weighted-score fusion;
- reciprocal-rank fusion;
- configurable content/collaborative weights;
- exclusion of already-known items;
- recommendation reasons based on contributing signals.

Cold-start behavior falls back to available content similarity, popularity from persisted application-user ratings, or deterministic catalog candidates.

### TMDb enrichment

`services/tmdb_service.py` and `services/movie_service.py` provide optional external enrichment without making TMDb a hard dependency for local catalog availability.

Implemented behavior includes:

- local-title/year matching to TMDb IDs;
- persisted local-to-TMDb mappings;
- persisted normalized enrichment;
- bounded request timeouts;
- retry/backoff for transient failures;
- process-local HTTP response caching;
- configurable fresh/stale/mapping TTLs;
- stale-cache fallback;
- configurable watch-provider region;
- posters/backdrops/details and TMDb similar titles on movie pages.

Without a TMDb API key, the local application continues to operate with reduced enrichment.

### Observability and deployment

The application exposes:

- `GET /health/live` for liveness;
- `GET /health/ready` for database + catalog readiness;
- text logging for local development;
- structured JSON logging for production;
- per-request method/path/status/duration telemetry.

The recommender is intentionally a degradable rather than critical readiness dependency because the application retains catalog/fallback behavior when it is disabled or unavailable.

Production startup is documented around `wsgi:app` and validated with Gunicorn in CI. See `DEPLOYMENT.md`.

## Testing and CI

GitHub Actions runs on pull requests and pushes to `main` and currently covers:

- Python 3.10 and Python 3.13 test boundaries;
- pytest with coverage across application, blueprints, data, database, models, services, and observability;
- Python compilation checks;
- fatal Ruff correctness rules;
- Bandit medium/high severity scanning;
- an explicit allowlist guard for trusted local pickle deserialization sites;
- `pip-audit` runtime dependency auditing;
- Gunicorn configuration validation, production boot, and liveness/readiness probes.

The CI workflows use read-only repository permissions and do not need application secrets or live TMDb credentials.

## Repository Cleanup

Phase 8 establishes the following repository hygiene rules:

- `data/movies.csv` is the only committed movie catalog copy.
- The removed root `recommendation.py` is not an active recommender; `models/content_based.py` is authoritative.
- Synthetic ratings are not generated as a production personalization source.
- Unused Marshmallow schemas/dependency were removed rather than retaining an unreferenced serialization layer.
- The obsolete pre-blueprint movie template and stale project-structure document were removed.
- Placeholder demo/instruction artifacts were removed while the actual fallback image/CSS used by templates remain.
- `.gitignore` has one consolidated set of rules for secrets, local state, model/cache artifacts, logs, editor files, and generated output.

## Current Repository Layout

The important runtime boundaries are:

- `app.py` — application factory and extension initialization.
- `blueprints/` — HTTP route layer.
- `services/` — application/business logic.
- `models/` — recommendation algorithms and evaluation helpers.
- `data/` — canonical catalog, offline ratings, and `DataLoader`.
- `database/` — Flask-SQLAlchemy initialization and ORM models.
- `forms/` — authentication forms.
- `templates/`, `static/` — web UI.
- `scripts/` — CI/security helper scripts.
- `tests/` — regression suite.
- `generate_embeddings.py` — full-catalog embedding cache utility.
- `model_training.py` — offline training/evaluation utility.
- `.github/workflows/` — CI and deployment smoke workflows.

See `README.md` for setup/usage and `CONTRIBUTING.md` for the developer contract.

## Security Posture

The current repository avoids committed runtime secrets, loads TMDb credentials from configuration, uses CSRF/session protections, bounds outbound requests, audits runtime dependencies, scans application code, and constrains pickle deserialization to explicitly reviewed local model/cache artifacts.

Security reporting guidance is in `SECURITY.md`.

## Future Work (Not Yet Implemented)

The following are reasonable future enhancements, not claims about current functionality:

- database migration tooling (for example, versioned schema migrations) for production schema evolution;
- a shared/external cache and production database configuration for multi-instance deployments;
- scheduled/background TMDb refresh instead of refresh-on-access behavior;
- container/image packaging and an opinionated deployment manifest;
- broader browser/end-to-end accessibility and UI tests;
- expanded offline recommendation evaluation/monitoring and model-quality benchmarks;
- higher test coverage in route/UI and offline evaluation paths;
- versioned releases/changelog automation.

These items are intentionally separated from the completed repair work so documentation does not describe implemented features as future work or future ideas as already shipped.
