# Movie Recommendation System

A Flask-based movie discovery and recommendation application with persistent user accounts, ratings and watchlists; transformer-based content similarity; collaborative and hybrid personalization; optional TMDb enrichment; and versioned SQLAlchemy database migrations.

The application is built around a single Flask application factory (`app.create_app`), SQLAlchemy-backed user interactions, one canonical local movie catalog, Alembic/Flask-Migrate schema history, and production checks that run in GitHub Actions.

## Features

- Browse, search, sort, paginate, and filter the local movie catalog by genre.
- Register and sign in with persistent SQLAlchemy-backed user accounts.
- Save ratings and watchlist entries that survive application restarts.
- Generate full-catalog content recommendations with Sentence Transformers embeddings.
- Train collaborative recommendations lazily from ratings created by registered application users.
- Combine content and collaborative candidates with configurable weighted or rank-based hybrid fusion.
- Explain recommendation signals and provide deterministic cold-start fallbacks.
- Enrich movie pages with optional TMDb metadata, artwork, similar titles, and region-specific watch providers.
- Persist TMDb mappings/enrichment while using retry, timeout, stale-cache, and expiry policies.
- Manage database creation and schema evolution with committed Flask-Migrate/Alembic revisions.
- Expose liveness/readiness endpoints and structured production logging.
- Validate Python 3.10 and 3.13, coverage, static/security checks, dependency auditing, real Sentence Transformers compatibility, migrations, and Gunicorn startup in GitHub Actions.

## Requirements

- Python 3.10 or newer. CI currently exercises Python 3.10 and Python 3.13 as the lower and upper supported boundaries.
- A local movie catalog. The repository's canonical catalog is `data/movies.csv`.
- `TMDB_API_KEY` only if TMDb enrichment is desired. Core local catalog functionality remains available without it.

The first content-recommender startup can download the configured Sentence Transformers model and build the embedding cache. Set `RECOMMENDER_ENABLED=false` when you only need catalog/web/health startup, or pre-generate embeddings with `generate_embeddings.py`.

Linux CPU-only environments can preinstall PyTorch from the official CPU wheel index before installing `requirements.txt`; CI does this to avoid downloading unused CUDA runtime packages.

## Quick Start

```bash
git clone https://github.com/AaryaMody1301/Movie-Recommendation-System.git
cd Movie-Recommendation-System

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env           # Windows: copy .env.example .env
flask --app app db upgrade
python run.py
```

Then open `http://127.0.0.1:5000/`.

For a lightweight startup that skips transformer initialization:

```bash
RECOMMENDER_ENABLED=false python run.py
```

`run.py` also supports `--rebuild-embeddings`, `--embedding-batch-size`, `--host`, `--port`, and `--no-debug`.

## Configuration

Copy `.env.example` to `.env` and change values appropriate for the environment. Important settings include:

| Setting | Purpose |
| --- | --- |
| `FLASK_ENV` | Selects `development`, `testing`, or `production`. Unknown values fail closed. |
| `SECRET_KEY` | Flask session/CSRF secret. Production refuses a missing or development fallback value. |
| `DATABASE_URI` | SQLAlchemy database URI. Relative SQLite URIs are stored under Flask's `instance/` directory. |
| `MOVIES_CSV` | Canonical local movie catalog path; default `data/movies.csv`. |
| `RATINGS_CSV` | Baseline/offline ratings dataset; not the source of online user personalization. |
| `RECOMMENDER_ENABLED` | Enables content-recommender initialization at startup. |
| `TRANSFORMER_MODEL` | Sentence Transformers model used for content embeddings. |
| `EMBEDDINGS_CACHE_PATH` | Fingerprint-validated content embedding cache. |
| `CONTENT_WEIGHT` / `COLLAB_WEIGHT` | Hybrid recommendation weights. |
| `TMDB_API_KEY` | Optional TMDb API credential. |
| `TMDB_WATCH_REGION` | Region used for watch-provider availability. |
| `LOG_LEVEL` / `LOG_FORMAT` | Application log severity and text/JSON output. |

See `.env.example` for the complete configuration surface and `DEPLOYMENT.md` for production-specific settings.

## Database Migrations

Database schema state is versioned under `migrations/`. Application startup does not call `db.create_all()`; create or upgrade a database explicitly:

```bash
flask --app app db upgrade
```

When ORM models change:

```bash
flask --app app db migrate -m "describe the schema change"
# Review the generated migration.
flask --app app db upgrade
flask --app app db check
```

For a database created by a repository version from before migrations were introduced, back it up and follow the one-time adoption procedure in `DEPLOYMENT.md` before applying future migrations.

## Data Contract

`data/movies.csv` is the single canonical movie catalog in the repository. Configure another catalog with `MOVIES_CSV` rather than adding a second copy.

`data/ratings.csv` is baseline/offline data used by data/model tooling. Production personalization does **not** treat it as registered-user behavior. Online collaborative recommendations are built from ratings persisted in the application database, so a new installation remains in cold-start/content mode until real user interactions meet the configured collaborative thresholds.

## Recommendation Architecture

The active recommendation path has three layers:

1. **Content** — `models/content_based.py` creates Sentence Transformers embeddings for the complete indexed catalog, validates caches using model/data fingerprints, and scores similarity on demand.
2. **Collaborative** — `models/collaborative_filtering.py` uses matrix factorization, while `services/recommendation_service.py` builds the online model lazily from SQLAlchemy-backed user ratings.
3. **Hybrid** — `models/hybrid_recommender.py` fuses bounded content and collaborative candidate lists using normalized weighted-score or reciprocal-rank strategies rather than a dense full-catalog pairwise matrix.

When user interaction data is insufficient, the service falls back to available content similarity, persisted application-user popularity, or deterministic catalog candidates.

## TMDb Enrichment

TMDb integration is optional and isolated from core catalog availability. The service uses bounded request timeouts, retry/backoff, process-local HTTP caching, persistent local-to-TMDb mappings, persistent normalized enrichment, stale-cache fallback, and configurable expiry windows.

Movie pages can show posters/backdrops, metadata, TMDb similar titles, and watch-provider availability for `TMDB_WATCH_REGION`. If TMDb is unavailable or no key is configured, local browsing continues to work.

## Project Structure

```text
.
├── app.py                    # Canonical Flask application factory
├── config.py                 # Environment-based configuration
├── observability.py          # Text/JSON logging configuration
├── run.py                    # Development server entry point
├── wsgi.py                   # Production-only WSGI object (wsgi:app)
├── blueprints/               # Auth, main, movies, recommendations, user, health routes
├── services/                 # Auth, movie, recommendation, user, and TMDb business logic
├── models/                   # Content, collaborative, hybrid, and evaluation code
├── data/                     # Canonical catalog, baseline ratings, DataLoader
├── database/                 # SQLAlchemy extension and persistent ORM models
├── migrations/               # Flask-Migrate/Alembic schema history
├── forms/                    # WTForms authentication forms
├── templates/                # Jinja templates
├── static/                   # CSS, JavaScript, and fallback image assets
├── scripts/                  # CI/security/recommender smoke helper scripts
├── tests/                    # Regression and production-hardening tests
├── generate_embeddings.py    # Pre-build the full-catalog embedding cache
├── model_training.py         # Offline model training/evaluation tooling
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Runtime + test/quality tooling
├── DEPLOYMENT.md             # Production, migrations, and health-check guidance
├── CONTRIBUTING.md           # Development and contribution workflow
├── TROUBLESHOOTING.md        # Common local/runtime problems
├── SECURITY.md               # Vulnerability reporting guidance
├── SUMMARY.md                # Current implementation summary
└── ROADMAP.md                # Historical repair roadmap
```

## Development and Tests

Install the development dependency set:

```bash
python -m pip install -r requirements-dev.txt
```

Run the test suite:

```bash
FLASK_ENV=testing RECOMMENDER_ENABLED=false pytest
```

To reproduce the CI coverage command:

```bash
pytest --cov=app --cov=blueprints --cov=data --cov=database --cov=models --cov=services --cov=observability --cov-report=term-missing
```

Coverage has an enforced 55% project floor. CI also runs compile/Ruff correctness checks, a trusted-pickle-boundary check, Bandit medium/high severity scanning, `pip-audit`, a real Sentence Transformers smoke test, migration drift validation, and production Gunicorn probes. See `CONTRIBUTING.md` for the exact commands and expectations.

## Production

The production WSGI target is `wsgi:app`. It refuses to start unless `FLASK_ENV=production`, and production configuration requires a private `SECRET_KEY`.

```bash
export FLASK_ENV=production
export SECRET_KEY='replace-with-a-long-random-secret'
flask --app app db upgrade
gunicorn --check-config wsgi:app
gunicorn --workers 2 --bind 0.0.0.0:8000 wsgi:app
```

Operational endpoints:

- `GET /health/live` — process liveness.
- `GET /health/ready` — database + local catalog readiness; returns 503 when a critical dependency is unavailable.

See `DEPLOYMENT.md` for environment variables, migration/adoption steps, structured logging, health semantics, and the deployment smoke boundary.

## Documentation

- `CONTRIBUTING.md` — development setup, migrations, tests, security/data boundaries, and pull request expectations.
- `TROUBLESHOOTING.md` — startup, model, database, TMDb, personalization, and health-check issues.
- `SECURITY.md` — how to report vulnerabilities responsibly.
- `SUMMARY.md` — the implemented architecture and remaining optional enhancements.
- `ROADMAP.md` — the completed eight-phase repair history plus final hardening.

## Current Limitations / Future Opportunities

The repaired repository is functional and continuously validated. Larger deployments may still choose to add an external/shared cache, a managed production database, background TMDb refresh workers, container packaging, broader browser/end-to-end tests, and deeper offline recommendation evaluation/monitoring.

## License

MIT. See `LICENSE`.
