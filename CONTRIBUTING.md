# Contributing

Contributions should preserve the repository's current architecture and the data/security boundaries established by the repair roadmap and final production-hardening pass.

## Development Setup

Use Python 3.10 or newer. GitHub Actions currently validates Python 3.10 and Python 3.13.

```bash
git clone https://github.com/AaryaMody1301/Movie-Recommendation-System.git
cd Movie-Recommendation-System

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env           # Windows: copy .env.example .env
flask --app app db upgrade
```

For most application/test work, avoid downloading the transformer model by setting:

```bash
export FLASK_ENV=testing
export RECOMMENDER_ENABLED=false
```

Run the development server with `python run.py` when the actual web application is needed.

## Architecture Boundaries

Keep these contracts intact unless a change deliberately replaces them and includes migration tests/documentation:

- `app.create_app` is the canonical Flask application factory.
- `wsgi:app` is production-only and requires `FLASK_ENV=production`.
- Routes live in `blueprints/`; request-independent business logic belongs in `services/`.
- `data/movies.csv` is the repository's canonical local movie catalog. Use `MOVIES_CSV` to point at another catalog instead of committing duplicate copies.
- `data/ratings.csv` is baseline/offline data. Online collaborative personalization must use ratings persisted by registered application users in SQLAlchemy.
- The application-owned `DataLoader` is the canonical catalog loader for request services.
- Content recommendations use `models/content_based.py`; do not reintroduce the removed standalone TF-IDF implementation.
- Hybrid recommendation code should fuse bounded candidate lists rather than allocate dense full-catalog pairwise matrices.
- TMDb enrichment must remain optional: network/API failures must not make the local catalog unusable.
- Database schema changes are versioned under `migrations/`; runtime startup must not call `db.create_all()` as a substitute for migrations.
- Pickle model/cache files are trusted local/admin-controlled artifacts only. Do not add request-uploaded or otherwise untrusted pickle deserialization.

## Tests

Run the regression suite before opening a pull request:

```bash
FLASK_ENV=testing RECOMMENDER_ENABLED=false TMDB_API_KEY='' LOG_FORMAT=text pytest
```

To reproduce the CI coverage command:

```bash
pytest --cov=app --cov=blueprints --cov=data --cov=database --cov=models --cov=services --cov=observability --cov-report=term-missing
```

Coverage has an enforced project floor of 55%. `ResourceWarning` is treated as an error, so tests that create database engines must return connections cleanly; shared test teardown disposes tracked SQLAlchemy engines after each case.

New behavior should include a focused regression test. Existing phase-named test files are historical; new tests may be grouped by feature when that is clearer.

## Database and Migration Changes

When changing ORM models or interaction semantics:

1. update the model and focused regression tests;
2. generate a migration:

```bash
flask --app app db migrate -m "describe the schema change"
```

3. review the generated revision—autogeneration is not a substitute for review;
4. apply and verify it:

```bash
flask --app app db upgrade
flask --app app db check
```

5. commit the model, migration, tests, and deployment/documentation changes together.

Preserve uniqueness/foreign-key constraints for user-owned records, and never substitute synthetic CSV interactions for production user ratings. When changing the movie catalog schema, update `DataLoader`, recommender cache fingerprint behavior, affected services, and readiness tests together.

## Quality and Security Checks

The CI quality job runs:

```bash
python -m compileall -q app.py config.py observability.py wsgi.py blueprints data database models services scripts tests migrations
ruff check app.py config.py observability.py wsgi.py blueprints data database models services scripts tests migrations --select E9,F63,F7,F82
python scripts/check_pickle_boundary.py
bandit -q -r app.py config.py observability.py wsgi.py blueprints data database models services -ll -s B301
```

Runtime dependencies are audited with:

```bash
pip-audit -r requirements.txt --progress-spinner off
```

CI additionally runs `scripts/ci_recommender_smoke.py` against the real Sentence Transformers model and runs migration upgrade/drift checks before Gunicorn deployment smoke tests.

Do not solve a failing security check by globally disabling it. If a finding is intentionally accepted, document the trust boundary narrowly and keep CI able to catch new occurrences.

## Production Configuration Changes

Production configuration is fail-closed. Do not add fallbacks for production secrets or silently map unknown environment names to development. If a new required production setting is introduced, update `.env.example`, `README.md`, `DEPLOYMENT.md`, CI smoke configuration, and tests together.

## TMDb Changes

Keep external calls bounded and cache-aware. Changes should preserve:

- configured request timeouts;
- retry/backoff only for transient failures;
- persistent mapping/enrichment cache semantics;
- configurable watch-provider region;
- graceful local-catalog behavior without an API key.

Tests must not require live TMDb credentials.

## Pull Requests

Keep changes scoped and describe:

1. what problem is being solved;
2. what behavior/contracts changed;
3. tests and checks run;
4. configuration, data, database, or deployment impact;
5. security implications, especially authentication, secrets, external requests, or deserialization.

A repository pull request template is provided under `.github/pull_request_template.md`.

Do not commit `.env`, API keys, database files, model/cache artifacts, logs, or other local state. Use `.env.example` for documented configuration names only.

## Documentation

Update documentation in the same pull request when changing public setup commands, configuration, routes, data contracts, recommendation semantics, migration behavior, health behavior, or production startup.

For deployment-specific information, keep `DEPLOYMENT.md` aligned with the actual WSGI/migration/health contract. For user/developer operational issues, update `TROUBLESHOOTING.md`.
