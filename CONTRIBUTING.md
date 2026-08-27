# Contributing

Contributions should preserve the repository's current architecture and the data/security boundaries established by the repair roadmap.

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
```

For most application/test work, avoid downloading the transformer model by setting:

```bash
export FLASK_ENV=testing
export RECOMMENDER_ENABLED=false
```

Run the development server with `python run.py` when the actual web application is needed.

## Architecture Boundaries

Please keep these contracts intact unless a change deliberately replaces them and includes migration tests/documentation:

- `app.create_app` is the canonical Flask application factory.
- Routes live in `blueprints/`; request-independent business logic belongs in `services/`.
- `data/movies.csv` is the repository's canonical local movie catalog. Use `MOVIES_CSV` to point at another catalog instead of committing duplicate copies.
- `data/ratings.csv` is baseline/offline data. Online collaborative personalization must use ratings persisted by registered application users in SQLAlchemy.
- The application-owned `DataLoader` is the canonical catalog loader for request services.
- Content recommendations use `models/content_based.py`; do not reintroduce the removed standalone TF-IDF implementation.
- Hybrid recommendation code should fuse bounded candidate lists rather than allocate dense full-catalog pairwise matrices.
- TMDb enrichment must remain optional: network/API failures must not make the local catalog unusable.
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

New behavior should include a focused regression test. Existing phase-named test files are historical; new tests may be grouped by feature when that is clearer.

## Quality and Security Checks

The CI quality job runs:

```bash
python -m compileall -q app.py observability.py blueprints data database models services scripts tests
ruff check app.py observability.py blueprints data database models services scripts tests --select E9,F63,F7,F82
python scripts/check_pickle_boundary.py
bandit -q -r app.py observability.py blueprints data database models services -ll -s B301
```

Runtime dependencies are audited with:

```bash
pip-audit -r requirements.txt --progress-spinner off
```

Do not solve a failing security check by globally disabling it. If a finding is intentionally accepted, document the trust boundary narrowly and keep CI able to catch new occurrences.

## Database and Data Changes

When changing ORM models or interaction semantics:

- preserve uniqueness/foreign-key constraints for user-owned records;
- update tests for persistence and duplicate handling;
- describe any migration requirement in the pull request and deployment documentation;
- never substitute synthetic CSV interactions for production user ratings.

When changing the movie catalog schema, update `DataLoader`, recommender cache fingerprint behavior, affected services, and readiness tests together.

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

Update documentation in the same pull request when changing public setup commands, configuration, routes, data contracts, recommendation semantics, health behavior, or production startup.

For deployment-specific information, keep `DEPLOYMENT.md` aligned with the actual WSGI/health contract. For user/developer operational issues, update `TROUBLESHOOTING.md`.
