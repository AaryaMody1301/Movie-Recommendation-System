# Production Deployment

This application exposes a production-only WSGI object as `wsgi:app` and is intended to run behind a production WSGI server such as Gunicorn. Database schema state is managed by Flask-Migrate/Alembic and must be upgraded before application processes start.

## Supported Python versions

The project baseline is Python 3.10+. CI exercises Python 3.10 and Python 3.13 as the supported lower and upper boundaries for the current dependency set.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-only Linux hosts, you can avoid pulling unused CUDA runtime wheels by installing the official CPU PyTorch build first, as CI does:

```bash
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For local development and CI tooling, install `requirements-dev.txt` instead.

## Required production configuration

At minimum, set:

```bash
export FLASK_ENV=production
export SECRET_KEY='replace-with-a-long-random-secret'
export DATABASE_URI='sqlite:///movie_recommender.db'
export MOVIES_CSV='data/movies.csv'
export LOG_FORMAT=json
```

`wsgi:app` refuses to initialize unless `FLASK_ENV=production`. Production also fails closed when `SECRET_KEY` is missing or still uses the public development fallback.

Generate a suitable secret, for example:

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

`TMDB_API_KEY` is optional. Without it, TMDb enrichment remains disabled and the local catalog continues to work.

`RECOMMENDER_ENABLED=true` loads the content recommender at application startup. Set it to `false` for lightweight web/health-only deployments or diagnostic startup. With the recommender disabled, catalog browsing remains available and readiness reports the recommender as `disabled` rather than failing the service.

Set `APP_VERSION` to a release/version identifier if your deployment system provides one. It is returned by the health endpoints.

See `.env.example` for the complete configuration surface, including TMDb cache/retry settings and recommendation tuning.

## Database migrations

Application startup does not create or mutate database tables. Apply committed migrations before starting Gunicorn:

```bash
flask --app app db upgrade
flask --app app db check
```

`db upgrade` applies migrations up to the repository head. `db check` fails if the ORM metadata would require another migration, providing a deployment guard against schema drift.

When models change during development:

```bash
flask --app app db migrate -m "describe the schema change"
# Review the generated revision carefully.
flask --app app db upgrade
flask --app app db check
```

Commit generated revisions under `migrations/` with the model change.

### Existing databases created before migrations

The first committed revision (`0001_initial`) represents the schema that existed when migration support was introduced. For an existing database already created by that schema, do **not** run the baseline `upgrade` directly because its tables already exist.

1. Back up the database.
2. Deploy the code containing `migrations/`, but do not start application workers yet.
3. Mark the matching existing schema as the baseline:

```bash
flask --app app db stamp 0001_initial
flask --app app db check
```

4. Continue only if `db check` reports no pending schema operations. If it reports drift, restore/retain the backup and reconcile that database with a reviewed migration rather than forcing the stamp.
5. Apply later revisions normally with `flask --app app db upgrade`.

For a new/empty database, always use `flask --app app db upgrade`; do not stamp it.

## Validate configuration

After configuration and migrations are ready, validate the WSGI target:

```bash
gunicorn --check-config wsgi:app
```

A missing production environment or secret causes this command to fail, which is intentional.

## Start Gunicorn

A minimal production command is:

```bash
gunicorn --workers 2 --bind 0.0.0.0:8000 wsgi:app
```

Tune worker count and timeout for the deployment environment and available memory. Content-model initialization can be memory-intensive, so validate worker count against the actual model/cache footprint before increasing it.

## Health endpoints

Two unauthenticated operational endpoints are available:

- `GET /health/live` returns HTTP 200 when the Flask process can serve requests.
- `GET /health/ready` returns HTTP 200 only when the database is reachable and the local movie catalog is loaded. It returns HTTP 503 when either critical dependency is unavailable.

The content recommender is reported as `ok`, `degraded`, or `disabled`, but it is intentionally not a critical readiness dependency because the application has catalog/recommendation fallbacks.

Example:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
```

## Logs

`LOG_LEVEL` controls severity. `LOG_FORMAT=json` emits one JSON object per line with UTC timestamp, level, logger, message, and exception information when present. Production defaults to JSON when `LOG_FORMAT` is not set.

Every HTTP request also emits a completion event containing method, path, status code, and request duration in milliseconds.

## CI deployment smoke test

`.github/workflows/deployment-smoke.yml` validates the production boundary on pull requests and pushes to `main`:

1. install CPU PyTorch plus runtime dependencies;
2. create an empty database with `flask --app app db upgrade`;
3. verify the migration head matches ORM metadata with `flask --app app db check`;
4. run `gunicorn --check-config wsgi:app`;
5. boot Gunicorn with `FLASK_ENV=production` and the recommender disabled;
6. verify `/health/live` and `/health/ready` return healthy responses.

The deployment smoke deliberately disables full recommender startup so it validates web/database/catalog deployment independently. The main CI workflow has a separate real Sentence Transformers smoke job to validate the model dependency boundary.
