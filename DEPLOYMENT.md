# Production Deployment

This application exposes a WSGI object as `wsgi:app` and is intended to run behind a production WSGI server such as Gunicorn.

## Supported Python versions

The project baseline is Python 3.10+. CI exercises Python 3.10 and Python 3.13 as the supported lower and upper boundaries for the current dependency set.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
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

`TMDB_API_KEY` is optional. Without it, TMDb enrichment remains disabled and the local catalog continues to work.

`RECOMMENDER_ENABLED=true` loads the content recommender at application startup. Set it to `false` for lightweight web/health-only deployments or diagnostic startup. With the recommender disabled, catalog browsing remains available and readiness reports the recommender as `disabled` rather than failing the service.

Set `APP_VERSION` to a release/version identifier if your deployment system provides one. It is returned by the health endpoints.

See `.env.example` for the complete configuration surface, including TMDb cache/retry settings and recommendation tuning.

## Validate configuration

Gunicorn supports validating configuration without starting a serving loop:

```bash
gunicorn --check-config wsgi:app
```

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

`.github/workflows/deployment-smoke.yml` performs the same basic production validation on pull requests and pushes to `main`:

1. install runtime dependencies;
2. run `gunicorn --check-config wsgi:app`;
3. boot Gunicorn with `FLASK_ENV=production` and the recommender disabled;
4. verify `/health/live` and `/health/ready` return healthy responses.

The smoke test deliberately disables the transformer recommender so it validates the web/database/catalog deployment boundary without downloading model assets from an external service.
