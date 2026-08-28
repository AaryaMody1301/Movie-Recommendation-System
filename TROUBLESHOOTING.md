# Troubleshooting

This guide covers common development and deployment problems for the current Flask application.

## Application does not start or the catalog is empty

Check the configured movie catalog first:

```bash
MOVIES_CSV=data/movies.csv
```

`data/movies.csv` is the canonical repository catalog. A replacement catalog must contain the fields required by `DataLoader` and recommender startup, including movie IDs, titles, genres, clean titles, and overview data after normalization.

For infrastructure debugging, skip transformer initialization:

```bash
RECOMMENDER_ENABLED=false python run.py
```

If catalog loading itself fails, `/health/ready` returns HTTP 503. Check the application logs for `Catalog data initialization failed` or a missing-column error.

## First startup is slow or downloads a model

With `RECOMMENDER_ENABLED=true`, the application initializes the Sentence Transformers content recommender. On a fresh machine this may download the configured `TRANSFORMER_MODEL` and build embeddings for the catalog.

Options:

- temporarily set `RECOMMENDER_ENABLED=false` to validate the web/database/catalog boundary without the transformer;
- build the cache separately with `python generate_embeddings.py`;
- reduce `EMBEDDING_BATCH_SIZE` if memory is constrained;
- use `python run.py --rebuild-embeddings` only when a forced rebuild is intended.

On CPU-only Linux machines, preinstall the official CPU PyTorch wheel before `requirements.txt` if pip otherwise resolves CUDA runtime packages:

```bash
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The embedding cache is fingerprint-validated against relevant catalog/model inputs, so incompatible caches are rebuilt rather than silently reused.

## Content recommendations are unavailable but browsing works

This is a supported degraded state. The catalog is initialized before the optional content recommender. If transformer initialization fails, browsing/search can remain available and readiness can report the recommender as degraded instead of failing the entire service.

Check logs for the recommender initialization exception and verify:

- the configured transformer model can be resolved;
- the machine has enough memory/disk space;
- `EMBEDDINGS_CACHE_PATH` is writable;
- the movie catalog contains the required content fields.

## Personalized recommendations are empty

Online personalization is based on ratings stored by registered users in the application database, not `data/ratings.csv`.

A collaborative model is only built after the persisted interaction set reaches the configured minimums:

```text
COLLAB_MIN_RATINGS=5
COLLAB_MIN_USERS=2
COLLAB_MIN_ITEMS=2
```

Before those thresholds are met, recommendations use content signals and deterministic/popularity fallbacks. Add real ratings through the application and retry.

If thresholds are met but collaborative results are still absent, inspect logs for `Persisted collaborative model build failed` and verify the persisted movie IDs exist in the current catalog.

## Database tables are missing

Application startup does not create tables automatically. For a new database, run:

```bash
flask --app app db upgrade
```

Then verify that the database matches the current ORM models:

```bash
flask --app app db current
flask --app app db check
```

If `db check` reports pending operations, do not use `db.create_all()` to bypass migrations. Generate/review the required migration during development or deploy the missing committed revision.

## Existing pre-migration database fails on the baseline revision

If the database already contains the current application tables but has no `alembic_version` table, it may have been created before Flask-Migrate was introduced. Back it up and follow the one-time adoption procedure in `DEPLOYMENT.md`.

For a database confirmed to match the baseline schema:

```bash
flask --app app db stamp 0001_initial
flask --app app db check
```

Only retain the stamp if `db check` reports no drift. Do not stamp an empty database; use `db upgrade` instead.

## Ratings or watchlist entries disappear after restart

Confirm `DATABASE_URI` points to the intended persistent database and that migrations have been applied.

The default relative SQLite URI is:

```text
sqlite:///movie_recommender.db
```

Flask-SQLAlchemy resolves that relative path inside Flask's `instance/` directory. Do not delete `instance/` if it contains the local database you intend to keep.

For external databases, ensure the URI is valid and the application process has network/credential access.

## Database/readiness errors

Use migration state plus the operational endpoints to separate schema, process, and dependency failures:

```bash
flask --app app db current
flask --app app db check
curl -i http://127.0.0.1:5000/health/live
curl -i http://127.0.0.1:5000/health/ready
```

- `/health/live` returning 200 means Flask can serve requests.
- `/health/ready` returning 503 means either the database or local movie catalog is unavailable.
- `db check` returning non-zero means the checked-out ORM models and migration history are out of sync.

The content recommender is intentionally non-critical to readiness because the application has fallback behavior.

## TMDb posters/details are missing

TMDb enrichment is optional. Verify `TMDB_API_KEY` is configured and valid if enrichment is expected.

Also check:

- `TMDB_REQUEST_TIMEOUT` for overly aggressive timeouts;
- retry/backoff settings for transient failures;
- `TMDB_LANGUAGE` and `TMDB_WATCH_REGION` for expected locale/provider results;
- logs for mapping/search failures or upstream HTTP errors.

The application persists local-to-TMDb mappings and normalized enrichment. A cached negative mapping or stale entry may remain until its configured TTL expires. Development-only resets should remove only the specific local cache/state you intend to rebuild, not arbitrary production data.

## Watch providers show the wrong country

Set the ISO-style region expected by TMDb using:

```text
TMDB_WATCH_REGION=IN
```

Restart the process after changing configuration. Previously persisted enrichment may remain until refresh/expiry rules permit an update.

## Gunicorn configuration or startup fails

The WSGI target intentionally fails unless production configuration is explicit. At minimum set:

```bash
export FLASK_ENV=production
export SECRET_KEY='a-private-random-secret'
flask --app app db upgrade
gunicorn --check-config wsgi:app
```

Then test a minimal server:

```bash
RECOMMENDER_ENABLED=false gunicorn --workers 1 --bind 127.0.0.1:8000 wsgi:app
```

Probe both health endpoints. If this works with the recommender disabled but fails when enabled, investigate model/cache resources rather than the WSGI boundary.

See `DEPLOYMENT.md` for the supported production configuration.

## Tests fail with ResourceWarning

`ResourceWarning` is an error in the pytest configuration. SQLAlchemy test engines are tracked and disposed after every test; new tests should use application contexts normally and must not retain database connections beyond the test.

If a new warning appears, find the code or fixture retaining the connection rather than filtering or suppressing the warning.

## Tests cannot import application modules

Run pytest from the repository root. `pyproject.toml` configures the root directory on pytest's Python path.

Install the complete development requirements first:

```bash
python -m pip install -r requirements-dev.txt
```

Then run:

```bash
FLASK_ENV=testing RECOMMENDER_ENABLED=false pytest
```

## Dependency or security CI fails

Reproduce the relevant job from `CONTRIBUTING.md`.

For dependency vulnerabilities, update the affected dependency boundary rather than suppressing `pip-audit`. For Bandit/pickle failures, remember that pickle loading is allowed only at the explicitly guarded local model/cache locations enforced by `scripts/check_pickle_boundary.py`.

If the separate recommender smoke fails, reproduce `python scripts/ci_recommender_smoke.py` with network access and CPU PyTorch installed. That job intentionally exercises the real model dependency boundary.

## JSON logs are not appearing in production

Production defaults to JSON logging unless `LOG_FORMAT` overrides it. Set explicitly if needed:

```text
FLASK_ENV=production
LOG_FORMAT=json
LOG_LEVEL=INFO
```

Each request should emit method, path, status code, and duration. If logs are absent, check the process supervisor/container log capture rather than adding a second application logger.

## Safe local reset

Generated files under `instance/` are intentionally ignored by Git. In a disposable development environment, you may remove selected generated caches/models and let the app rebuild them.

Do **not** delete the instance database if it contains user accounts, ratings, watchlists, or persisted TMDb state you need. Back up state before destructive resets.
