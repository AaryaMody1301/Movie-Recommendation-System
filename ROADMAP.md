# Movie Recommendation System Repair Roadmap

This roadmap converts the repository audit into independently testable repair phases.

## Phase 1 — Security, dependencies, and startup foundation

Goal: make the current application safe to configure and predictable to start before deeper feature work.

- Remove committed TMDb credentials from source.
- Require TMDb credentials through environment/application configuration only.
- Fail closed when TMDb is unavailable instead of generating fake movie data.
- Add bounded HTTP timeouts to TMDb requests.
- Restore required runtime dependencies, including pandas.
- Replace the obsolete `surprise` requirement with `scikit-surprise`.
- Establish Python 3.10+ as the supported baseline.
- Keep `app.create_app` as the canonical application factory during the migration.
- Make the WSGI entry point deterministic for Gunicorn/import-based servers.
- Align `.env.example` and configuration defaults.

**Status: complete and merged.**

## Phase 2 — Application architecture and database integrity

Goal: remove the two competing Flask architectures and make user data persistence coherent.

- Choose one route architecture and register only supported blueprints.
- Remove or migrate duplicate routes from `app.py`.
- Replace the SQLAlchemy stub with the real Flask-SQLAlchemy/SQLAlchemy stack.
- Reconcile `schema.sql`, ORM models, and authentication service field names.
- Add foreign keys and uniqueness constraints for ratings/watchlists.
- Stop silently falling back to in-memory SQLite for unsupported database URIs.
- Create missing forms and templates required by enabled blueprints.

**Status: complete and merged.**

## Phase 3 — Data/service contract repair

Goal: make every service operate against one documented DataLoader API.

- Standardize on `movies_df`, `ratings_df`, and public getter methods.
- Remove direct references to nonexistent `data_loader.movies` / `data_loader.ratings` attributes.
- Remove duplicate DataLoader instances and use the application-owned instance.
- Align blueprint imports with real service function names and signatures.
- Persist user ratings/watchlists instead of using placeholders or in-memory mutations.
- Add validation for pagination, sorting, and API request payloads.

**Status: complete and merged.**

## Phase 4 — Search, browse, and recommendation correctness

Goal: fix visible product behavior before optimizing model quality.

- Remove premature `.head()` truncation from search and genre browsing.
- Make search literal-safe and pagination-correct.
- Match genres as parsed values rather than regex substrings.
- Normalize fallback recommendation output shapes.
- Ensure all movie IDs can receive recommendations, not only a random startup sample.
- Validate embedding caches against dataset/model/config fingerprints.
- Wire documented model settings into the active application.

**Status: complete and merged.**

## Phase 5 — Personalized, collaborative, and hybrid recommendations

Goal: rebuild personalization on top of correct persistent user interactions.

- Replace synthetic ratings as the production recommendation source.
- Repair collaborative recommender raw-ID handling and model serialization.
- Rebuild personalized recommendations from persisted user ratings.
- Implement hybrid scoring with normalized, configurable weights.
- Avoid dense full-catalog O(N^2) similarity matrices where possible.
- Add recommendation explanations based on real contributing signals.

**Status: complete and merged.**

## Phase 6 — TMDb enrichment and performance

Goal: make external enrichment reliable without making page loads fragile.

- Improve local-to-TMDb movie matching and store resolved TMDb IDs.
- Introduce durable/cacheable enrichment rather than repeating title searches.
- Add retry/backoff policy for transient TMDb failures.
- Move expensive enrichment away from large synchronous page-load batches.
- Make watch-provider region configurable.
- Establish cache invalidation and expiry rules.

**Status: in progress on `fix/phase-6-tmdb-performance`.**

## Phase 7 — Tests, CI, deployment, and observability

Goal: make regressions difficult to reintroduce.

- Add unit tests for DataLoader, services, auth, TMDb client, and recommenders.
- Add Flask route/integration smoke tests.
- Add GitHub Actions for supported Python versions.
- Add linting/static checks and dependency/security checks.
- Verify Gunicorn production startup.
- Add structured logging and health/readiness endpoints.
- Document production environment variables and deployment commands.

## Phase 8 — Repository cleanup and documentation

Goal: leave a coherent project rather than a collection of historical implementations.

- Remove obsolete/dead modules after migrated functionality is verified.
- Remove duplicate movie dataset copies or clearly define the canonical data source.
- Update README/project structure to match reality.
- Update SUMMARY.md to distinguish implemented features from future work.
- Add contribution/development instructions and troubleshooting guidance.
- Complete final repository hygiene and historical cleanup documentation.
