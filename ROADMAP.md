# Movie Recommendation System Repair Roadmap

This roadmap records the eight independently testable phases used to repair and consolidate the repository, followed by the final production-hardening pass. Completed phases are historical milestones; the current implementation is summarized in `SUMMARY.md`.

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
- Reconcile ORM models and authentication service field names.
- Add foreign keys and uniqueness constraints for ratings/watchlists.
- Stop silently falling back to in-memory SQLite for unsupported database URIs.
- Create missing forms and templates required by enabled blueprints.

**Status: complete and merged.**

## Phase 3 — Data/service contract repair

Goal: make every service operate against one documented DataLoader API.

- Standardize on `movies_df`, `ratings_df`, and public getter methods.
- Remove direct references to nonexistent loader attributes.
- Remove duplicate DataLoader instances and use the application-owned instance.
- Align blueprint imports with real service function names and signatures.
- Persist user ratings/watchlists instead of using placeholders or in-memory mutations.
- Add validation for pagination, sorting, and API request payloads.

**Status: complete and merged.**

## Phase 4 — Search, browse, and recommendation correctness

Goal: fix visible product behavior before optimizing model quality.

- Remove premature result truncation from search and genre browsing.
- Make search literal-safe and pagination-correct.
- Match genres as parsed values rather than regex substrings.
- Normalize fallback recommendation output shapes.
- Ensure all movie IDs can receive recommendations.
- Validate embedding caches against dataset/model/config fingerprints.
- Wire documented model settings into the active application.

**Status: complete and merged.**

## Phase 5 — Personalized, collaborative, and hybrid recommendations

Goal: rebuild personalization on top of correct persistent user interactions.

- Replace synthetic ratings as the production recommendation source.
- Repair collaborative recommender raw-ID handling and model serialization.
- Rebuild personalized recommendations from persisted user ratings.
- Implement hybrid scoring with normalized, configurable weights.
- Avoid dense full-catalog O(N²) similarity matrices where possible.
- Add recommendation explanations based on real contributing signals.

**Status: complete and merged.**

## Phase 6 — TMDb enrichment and performance

Goal: make external enrichment reliable without making page loads fragile.

- Improve local-to-TMDb movie matching and persist resolved mappings.
- Introduce durable/cacheable enrichment rather than repeating title searches.
- Add retry/backoff policy for transient TMDb failures.
- Move expensive enrichment away from large synchronous page-load batches.
- Make watch-provider region configurable.
- Establish cache invalidation and expiry rules.

**Status: complete and merged.**

## Phase 7 — Tests, CI, deployment, and observability

Goal: make regressions difficult to reintroduce.

- Add unit tests for DataLoader, services, auth, TMDb client, and recommenders.
- Add Flask route/integration smoke tests.
- Add GitHub Actions for supported Python versions.
- Add linting/static checks and dependency/security checks.
- Verify Gunicorn production startup.
- Add structured logging and health/readiness endpoints.
- Document production environment variables and deployment commands.

**Status: complete and merged via PR #7.**

## Phase 8 — Repository cleanup and documentation

Goal: leave a coherent project rather than a collection of historical implementations.

- Remove obsolete/dead modules after migrated functionality is verified.
- Establish `data/movies.csv` as the single committed movie catalog.
- Update README/project structure to match reality.
- Update `SUMMARY.md` to distinguish implemented features from future work.
- Add contribution/development instructions and troubleshooting guidance.
- Complete final repository hygiene and historical cleanup documentation.

**Status: complete and merged via PR #8.**

## Post-roadmap production hardening

The final validation pass closed issues found only after all eight repair phases were merged:

- production rejects missing/public development secret keys and unknown environment names;
- `wsgi:app` is explicitly production-only;
- Flask-Migrate/Alembic owns database creation and schema evolution;
- a reviewed baseline migration represents the current application schema;
- deployment CI upgrades the database and checks migration/ORM drift before Gunicorn starts;
- SQLite test engines are disposed cleanly and `ResourceWarning` is an error;
- coverage has an enforced 55% floor;
- Linux CI preinstalls CPU-only PyTorch instead of downloading unused CUDA runtimes;
- CI exercises the real Sentence Transformers model boundary on a small catalog;
- setup, deployment, contribution, summary, and roadmap documentation match the hardened runtime contract.

**Status: complete in the final production-hardening pass.**
