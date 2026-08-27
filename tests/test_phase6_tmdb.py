"""Regression tests for Phase 6 TMDb enrichment and request behavior."""

from datetime import datetime, timedelta, timezone

from flask import Flask

from database.db import db, init_app as init_database
from database.models import MovieTmdbMapping, TmdbEnrichmentCache
import services.movie_service as movie_service
import services.tmdb_service as tmdb_service


def _app(**overrides):
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        TMDB_API_KEY="test-key",
        TMDB_LANGUAGE="en-US",
        TMDB_WATCH_REGION="US",
        TMDB_MAPPING_TTL=3600,
        TMDB_NEGATIVE_MAPPING_TTL=60,
        TMDB_ENRICHMENT_TTL=3600,
        TMDB_STALE_CACHE_TTL=86400,
        TMDB_HTTP_CACHE_TTL=0,
        TMDB_REQUEST_TIMEOUT=2,
        TMDB_RETRY_TOTAL=2,
        TMDB_RETRY_BACKOFF=0.25,
    )
    app.config.update(overrides)
    init_database(app)
    return app


def _movie(movie_id=1, title="Example Movie (2020)"):
    return {
        "movieId": movie_id,
        "title": title,
        "year": 2020,
        "genres": "Drama",
        "overview": "",
    }


def test_tmdb_mapping_is_persisted_and_catalog_changes_invalidate_it(monkeypatch):
    app = _app()
    calls = []

    def resolve(title, year, language=None, region=None):
        calls.append((title, year))
        return 321

    monkeypatch.setattr(movie_service, "find_tmdb_id_for_movie", resolve)

    with app.app_context():
        movie = _movie()
        assert movie_service.get_tmdb_id_for_movie(movie, allow_remote=True) == 321
        mapping = db.session.get(MovieTmdbMapping, 1)
        assert mapping is not None
        assert mapping.tmdb_id == 321
        assert mapping.status == "resolved"
        assert mapping.matched_by == "title_year"
        assert len(calls) == 1

        def should_not_resolve(*args, **kwargs):
            raise AssertionError("fresh durable mapping should prevent another title search")

        monkeypatch.setattr(movie_service, "find_tmdb_id_for_movie", should_not_resolve)
        assert movie_service.get_tmdb_id_for_movie(movie, allow_remote=True) == 321

        changed = dict(movie)
        changed["title"] = "Different Movie (2020)"
        # Cache-only list rendering must not trust identity data derived from a changed title.
        assert movie_service.get_tmdb_id_for_movie(changed, allow_remote=False) is None


def test_list_enrichment_is_cache_only_and_can_reuse_durable_payload(monkeypatch):
    app = _app()

    with app.app_context():
        movie = _movie()
        key = movie_service._catalog_key(movie)
        now = datetime.now(timezone.utc)
        db.session.add(
            MovieTmdbMapping(
                local_movie_id=1,
                tmdb_id=77,
                catalog_key=key,
                status="resolved",
                matched_by="title_year",
                checked_at=now,
                expires_at=now + timedelta(hours=1),
            )
        )
        db.session.add(
            TmdbEnrichmentCache(
                tmdb_id=77,
                language="en-US",
                region="US",
                payload={
                    "id": 77,
                    "title": "Example Movie",
                    "poster_url": "https://image.example/poster.jpg",
                    "backdrop_url": None,
                    "watch_providers": {"region": "US", "flatrate": [], "rent": [], "buy": []},
                },
                fetched_at=now,
                expires_at=now + timedelta(hours=1),
            )
        )
        db.session.commit()

        def no_remote(*args, **kwargs):
            raise AssertionError("list enrichment must not make remote TMDb calls")

        monkeypatch.setattr(movie_service, "find_tmdb_id_for_movie", no_remote)
        monkeypatch.setattr(tmdb_service, "_make_request", no_remote)

        enriched = movie_service.enrich_movies_list([movie])
        assert enriched[0]["tmdb_id"] == 77
        assert enriched[0]["tmdb_poster_url"] == "https://image.example/poster.jpg"

        uncached = _movie(movie_id=2, title="Uncached Movie (2020)")
        result = movie_service.enrich_movies_list([uncached])
        assert result[0]["tmdb_id"] is None
        assert result[0]["tmdb_poster_url"] == "/static/img/movie-placeholder.jpg"


def test_movie_details_persist_and_serve_stale_when_refresh_fails(monkeypatch):
    app = _app(TMDB_WATCH_REGION="US")
    calls = []

    def fake_request(endpoint, params=None):
        calls.append((endpoint, dict(params or {})))
        if endpoint == "/movie/10":
            return {
                "id": 10,
                "title": "Cached Movie",
                "original_title": "Cached Movie",
                "overview": "Overview",
                "release_date": "2020-01-01",
                "runtime": 100,
                "genres": [{"id": 18, "name": "Drama"}],
                "vote_average": 7.5,
                "vote_count": 100,
                "popularity": 12.0,
                "poster_path": "/poster.jpg",
                "backdrop_path": "/backdrop.jpg",
                "production_companies": [],
                "production_countries": [],
                "credits": {"cast": [], "crew": []},
                "videos": {"results": []},
                "keywords": {"keywords": []},
            }
        if endpoint == "/movie/10/watch/providers":
            return {
                "results": {
                    "US": {
                        "link": "https://www.themoviedb.org/movie/10/watch",
                        "flatrate": [{"provider_id": 1, "provider_name": "Example Stream"}],
                    }
                }
            }
        raise AssertionError(endpoint)

    monkeypatch.setattr(tmdb_service, "_make_request", fake_request)

    with app.app_context():
        details = tmdb_service.get_movie_details(10, force_refresh=True)
        assert details["poster_url"].endswith("/poster.jpg")
        assert details["watch_providers"]["region"] == "US"
        assert details["watch_providers"]["flatrate"][0]["provider_name"] == "Example Stream"
        assert len(calls) == 2
        assert calls[0][1]["append_to_response"] == "keywords,videos,credits"

        row = db.session.execute(
            db.select(TmdbEnrichmentCache).where(TmdbEnrichmentCache.tmdb_id == 10)
        ).scalar_one()
        assert row.payload["title"] == "Cached Movie"

        def should_not_request(*args, **kwargs):
            raise AssertionError("fresh durable cache should prevent HTTP calls")

        monkeypatch.setattr(tmdb_service, "_make_request", should_not_request)
        cached = tmdb_service.get_movie_details(10)
        assert cached["title"] == "Cached Movie"

        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        row.fetched_at = datetime.now(timezone.utc) - timedelta(hours=1)
        db.session.commit()
        monkeypatch.setattr(tmdb_service, "_make_request", lambda *args, **kwargs: None)
        stale = tmdb_service.get_movie_details(10, allow_remote=True, allow_stale=True)
        assert stale["title"] == "Cached Movie"


def test_retry_session_honors_configured_429_backoff():
    app = _app(TMDB_RETRY_TOTAL=3, TMDB_RETRY_BACKOFF=0.75)
    tmdb_service._SESSION = None
    tmdb_service._SESSION_SIGNATURE = None

    with app.app_context():
        session = tmdb_service._get_session()
        retry = session.get_adapter("https://").max_retries
        assert retry.total == 3
        assert retry.backoff_factor == 0.75
        assert 429 in retry.status_forcelist
        assert retry.respect_retry_after_header is True
        assert "GET" in retry.allowed_methods


def test_title_year_matching_does_not_blindly_take_first_result(monkeypatch):
    monkeypatch.setattr(
        tmdb_service,
        "search_movie_by_title",
        lambda *args, **kwargs: [
            {
                "id": 1,
                "title": "Different Movie",
                "original_title": "Different Movie",
                "release_date": "2020-01-01",
                "vote_count": 9999,
            },
            {
                "id": 2,
                "title": "Exact Movie",
                "original_title": "Exact Movie",
                "release_date": "2020-05-01",
                "vote_count": 20,
            },
        ],
    )
    assert tmdb_service.find_tmdb_id_for_movie("Exact Movie", 2020) == 2
    assert tmdb_service.find_tmdb_id_for_movie("Missing Movie", 2020) is None


def test_watch_provider_region_comes_from_configuration(monkeypatch):
    app = _app(TMDB_WATCH_REGION="US")
    monkeypatch.setattr(
        tmdb_service,
        "_make_request",
        lambda *args, **kwargs: {
            "results": {
                "IN": {"flatrate": [{"provider_name": "India Service"}]},
                "US": {
                    "link": "https://www.themoviedb.org/watch/us",
                    "flatrate": [{"provider_name": "US Service"}],
                },
            }
        },
    )
    with app.app_context():
        providers = tmdb_service.get_watch_providers(10)
        assert providers["region"] == "US"
        assert providers["flatrate"][0]["provider_name"] == "US Service"
