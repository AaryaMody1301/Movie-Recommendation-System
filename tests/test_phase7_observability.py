"""Regression tests for Phase 7 health, startup, and logging behavior."""

import json
import logging

import app as app_module
from observability import JsonFormatter


def _write_catalog(tmp_path):
    path = tmp_path / "movies.csv"
    path.write_text(
        "movieId,title,genres\n"
        "1,Example Movie (2020),Drama|Comedy\n"
        "2,Second Movie (2021),Drama\n",
        encoding="utf-8",
    )
    return path


def _create_app(tmp_path, **overrides):
    config = {
        "TESTING": True,
        "SECRET_KEY": "phase7-test-secret",
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "MOVIES_CSV": str(_write_catalog(tmp_path)),
        "RATINGS_CSV": "",
        "WTF_CSRF_ENABLED": False,
        "RECOMMENDER_ENABLED": False,
        "APP_VERSION": "phase7-test",
        "LOG_FORMAT": "text",
    }
    config.update(overrides)
    return app_module.create_app(test_config=config)


def test_liveness_and_readiness_report_critical_state(tmp_path):
    app = _create_app(tmp_path)
    client = app.test_client()

    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.get_json() == {
        "status": "ok",
        "service": "movie-recommendation-system",
        "version": "phase7-test",
    }

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    payload = ready.get_json()
    assert payload["status"] == "ready"
    assert payload["checks"]["database"]["status"] == "ok"
    assert payload["checks"]["catalog"] == {"status": "ok", "movies": 2}
    assert payload["checks"]["recommender"]["status"] == "disabled"


def test_readiness_fails_when_catalog_is_unavailable(tmp_path):
    app = _create_app(tmp_path)
    app.data_loader = None

    response = app.test_client().get("/health/ready")
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["catalog"]["status"] == "error"


def test_disabled_recommender_never_constructs_transformer(monkeypatch, tmp_path):
    class FailIfConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("recommender should be skipped when disabled")

    monkeypatch.setattr(app_module, "ContentBasedRecommender", FailIfConstructed)
    app = _create_app(tmp_path, RECOMMENDER_ENABLED=False)

    assert app.data_loader is not None
    assert app.recommender is None


def test_json_formatter_emits_machine_readable_record():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="phase7.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request_complete status=%s",
        args=(200,),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "phase7.test"
    assert payload["message"] == "request_complete status=200"
    assert payload["timestamp"].endswith("+00:00")
