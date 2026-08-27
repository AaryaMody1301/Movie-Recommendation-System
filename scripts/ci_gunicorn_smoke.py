"""Boot the production WSGI app under Gunicorn and probe operational endpoints."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


BASE_URL = "http://127.0.0.1:8000"


def _get_json(path: str):
    with urlopen(f"{BASE_URL}{path}", timeout=2) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    process = subprocess.Popen(
        [
            "gunicorn",
            "--bind",
            "127.0.0.1:8000",
            "--workers",
            "1",
            "--timeout",
            "60",
            "wsgi:app",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        live = None
        for _ in range(30):
            if process.poll() is not None:
                break
            try:
                live = _get_json("/health/live")
                break
            except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
                time.sleep(1)

        if live is None:
            raise RuntimeError("Gunicorn did not become live")

        ready = _get_json("/health/ready")
        assert live["status"] == "ok"
        assert ready["status"] == "ready"
        assert ready["checks"]["database"]["status"] == "ok"
        assert ready["checks"]["catalog"]["status"] == "ok"
        print(json.dumps({"live": live, "ready": ready}, indent=2))
        return 0
    except Exception as exc:
        print(f"Deployment smoke test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=5)
        if output:
            print(output)


if __name__ == "__main__":
    raise SystemExit(main())
