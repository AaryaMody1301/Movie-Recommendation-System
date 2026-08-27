"""Application logging configuration helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record for production log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    """Configure the root logger once using text or structured JSON output."""
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    formatter: logging.Formatter
    if str(log_format).lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    root = logging.getLogger()
    root.setLevel(resolved_level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)
        return

    for handler in root.handlers:
        handler.setFormatter(formatter)
        handler.setLevel(resolved_level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a standard library logger after application configuration."""
    return logging.getLogger(name)
