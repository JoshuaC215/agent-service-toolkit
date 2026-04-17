import json
import logging
import os


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """
    Initialize unified logging for the application.

    - Reads LOG_LEVEL (default: INFO) and LOG_FORMAT (text|json, default: text) from environment.
    - Configures a single StreamHandler on the root logger if none exist.
    - Applies the chosen formatter to all existing StreamHandlers to avoid duplicates.
    """
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    format_type = (fmt or os.getenv("LOG_FORMAT", "text")).lower()

    root = logging.getLogger()
    root.setLevel(log_level)

    # Create formatter
    if format_type == "json":
        formatter = _JsonFormatter()
    else:
        # Human-readable default text formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    # Ensure at least one StreamHandler exists, without duplicating
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Apply formatter to all StreamHandlers to keep consistency
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)
            handler.setFormatter(formatter)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter to avoid extra dependencies."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Attach exception info if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
