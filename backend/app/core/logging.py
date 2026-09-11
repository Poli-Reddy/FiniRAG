"""
FiniRAG Logging Module.

Structured logging with data protection policy (Correction #10):

Logs MAY contain:
  - Document ID, filename, document type
  - Processing stage, status, error codes
  - Provider name, page count, character count, field count
  - Processing latency, validation summary counts

Logs MUST NOT contain:
  - API keys or credentials
  - Full OCR text output
  - Full LLM prompt or response text
  - Extracted financial data (amounts, account numbers)
  - Customer/vendor names, addresses, tax IDs
  - Raw OCR/LLM provider responses (except at DEBUG level)
"""

import logging
import sys

from backend.app.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("finirag")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Prevent duplicate handlers on reload
    if logger.handlers:
        return logger

    # Console handler with structured format. Context passed via
    # extra={...} is appended so document/provider/page info is kept.
    class _ContextFormatter(logging.Formatter):
        _SKIP = {"asctime", "levelname", "name", "message", "args", "exc_info",
                 "exc_text", "stack_info", "lineno", "pathname", "filename",
                 "module", "funcName", "created", "msecs", "relativeCreated",
                 "thread", "threadName", "process", "processName", "msg"}

        def format(self, record: logging.LogRecord) -> str:
            base = super().format(record)
            ctx = {k: v for k, v in record.__dict__.items() if k not in self._SKIP}
            if ctx:
                parts = " ".join(f"{k}={v}" for k, v in sorted(ctx.items()))
                return f"{base} | {parts}"
            return base

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = _ContextFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


# Module-level logger
logger = setup_logging()
