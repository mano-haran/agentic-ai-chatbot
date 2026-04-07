"""
Application logger.

Provides a singleton `logger` object with severity-based methods:
    logger.debug(category, msg, **data)
    logger.info(category, msg, **data)
    logger.warning(category, msg, **data)
    logger.error(category, msg, exc=None, **data)

Controlled by LOG_LEVEL in .env:
    OFF      — no output (default)
    ERROR    — errors only
    WARNING  — warnings and errors
    INFO     — info, warnings, errors
    DEBUG    — everything

All output goes to LOG_FILE (default: app.log).

Usage:
    from framework.core.log import logger

    logger.debug("ROUTING", "pattern scan", matches=["jenkins_log_analysis"])
    logger.info("WORKFLOW", "selected", workflow="jenkins_log_analysis")
    logger.warning("AGENT", "no response produced", agent="log_fetcher_agent")
    logger.error("ERROR", "stream_steps raised", exc=e, workflow="jenkins_log_analysis")
"""

import logging
import traceback
from pathlib import Path

import config

# Map string level names to Python logging integer levels.
_LEVEL_MAP: dict[str, int] = {
    "DEBUG":   logging.DEBUG,
    "INFO":    logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR":   logging.ERROR,
    "OFF":     logging.CRITICAL + 1,   # effectively disabled
}

_LOG_FORMAT = "%(asctime)s.%(msecs)03d  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class AppLogger:
    """
    Severity-based logger for the agentic framework.

    Each method accepts:
        category (str)  — short uppercase label, e.g. ROUTING, AGENT, WORKFLOW
        message  (str)  — human-readable description of what happened
        **data          — optional key=value pairs appended after the message;
                          values longer than 400 chars are truncated

    logger.error() additionally accepts:
        exc (Exception) — when supplied, the full traceback is appended
    """

    def __init__(self) -> None:
        self._log = logging.getLogger("agentfw")
        self._initialized = False

    # ── Initialisation (lazy, first use) ─────────────────────────────────────

    def _init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        level_name = getattr(config, "LOG_LEVEL", "OFF").upper()
        level = _LEVEL_MAP.get(level_name, logging.CRITICAL + 1)

        if level > logging.CRITICAL:
            self._log.addHandler(logging.NullHandler())
            return

        log_path = Path(getattr(config, "LOG_FILE", "app.log"))
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        self._log.addHandler(handler)
        self._log.setLevel(level)
        print(f"[logger] level={level_name} → {log_path.resolve()}", flush=True)

    # ── Formatting helper ─────────────────────────────────────────────────────

    def _format(self, severity: str, category: str, message: str, data: dict) -> str:
        extras = []
        for k, v in data.items():
            sv = str(v)
            if len(sv) > 400:
                sv = sv[:397] + "…"
            extras.append(f"{k}={sv!r}")
        line = f"[{severity:<7}] [{category:<9}] {message}"
        if extras:
            line += "  |  " + "  ".join(extras)
        return line

    # ── Public severity methods ───────────────────────────────────────────────

    def debug(self, category: str, msg: str, **data) -> None:
        """Log a DEBUG-level entry. Visible when LOG_LEVEL=DEBUG."""
        self._init()
        if self._log.isEnabledFor(logging.DEBUG):
            self._log.debug(self._format("DEBUG", category, msg, data))

    def info(self, category: str, msg: str, **data) -> None:
        """Log an INFO-level entry. Visible when LOG_LEVEL=INFO or DEBUG."""
        self._init()
        if self._log.isEnabledFor(logging.INFO):
            self._log.info(self._format("INFO", category, msg, data))

    def warning(self, category: str, msg: str, **data) -> None:
        """Log a WARNING-level entry. Visible when LOG_LEVEL=WARNING, INFO, or DEBUG."""
        self._init()
        if self._log.isEnabledFor(logging.WARNING):
            self._log.warning(self._format("WARNING", category, msg, data))

    def error(self, category: str, msg: str, exc: Exception | None = None, **data) -> None:
        """
        Log an ERROR-level entry. Visible when LOG_LEVEL=ERROR, WARNING, INFO, or DEBUG.

        Args:
            category: Short uppercase label (e.g. ERROR, WORKFLOW, AGENT).
            msg:      Description of what went wrong.
            exc:      Optional exception — when supplied, the full traceback
                      is appended to the log entry.
            **data:   Additional key=value context. Any key name is allowed,
                      including 'message', since it no longer conflicts with
                      a parameter name.
        """
        self._init()
        if not self._log.isEnabledFor(logging.ERROR):
            return
        line = self._format("ERROR", category, msg, data)
        if exc is not None:
            tb = traceback.format_exc()
            line += f"\n  exception={exc!r}\n{tb}"
        self._log.error(line)


# ── Singleton exported to the rest of the codebase ───────────────────────────
logger = AppLogger()
