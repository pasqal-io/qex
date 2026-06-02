"""Centralised logging configuration for QEX.

QEX uses ``loguru`` for human-readable event logs. By default these are quiet
(``WARNING`` and above) so a normal run is not flooded with per-system INFO
chatter; the detailed INFO/DEBUG messages (grid sizes, parser steps, per-method
notes) are only shown when you raise the level.

Two ways to control verbosity, from coarse to fine:

  - ``debug: true``           -> shortcut for ``level: DEBUG`` (verbose format).
  - ``logging.level: TRACE``  -> set ANY loguru level explicitly. This is the
    global knob: ``TRACE`` < ``DEBUG`` < ``INFO`` < ``SUCCESS`` < ``WARNING`` <
    ``ERROR`` < ``CRITICAL``. ``TRACE`` prints absolutely everything.

An explicit ``level`` always wins over the ``debug`` shortcut. Either can be set
via config (consumed by ``run_experiment`` / the CLI / the example) or in code
(``configure_logging(level="TRACE")``).

This keeps a tqdm progress bar (training, data generation) as the primary
on-screen signal, with verbose logs reserved for when you actually want them.
"""

from __future__ import annotations

import sys

from loguru import logger

# tqdm-friendly sink: writing through tqdm.write keeps log lines from corrupting
# an active progress bar.
try:
    from tqdm import tqdm

    def _sink(message: str) -> None:
        tqdm.write(message, end="")

except Exception:  # pragma: no cover - tqdm always present, but stay safe
    def _sink(message: str) -> None:
        sys.stderr.write(message)


# Verbose, source-annotated format used whenever the level is below WARNING
# (i.e. you asked to see the INFO/DEBUG/TRACE pipeline chatter).
_VERBOSE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
# Compact format for the quiet (WARNING+) default: just the level and message.
_QUIET_FORMAT = "<level>{level}</level>: {message}"

# Loguru's standard severities, low -> high. Used to validate an explicit level
# and to decide which format (verbose vs quiet) to use.
_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")

_configured = False
_current_level: str = "WARNING"


def configure_logging(
    debug: bool = False,
    *,
    level: str | None = None,
    force: bool = False,
) -> None:
    """Configure the loguru sink and global level for QEX.

    Args:
        debug: shortcut — if True (and ``level`` is not given), show ``DEBUG``+.
        level: explicit global level, e.g. ``"TRACE"``/``"DEBUG"``/``"INFO"``/
            ``"WARNING"``. Case-insensitive. Overrides ``debug`` when given;
            ``"TRACE"`` prints everything. Defaults to ``WARNING`` (quiet).
        force: re-apply even if logging was already configured this session.

    Raises:
        ValueError: if ``level`` is not a recognised loguru level.
    """
    global _configured, _current_level

    if level is not None:
        resolved = str(level).strip().upper()
        if resolved not in _LEVELS:
            raise ValueError(
                f"Unknown logging level {level!r}; choose one of {_LEVELS}."
            )
    else:
        resolved = "DEBUG" if debug else "WARNING"

    # Re-apply if forced OR the requested level changed (so bumping the level
    # mid-session actually takes effect without needing force=True).
    if _configured and not force and resolved == _current_level:
        return

    logger.remove()  # drop any existing handlers (incl. loguru's default)
    # Use the verbose format whenever we're showing sub-WARNING chatter.
    fmt = _VERBOSE_FORMAT if _LEVELS.index(resolved) < _LEVELS.index("WARNING") else _QUIET_FORMAT
    logger.add(_sink, level=resolved, format=fmt, colorize=True)
    _configured = True
    _current_level = resolved


def set_level(level: str) -> None:
    """Set the global log level explicitly (e.g. ``set_level("TRACE")``)."""
    configure_logging(level=level, force=True)


def set_debug(debug: bool) -> None:
    """Convenience: (re)configure logging to debug or quiet."""
    configure_logging(debug=debug, force=True)
