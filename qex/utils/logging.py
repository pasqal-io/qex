"""Centralised logging configuration for QEX.

QEX uses ``loguru`` for human-readable event logs. By default these are quiet
(``WARNING`` and above) so a normal run is not flooded with per-system INFO
chatter; the detailed INFO/DEBUG messages (grid sizes, parser steps, per-method
notes) are only shown when **debug mode** is on.

Turn on debug logging either:
  - via config: ``debug: true`` (consumed by ``run_experiment`` / the CLI), or
  - in code: ``configure_logging(debug=True)``.

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


_DEBUG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
_QUIET_FORMAT = "<level>{level}</level>: {message}"

_configured = False


def configure_logging(debug: bool = False, *, force: bool = False) -> None:
    """Configure the loguru sink and level for QEX.

    Args:
        debug: if True, show DEBUG+ with a detailed format (the verbose data
            generation / setup logs become visible); otherwise only WARNING+.
        force: re-apply even if logging was already configured this session.
    """
    global _configured
    if _configured and not force:
        return

    logger.remove()  # drop loguru's default stderr handler
    if debug:
        logger.add(_sink, level="DEBUG", format=_DEBUG_FORMAT, colorize=True)
    else:
        logger.add(_sink, level="WARNING", format=_QUIET_FORMAT, colorize=True)
    _configured = True


def set_debug(debug: bool) -> None:
    """Convenience: (re)configure logging to debug or quiet."""
    configure_logging(debug=debug, force=True)
