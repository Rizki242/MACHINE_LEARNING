"""Penyiapan logging seragam untuk seluruh modul FurnaceGuard AI."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Pasang handler stdout sekali saja.

    Dipanggil ulang tanpa efek samping, sehingga aman dipakai di skrip
    maupun notebook.
    """
    global _CONFIGURED
    if _CONFIGURED:
        if level:
            logging.getLogger("furnaceguard").setLevel(level.upper())
        return

    # Konsol Windows default ke cp1252 dan merusak karakter seperti "§"
    # serta nama peralatan berhuruf non-ASCII.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass

    if level is None:
        # Impor di dalam fungsi agar tidak melingkar dengan core.config
        from backend.app.core.config import get_settings

        level = get_settings().log_level

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger("furnaceguard")
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Ambil logger di bawah namespace ``furnaceguard``."""
    setup_logging()
    suffix = name.split("backend.app.")[-1]
    return logging.getLogger(f"furnaceguard.{suffix}")
