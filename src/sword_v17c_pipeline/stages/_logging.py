"""Shared logging for v17c pipeline stages."""

from datetime import datetime as dt


def log(msg: str) -> None:
    """Log message with timestamp."""
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)
