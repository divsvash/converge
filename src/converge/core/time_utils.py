"""Tiny shared utility -- avoids pipeline.py and execution.py needing to import from each other."""
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
