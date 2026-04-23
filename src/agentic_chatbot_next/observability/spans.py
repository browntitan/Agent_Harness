from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, Iterator


@contextmanager
def timed_span(name: str, *, metadata: Dict[str, Any] | None = None) -> Iterator[Dict[str, Any]]:
    start = perf_counter()
    span = {"name": name, "metadata": dict(metadata or {})}
    try:
        yield span
    finally:
        span["elapsed_seconds"] = round(perf_counter() - start, 6)
