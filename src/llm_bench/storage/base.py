"""Storage protocol for benchmark results."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from llm_bench.types import BenchmarkResult


class Storage(Protocol):
    """Persist and retrieve benchmark results.

    Implementations must support save_result() and load_results().
    Override settings are optional (used by dashboard model override controls).
    """

    def save_result(self, result: BenchmarkResult) -> None:
        """Persist a single benchmark result."""
        ...

    def save_results(self, results: list[BenchmarkResult]) -> None:
        """Persist multiple benchmark results."""
        ...

    def load_results(
        self,
        task_name: str | None = None,
        since: datetime | None = None,
    ) -> list[dict]:
        """Load results as dicts. Optionally filter by task and/or time range."""
        ...

    def get_override(self, task_name: str) -> str | None:
        """Get the manual model override for a task, if any."""
        ...

    def set_override(self, task_name: str, config_name: str | None) -> None:
        """Set or clear a manual model override for a task."""
        ...
