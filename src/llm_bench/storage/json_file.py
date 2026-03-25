"""JSON file storage backend — simplest possible persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llm_bench.types import BenchmarkResult

_DEFAULT_DIR = Path.home() / ".llm-bench" / "results"


class JSONFileStorage:
    """Stores each benchmark run as a JSON file.

    Args:
        directory: Directory for JSON result files. Defaults to ~/.llm-bench/results/.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = Path(directory) if directory else _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._overrides_path = self._dir / "_overrides.json"

    def save_result(self, result: BenchmarkResult) -> None:
        # Append to a single results file per day
        date_str = result.timestamp[:10] if result.timestamp else "unknown"
        path = self._dir / f"results_{date_str}.json"

        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.append(result.to_dict())
        path.write_text(json.dumps(existing, indent=2))

    def save_results(self, results: list[BenchmarkResult]) -> None:
        for r in results:
            self.save_result(r)

    def load_results(
        self,
        task_name: str | None = None,
        since: datetime | None = None,
    ) -> list[dict]:
        all_results: list[dict] = []

        for path in sorted(self._dir.glob("results_*.json")):
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    all_results.extend(data)
            except (json.JSONDecodeError, OSError):
                continue

        # Filter
        if task_name:
            all_results = [
                r for r in all_results
                if (r.get("task_name") or r.get("call_type")) == task_name
            ]
        if since:
            since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
            all_results = [
                r for r in all_results
                if r.get("timestamp", "") >= since_str
            ]

        return all_results

    def get_override(self, task_name: str) -> str | None:
        overrides = self._load_overrides()
        return overrides.get(task_name)

    def set_override(self, task_name: str, config_name: str | None) -> None:
        overrides = self._load_overrides()
        if config_name:
            overrides[task_name] = config_name
        else:
            overrides.pop(task_name, None)
        self._overrides_path.write_text(json.dumps(overrides, indent=2))

    def get_all_overrides(self) -> dict[str, str]:
        return self._load_overrides()

    def _load_overrides(self) -> dict[str, str]:
        if self._overrides_path.exists():
            try:
                return json.loads(self._overrides_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}
