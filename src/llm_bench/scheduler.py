"""Continuous benchmark runner — periodically re-tests models in the background."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable

from llm_bench.runner import BenchmarkRunner

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 60 * 60  # 1 hour


class ContinuousRunner:
    """Background thread that runs one random benchmark every interval.

    Args:
        runner: BenchmarkRunner instance.
        interval_seconds: Seconds between benchmark runs.
        excluded_configs: Config names to skip (e.g., expensive models).
        should_pause: Callback returning True to pause benchmarking
                      (e.g., when spend limit is reached).
        warmup_seconds: Initial delay before first benchmark run.
    """

    def __init__(
        self,
        runner: BenchmarkRunner,
        interval_seconds: int = DEFAULT_INTERVAL,
        excluded_configs: set[str] | None = None,
        should_pause: Callable[[], bool] | None = None,
        warmup_seconds: int = 30,
    ) -> None:
        self._runner = runner
        self._interval = interval_seconds
        self._excluded = excluded_configs or set()
        self._should_pause = should_pause
        self._warmup = warmup_seconds
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background benchmark thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, args=(self._stop_event,), daemon=True
        )
        self._thread.start()
        logger.info(
            "Continuous benchmark runner started (interval=%ds, excluded=%s)",
            self._interval,
            self._excluded,
        )

    def stop(self) -> None:
        """Stop the background benchmark thread."""
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._stop_event = None
        self._thread = None
        logger.info("Continuous benchmark runner stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self, stop_event: threading.Event) -> None:
        # Warmup delay
        for _ in range(self._warmup):
            if stop_event.is_set():
                return
            time.sleep(1)

        logger.info("Continuous benchmark loop started")

        while not stop_event.is_set():
            if self._should_pause and self._should_pause():
                logger.info("Continuous benchmarks paused")
            else:
                try:
                    self._run_one_random()
                except Exception as e:
                    logger.error("Continuous benchmark error: %s", e)

            # Wait for next interval
            for _ in range(self._interval):
                if stop_event.is_set():
                    return
                time.sleep(1)

    def _run_one_random(self) -> None:
        """Run one random (task, config, fixture) benchmark."""
        task_names = self._runner.task_names
        if not task_names:
            return

        task_name = random.choice(task_names)

        # Get available configs for this task, excluding expensive ones
        from llm_bench.configs import ModelRegistry

        registry = self._runner._registry
        configs = [
            c for c in registry.all_configs
            if c.name not in self._excluded
        ]
        if not configs:
            return
        config = random.choice(configs)

        # Get a random fixture
        task = self._runner._tasks.get(task_name)
        if not task or not task.fixtures:
            return
        fixture = random.choice(task.fixtures)

        result = self._runner.run_one(task_name, config, fixture)
        logger.info(
            "Continuous benchmark: %s/%s quality=%.3f cost=$%.6f",
            result.task_name,
            result.config_name,
            result.quality_score,
            result.cost_usd,
        )
