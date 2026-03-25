"""Benchmark runner — executes benchmarks across models and scores results."""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_bench.client import Client
from llm_bench.configs import CONFIG_BY_NAME, ModelConfig, ModelRegistry
from llm_bench.types import BenchmarkResult, Fixture, TaskDefinition

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks: sends fixtures to models via OpenRouter, scores responses.

    Args:
        client: OpenRouter client instance.
        tasks: List of task definitions (each with scorer + fixtures).
        registry: Model registry. Defaults to built-in configs.
        storage: Optional storage backend to auto-persist results.
    """

    def __init__(
        self,
        client: Client,
        tasks: list[TaskDefinition],
        registry: ModelRegistry | None = None,
        storage: Any = None,
    ) -> None:
        self._client = client
        self._tasks = {t.name: t for t in tasks}
        self._registry = registry or ModelRegistry()
        self._storage = storage

    def run_one(
        self,
        task_name: str,
        config: ModelConfig | str,
        fixture: Fixture | None = None,
        run_index: int = 0,
        **scorer_kwargs: Any,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            task_name: Name of the task to benchmark.
            config: ModelConfig or config name string.
            fixture: Specific fixture to use. If None, uses the first fixture for the task.
            run_index: Index for this run (for multi-run benchmarks).
            **scorer_kwargs: Extra kwargs passed to the scorer's score() method.

        Returns:
            BenchmarkResult with quality score, cost, and latency.
        """
        task = self._tasks.get(task_name)
        if not task:
            raise ValueError(f"Unknown task: {task_name!r}. Available: {list(self._tasks.keys())}")

        if isinstance(config, str):
            mc = self._registry.get(config)
            if not mc:
                raise ValueError(f"Unknown config: {config!r}")
            config = mc

        if fixture is None:
            if not task.fixtures:
                raise ValueError(f"Task {task_name!r} has no fixtures")
            fixture = task.fixtures[0]

        # Make the LLM call
        error: str | None = None
        raw_response = ""
        input_tokens = output_tokens = 0
        cost_usd = 0.0

        t0 = time.monotonic()
        try:
            response = self._client.call(
                model=config.model,
                system=fixture.system_prompt,
                user_message=fixture.user_prompt,
                thinking_budget=config.thinking_budget,
            )
            raw_response = response.text
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            cost_usd = response.cost_usd
        except Exception as e:
            error = str(e)
            logger.warning(
                "API call failed [%s/%s/%s]: %s",
                task_name, config.name, fixture.fixture_id, e,
            )
        latency = time.monotonic() - t0

        # Score the response
        sub_scores: dict = {}
        quality_score = 0.0

        if error:
            sub_scores["error"] = error
        else:
            try:
                scored = task.scorer.score(raw_response, fixture, **scorer_kwargs)
                quality_score = scored.pop("score", 0.0)
                sub_scores = scored
                if scored.get("error"):
                    error = scored["error"]
            except Exception as e:
                error = f"Scoring failed: {e}"
                logger.warning(
                    "Scoring failed [%s/%s/%s]: %s",
                    task_name, config.name, fixture.fixture_id, e,
                )

        logger.info(
            "Benchmark [%s/%s/%s run%d]: quality=%.3f cost=$%.6f latency=%.1fs",
            task_name, config.name, fixture.fixture_id, run_index,
            quality_score, cost_usd, latency,
        )

        result = BenchmarkResult(
            task_name=task_name,
            config_name=config.name,
            fixture_id=fixture.fixture_id,
            quality_score=quality_score,
            sub_scores=sub_scores,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_s=latency,
            raw_response=raw_response[:2000],
            error=error,
            run_index=run_index,
        )

        if self._storage:
            try:
                self._storage.save_result(result)
            except Exception as e:
                logger.error("Failed to persist result: %s", e)

        return result

    def run_task(
        self,
        task_name: str,
        n_runs: int = 1,
        configs: list[str] | None = None,
        **scorer_kwargs: Any,
    ) -> list[BenchmarkResult]:
        """Run all (config, fixture) combinations for a task.

        Args:
            task_name: Name of the task.
            n_runs: Number of runs per (config, fixture) pair.
            configs: Subset of config names to test. None = use task's models or all.
            **scorer_kwargs: Extra kwargs for the scorer.

        Returns:
            List of BenchmarkResults.
        """
        task = self._tasks.get(task_name)
        if not task:
            raise ValueError(f"Unknown task: {task_name!r}")

        # Determine which configs to test
        if configs:
            model_configs = self._registry.subset(configs)
        elif task.models:
            model_configs = self._registry.subset(task.models)
        else:
            model_configs = self._registry.all_configs

        results: list[BenchmarkResult] = []
        for config in model_configs:
            for fixture in task.fixtures:
                for run_idx in range(n_runs):
                    result = self.run_one(
                        task_name, config, fixture, run_idx, **scorer_kwargs
                    )
                    results.append(result)

        return results

    def run_all(self, n_runs: int = 1, **scorer_kwargs: Any) -> list[BenchmarkResult]:
        """Run benchmarks for all tasks.

        Args:
            n_runs: Number of runs per (config, fixture) pair.
            **scorer_kwargs: Extra kwargs for all scorers.

        Returns:
            List of all BenchmarkResults across all tasks.
        """
        results: list[BenchmarkResult] = []
        for task_name in self._tasks:
            results.extend(self.run_task(task_name, n_runs, **scorer_kwargs))
        return results

    @property
    def task_names(self) -> list[str]:
        return list(self._tasks.keys())
