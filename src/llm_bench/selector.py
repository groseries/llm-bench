"""Auto-selector — connects benchmark results to production model routing.

Reads stored benchmark data, computes Pareto-optimal recommendations,
and exposes get_best_config() for production routing decisions.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_bench.analyze import aggregate, get_recommendations, recommend
from llm_bench.configs import CONFIG_BY_NAME, ModelConfig, ModelRegistry
from llm_bench.types import TaskDefinition

logger = logging.getLogger(__name__)


class AutoSelector:
    """Benchmark-driven model selector.

    Periodically call refresh() to update recommendations from stored benchmark data.
    Call get_best_config() in your production code to get the optimal model for a task.

    Args:
        storage: Storage backend with benchmark results.
        tasks: Task definitions (provides quality floors).
        registry: Model registry. Defaults to built-in configs.
        cycle_time_cost: USD opportunity cost of a wasted LLM cycle.
    """

    def __init__(
        self,
        storage: Any,
        tasks: list[TaskDefinition],
        registry: ModelRegistry | None = None,
        cycle_time_cost: float = 0.002,
    ) -> None:
        self._storage = storage
        self._tasks = {t.name: t for t in tasks}
        self._registry = registry or ModelRegistry()
        self._cycle_time_cost = cycle_time_cost

        # Cache: {task_name: config_name}
        self._recommendations: dict[str, str] = {}
        # Extended cache: {task_name: ModelConfig}
        self._best_configs: dict[str, ModelConfig] = {}

    def refresh(self) -> dict[str, str]:
        """Re-analyze stored results and update recommendations.

        Call this periodically (e.g., at the start of each cycle).

        Returns:
            {task_name: config_name} for all tasks with qualifying recommendations.
        """
        results = self._storage.load_results()
        if not results:
            logger.debug("No benchmark results available for auto-selection")
            return self._recommendations

        quality_floors = {name: t.quality_floor for name, t in self._tasks.items()}

        new_recs = get_recommendations(
            results,
            quality_floors=quality_floors,
            cycle_time_cost=self._cycle_time_cost,
        )

        # Apply manual overrides
        for task_name in self._tasks:
            override = self._storage.get_override(task_name)
            if override:
                new_recs[task_name] = override

        # Resolve config names to ModelConfig objects
        new_best: dict[str, ModelConfig] = {}
        for task_name, config_name in new_recs.items():
            mc = self._registry.get(config_name)
            if mc:
                new_best[task_name] = mc

        if new_recs != self._recommendations:
            logger.info("Model recommendations updated: %s", new_recs)

        self._recommendations = new_recs
        self._best_configs = new_best
        return self._recommendations

    def get_best_config(self, task_name: str) -> ModelConfig | None:
        """Get the current best ModelConfig for a task.

        Returns None if no recommendation exists (call refresh() first).
        """
        return self._best_configs.get(task_name)

    def get_best_model(self, task_name: str) -> str | None:
        """Get the current best model slug for a task.

        Convenience method — returns the OpenRouter model slug directly.
        """
        mc = self.get_best_config(task_name)
        return mc.model if mc else None

    @property
    def recommendations(self) -> dict[str, str]:
        """Current recommendations: {task_name: config_name}."""
        return dict(self._recommendations)

    def get_stats(self) -> dict[str, dict[str, dict]]:
        """Get aggregated stats for all tasks from stored results.

        Returns the same structure as analyze.aggregate().
        """
        results = self._storage.load_results()
        if not results:
            return {}
        return aggregate(results)
