"""FastAPI dashboard router — mount in your app to get benchmark visualization.

Usage:
    from llm_bench.dashboard import create_benchmark_router
    router = create_benchmark_router(storage=my_storage, tasks=my_tasks)
    app.include_router(router, prefix="/benchmarks")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_bench.analyze import aggregate, pareto_frontier, recommend
from llm_bench.configs import ALL_CONFIGS
from llm_bench.types import TaskDefinition

_STATIC_DIR = Path(__file__).parent / "static"


def create_benchmark_router(
    storage: Any,
    tasks: list[TaskDefinition],
    configs: list[str] | None = None,
    cycle_time_cost: float = 0.002,
) -> Any:
    """Create a FastAPI router with benchmark dashboard endpoints.

    Args:
        storage: Storage backend instance.
        tasks: List of task definitions.
        configs: Available config names for override dropdowns. None = all built-in.
        cycle_time_cost: USD opportunity cost for cost-per-success calculation.

    Returns:
        FastAPI APIRouter instance.
    """
    try:
        from fastapi import APIRouter, Request
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError:
        raise ImportError(
            "Dashboard requires FastAPI. Install with: pip install llm-bench[dashboard]"
        )

    router = APIRouter()
    task_floors = {t.name: t.quality_floor for t in tasks}
    available_configs = configs or [c.name for c in ALL_CONFIGS]

    @router.get("/")
    async def dashboard_page():
        return FileResponse(_STATIC_DIR / "benchmark.html")

    @router.get("/api/results")
    async def get_results():
        results = storage.load_results()
        agg = aggregate(results) if results else {}

        # Build response in the format the dashboard expects
        formatted_results: dict[str, list[dict]] = {}
        recommendations: dict[str, str] = {}
        total_runs = 0

        for task_name, configs_data in agg.items():
            frontier = pareto_frontier(configs_data)
            floor = task_floors.get(task_name, 0.5)
            rec_name, _ = recommend(task_name, configs_data, floor, cycle_time_cost)
            if rec_name:
                recommendations[task_name] = rec_name

            rows = []
            for config_name, stats in sorted(
                configs_data.items(),
                key=lambda x: (-x[1]["mean_quality"], x[1]["mean_cost"]),
            ):
                total_runs += stats["n_runs"]
                rows.append({
                    "config": config_name,
                    "quality_mean": stats["mean_quality"],
                    "quality_std": stats["std_quality"],
                    "cost_mean": stats["mean_cost"],
                    "latency_mean": stats["mean_latency"],
                    "n": stats["n_runs"],
                    "errors": stats["n_errors"],
                    "rate_limited": stats["n_rate_limited"],
                    "sub_scores": stats.get("sub_scores", {}),
                    "pareto": config_name in frontier,
                })
            formatted_results[task_name] = rows

        # Load overrides
        overrides = {}
        if hasattr(storage, "get_all_overrides"):
            overrides = storage.get_all_overrides()

        return JSONResponse({
            "total_runs": total_runs,
            "results": formatted_results,
            "recommendations": recommendations,
            "overrides": overrides,
            "available_configs": available_configs,
        })

    @router.post("/api/overrides")
    async def set_override(request: Request):
        body = await request.json()
        task_name = body.get("task_name")
        config_name = body.get("config_name")
        if task_name:
            storage.set_override(task_name, config_name)
        return JSONResponse({"ok": True})

    return router
