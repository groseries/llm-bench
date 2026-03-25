"""Analysis engine — aggregate benchmark results and find Pareto-optimal models.

Core functions:
    aggregate()        — group results by (task, config), compute stats
    pareto_frontier()  — find non-dominated configs on quality vs cost
    cost_per_success() — effective cost including time waste from failures
    recommend()        — select cheapest config above quality floor

These functions operate on plain dicts (no DB or storage dependency).
"""

from __future__ import annotations

import statistics
from collections import defaultdict

# Transient errors that don't reflect model quality
_TRANSIENT_ERRORS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "spending cap", "quota")


def _is_transient_error(r: dict) -> bool:
    err = str(r.get("error", ""))
    return any(t in err for t in _TRANSIENT_ERRORS)


def aggregate(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Group results by (task_name, config_name) and compute stats.

    Args:
        results: List of dicts with keys: task_name (or call_type), config_name,
                 quality_score, cost_usd, latency_s, error.

    Returns:
        Nested dict: {task_name: {config_name: {mean_quality, std_quality,
        mean_cost, mean_latency, n_runs, n_errors, n_rate_limited, error_rate}}}
    """
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        task = r.get("task_name") or r.get("call_type", "unknown")
        grouped[task][r["config_name"]].append(r)

    agg = {}
    for task_name, configs in grouped.items():
        agg[task_name] = {}
        for config_name, runs in configs.items():
            real_runs = [r for r in runs if not _is_transient_error(r)]
            qualities = [r["quality_score"] for r in real_runs if not r.get("error")]
            costs = [r["cost_usd"] for r in real_runs if not r.get("error")]
            latencies = [r["latency_s"] for r in real_runs if not r.get("error")]
            quality_errors = sum(1 for r in real_runs if r.get("error"))
            rate_limited = sum(1 for r in runs if _is_transient_error(r))
            n_real = len(real_runs)

            # Aggregate sub_scores across runs
            sub_score_keys: set[str] = set()
            for r in real_runs:
                if r.get("sub_scores") and isinstance(r["sub_scores"], dict):
                    sub_score_keys.update(
                        k for k, v in r["sub_scores"].items()
                        if isinstance(v, (int, float))
                    )
            sub_scores_agg = {}
            for key in sorted(sub_score_keys):
                values = [
                    r["sub_scores"][key]
                    for r in real_runs
                    if r.get("sub_scores") and isinstance(r["sub_scores"].get(key), (int, float))
                    and not r.get("error")
                ]
                if values:
                    sub_scores_agg[key] = statistics.mean(values)

            agg[task_name][config_name] = {
                "mean_quality": statistics.mean(qualities) if qualities else 0.0,
                "std_quality": statistics.stdev(qualities) if len(qualities) > 1 else 0.0,
                "mean_cost": statistics.mean(costs) if costs else 0.0,
                "mean_latency": statistics.mean(latencies) if latencies else 0.0,
                "n_runs": n_real,
                "n_errors": quality_errors,
                "n_rate_limited": rate_limited,
                "error_rate": quality_errors / n_real if n_real else 0.0,
                "sub_scores": sub_scores_agg,
            }
    return agg


def pareto_frontier(configs: dict[str, dict]) -> set[str]:
    """Find configs on the Pareto frontier: no other config dominates on both quality AND cost.

    A config is dominated if there exists another with higher quality AND lower cost.
    """
    names = list(configs.keys())
    dominated = set()
    for a in names:
        for b in names:
            if a == b:
                continue
            qa = configs[a]["mean_quality"]
            ca = configs[a]["mean_cost"]
            qb = configs[b]["mean_quality"]
            cb = configs[b]["mean_cost"]
            if qb >= qa and cb <= ca and (qb > qa or cb < ca):
                dominated.add(a)
                break
    return set(names) - dominated


# Default cycle time cost — override this based on your workload
DEFAULT_CYCLE_TIME_COST = 0.002  # USD


def cost_per_success(stats: dict, cycle_time_cost: float = DEFAULT_CYCLE_TIME_COST) -> float:
    """Effective cost per successful outcome, including time waste from failures.

    Formula: (mean_cost + cycle_time_cost * (1/quality - 1)) / quality

    When quality is 0, returns infinity.

    Args:
        stats: Dict with mean_quality and mean_cost.
        cycle_time_cost: USD opportunity cost of a wasted LLM cycle.
    """
    q = stats["mean_quality"]
    if q <= 0:
        return float("inf")
    c = stats["mean_cost"]
    return (c + cycle_time_cost * (1.0 / q - 1.0)) / q


def recommend(
    task_name: str,
    configs: dict[str, dict],
    quality_floor: float = 0.5,
    cycle_time_cost: float = DEFAULT_CYCLE_TIME_COST,
) -> tuple[str | None, dict | None]:
    """Select config with lowest cost-per-success above quality floor.

    Args:
        task_name: Name of the task (for logging).
        configs: {config_name: stats_dict} from aggregate().
        quality_floor: Minimum acceptable quality score [0, 1].
        cycle_time_cost: USD opportunity cost of a wasted cycle.

    Returns:
        (config_name, stats) or (None, None) if no config qualifies.
    """
    candidates = {
        name: stats
        for name, stats in configs.items()
        if stats["mean_quality"] >= quality_floor
    }
    if not candidates:
        return None, None
    best_name = min(
        candidates,
        key=lambda n: cost_per_success(candidates[n], cycle_time_cost),
    )
    return best_name, candidates[best_name]


def get_recommendations(
    results: list[dict],
    quality_floors: dict[str, float] | None = None,
    default_floor: float = 0.5,
    cycle_time_cost: float = DEFAULT_CYCLE_TIME_COST,
) -> dict[str, str]:
    """Compute recommendations for all tasks from a list of benchmark results.

    Args:
        results: List of result dicts (from storage or runner).
        quality_floors: {task_name: floor} overrides. Falls back to default_floor.
        cycle_time_cost: USD opportunity cost of a wasted cycle.

    Returns:
        {task_name: config_name} for tasks with a qualifying recommendation.
    """
    if not results:
        return {}

    agg = aggregate(results)
    recs: dict[str, str] = {}
    for task_name, configs in agg.items():
        floor = (quality_floors or {}).get(task_name, default_floor)
        name, _ = recommend(task_name, configs, floor, cycle_time_cost)
        if name:
            recs[task_name] = name
    return recs


def format_table(
    task_name: str,
    configs: dict[str, dict],
    quality_floor: float = 0.5,
    cycle_time_cost: float = DEFAULT_CYCLE_TIME_COST,
) -> str:
    """Format a human-readable comparison table for a task.

    Returns a multi-line string suitable for terminal output.
    """
    frontier = pareto_frontier(configs)
    rec_name, _ = recommend(task_name, configs, quality_floor, cycle_time_cost)

    rows = []
    for name, stats in sorted(
        configs.items(), key=lambda x: (-x[1]["mean_quality"], x[1]["mean_cost"])
    ):
        on_frontier = "P" if name in frontier else " "
        is_rec = "<" if name == rec_name else " "
        above_floor = "+" if stats["mean_quality"] >= quality_floor else "-"
        cps = cost_per_success(stats, cycle_time_cost)
        cps_str = f"${cps:.5f}" if cps < 1.0 else "inf"
        rows.append(
            f"  {on_frontier} {above_floor} {is_rec} "
            f"{name:<28} quality={stats['mean_quality']:.3f}+/-{stats['std_quality']:.3f}  "
            f"cost=${stats['mean_cost']:.5f}  $/ok={cps_str}  "
            f"lat={stats['mean_latency']:.1f}s  "
            f"n={stats['n_runs']}  errors={stats['n_errors']}"
        )

    header = f"\n{'=' * 90}\n{task_name.upper()} (floor={quality_floor:.1f})\n{'=' * 90}"
    legend = "  Legend: P=Pareto frontier  +/-=above/below floor  <=recommended"
    return header + "\n" + legend + "\n" + "\n".join(rows)
