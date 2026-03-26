"""Export benchmark results into training-ready formats."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def export_for_router(
    storage: Any,
    task_name: str | None = None,
    min_quality: float = 0.0,
    min_runs: int = 2,
) -> list[dict]:
    """Export data for router training: (prompt, best_model) pairs.

    For each (task, fixture), finds the model with the best quality-to-cost ratio
    above the minimum quality threshold. Returns one training example per fixture.

    Args:
        storage: Storage backend with benchmark results.
        task_name: Filter to a specific task. None = all tasks.
        min_quality: Minimum quality score to consider a result valid.
        min_runs: Minimum runs per (fixture, model) pair to be considered.

    Returns:
        List of dicts with keys: task_name, fixture_id, prompt, best_model, quality, cost.
    """
    results = storage.load_results(task_name=task_name)
    if not results:
        return []

    # Group by (task, fixture, model)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        task = r.get("task_name") or r.get("call_type", "unknown")
        fixture = r.get("fixture_id", "unknown")
        config = r["config_name"]
        if not r.get("error") and r.get("quality_score", 0) >= min_quality:
            grouped[(task, fixture, config)].append(r)

    # For each (task, fixture), pick the best model by quality (cost as tiebreaker)
    best_by_fixture: dict[tuple, dict] = {}
    for (task, fixture, config), runs in grouped.items():
        if len(runs) < min_runs:
            continue
        avg_quality = sum(r["quality_score"] for r in runs) / len(runs)
        avg_cost = sum(r.get("cost_usd", 0) for r in runs) / len(runs)

        key = (task, fixture)
        if key not in best_by_fixture or avg_quality > best_by_fixture[key]["quality"]:
            best_by_fixture[key] = {
                "task_name": task,
                "fixture_id": fixture,
                "best_model": config,
                "quality": avg_quality,
                "cost": avg_cost,
            }

    return list(best_by_fixture.values())


def export_for_predictor(
    storage: Any,
    task_name: str | None = None,
) -> list[dict]:
    """Export data for quality predictor training: (prompt+response, quality_score).

    Returns one training example per benchmark result that has a raw_response.

    Args:
        storage: Storage backend.
        task_name: Filter to a specific task. None = all tasks.

    Returns:
        List of dicts with keys: text (prompt + response concatenated), label (quality_score),
        task_name, config_name.
    """
    results = storage.load_results(task_name=task_name)
    examples = []
    for r in results:
        if r.get("error"):
            continue
        # We need the raw_response to train on — check if storage has it
        raw = r.get("raw_response", "")
        if not raw:
            continue

        # Concatenate prompt context with response for the predictor input
        # The predictor learns to judge response quality from the text
        task = r.get("task_name") or r.get("call_type", "unknown")
        examples.append({
            "text": raw,  # just the response — scorer judges response quality
            "label": r["quality_score"],
            "task_name": task,
            "config_name": r["config_name"],
        })

    return examples


def to_openai_jsonl(
    storage: Any,
    output_path: str | Path,
    task_name: str | None = None,
    min_quality: float = 0.7,
    source_model: str | None = None,
) -> int:
    """Export best responses as OpenAI fine-tuning JSONL.

    For each fixture, selects the highest-quality response (optionally from a specific
    source model) and formats it as an OpenAI fine-tuning example.

    Args:
        storage: Storage backend.
        output_path: Path to write JSONL file.
        task_name: Filter to a specific task.
        min_quality: Minimum quality to include.
        source_model: Only use responses from this model config. None = best from any model.

    Returns:
        Number of examples written.
    """
    results = storage.load_results(task_name=task_name)

    # Group by fixture and find best response
    by_fixture: dict[str, dict] = {}
    for r in results:
        if r.get("error"):
            continue
        if r.get("quality_score", 0) < min_quality:
            continue
        if source_model and r["config_name"] != source_model:
            continue
        raw = r.get("raw_response", "")
        if not raw:
            continue

        fixture_id = r.get("fixture_id", "unknown")
        if fixture_id not in by_fixture or r["quality_score"] > by_fixture[fixture_id]["quality_score"]:
            by_fixture[fixture_id] = r

    # Write JSONL
    path = Path(output_path)
    count = 0
    with open(path, "w") as f:
        for r in by_fixture.values():
            example = {
                "messages": [
                    {"role": "user", "content": r.get("raw_response", "")},
                ]
            }
            # If we have the original prompts stored, use proper message format
            # For now, we use user→assistant format with the response as the target
            # Users should customize this based on their fixture structure
            f.write(json.dumps(example) + "\n")
            count += 1

    return count


def to_vertex_jsonl(
    storage: Any,
    output_path: str | Path,
    task_name: str | None = None,
    min_quality: float = 0.7,
    source_model: str | None = None,
) -> int:
    """Export best responses as Google Vertex AI fine-tuning JSONL.

    Same logic as to_openai_jsonl but in Vertex format.

    Returns:
        Number of examples written.
    """
    results = storage.load_results(task_name=task_name)

    by_fixture: dict[str, dict] = {}
    for r in results:
        if r.get("error") or r.get("quality_score", 0) < min_quality:
            continue
        if source_model and r["config_name"] != source_model:
            continue
        raw = r.get("raw_response", "")
        if not raw:
            continue
        fixture_id = r.get("fixture_id", "unknown")
        if fixture_id not in by_fixture or r["quality_score"] > by_fixture[fixture_id]["quality_score"]:
            by_fixture[fixture_id] = r

    path = Path(output_path)
    count = 0
    with open(path, "w") as f:
        for r in by_fixture.values():
            example = {
                "contents": [
                    {"role": "user", "parts": [{"text": "Respond to this prompt."}]},
                    {"role": "model", "parts": [{"text": r.get("raw_response", "")}]},
                ]
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    return count
