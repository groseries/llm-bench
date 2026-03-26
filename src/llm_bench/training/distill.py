"""Distillation — prepare and submit fine-tuning jobs from benchmark data.

Exports highest-quality responses from expensive models, formatted for
provider-specific fine-tuning APIs.

Usage:
    from llm_bench.training import distill
    distill.prepare_openai(storage, "summarize", "finetune.jsonl", min_quality=0.7)
    job = distill.submit_openai("finetune.jsonl", "gpt-4.1-nano", api_key="sk-...")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def prepare_openai(
    storage: Any,
    task_name: str,
    output_path: str | Path,
    min_quality: float = 0.7,
    source_model: str | None = None,
    system_prompt: str | None = None,
) -> int:
    """Prepare OpenAI fine-tuning JSONL from best benchmark responses.

    Selects the highest-quality response per fixture and formats as
    OpenAI chat fine-tuning examples.

    Args:
        storage: Storage backend.
        task_name: Task to export data for.
        output_path: Path to write JSONL file.
        min_quality: Minimum quality score to include.
        source_model: Only use responses from this model. None = best from any.
        system_prompt: System prompt to include in examples. If None, omitted.

    Returns:
        Number of examples written.
    """
    results = storage.load_results(task_name=task_name)

    # Group by fixture, keep best response
    best: dict[str, dict] = {}
    for r in results:
        if r.get("error") or r.get("quality_score", 0) < min_quality:
            continue
        if source_model and r["config_name"] != source_model:
            continue
        raw = r.get("raw_response", "")
        if not raw:
            continue
        fid = r.get("fixture_id", "unknown")
        if fid not in best or r["quality_score"] > best[fid]["quality_score"]:
            best[fid] = r

    path = Path(output_path)
    count = 0
    with open(path, "w") as f:
        for r in best.values():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Use fixture_id as a reference; in practice users should
            # reconstruct the original user prompt from their fixtures
            messages.append({"role": "user", "content": f"[fixture: {r.get('fixture_id', 'unknown')}]"})
            messages.append({"role": "assistant", "content": r.get("raw_response", "")})
            f.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    logger.info("Wrote %d fine-tuning examples to %s", count, path)
    return count


def submit_openai(
    training_file: str | Path,
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    suffix: str | None = None,
    n_epochs: int | str = "auto",
) -> Any:
    """Submit a fine-tuning job to OpenAI.

    Args:
        training_file: Path to JSONL training data.
        model: Base model to fine-tune.
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        suffix: Custom suffix for the fine-tuned model name.
        n_epochs: Number of training epochs. "auto" lets OpenAI decide.

    Returns:
        OpenAI FineTuningJob object.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI SDK required. Install with: pip install openai")

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # Upload training file
    path = Path(training_file)
    with open(path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    logger.info("Uploaded training file: %s (%s)", file_obj.id, path.name)

    # Create fine-tuning job
    kwargs: dict[str, Any] = {
        "training_file": file_obj.id,
        "model": model,
    }
    if suffix:
        kwargs["suffix"] = suffix
    if n_epochs != "auto":
        kwargs["hyperparameters"] = {"n_epochs": int(n_epochs)}

    job = client.fine_tuning.jobs.create(**kwargs)
    logger.info("Fine-tuning job created: %s (model: %s)", job.id, model)
    return job


def prepare_vertex(
    storage: Any,
    task_name: str,
    output_path: str | Path,
    min_quality: float = 0.7,
    source_model: str | None = None,
) -> int:
    """Prepare Google Vertex AI fine-tuning JSONL.

    Same selection logic as prepare_openai but in Vertex format.

    Returns:
        Number of examples written.
    """
    results = storage.load_results(task_name=task_name)

    best: dict[str, dict] = {}
    for r in results:
        if r.get("error") or r.get("quality_score", 0) < min_quality:
            continue
        if source_model and r["config_name"] != source_model:
            continue
        raw = r.get("raw_response", "")
        if not raw:
            continue
        fid = r.get("fixture_id", "unknown")
        if fid not in best or r["quality_score"] > best[fid]["quality_score"]:
            best[fid] = r

    path = Path(output_path)
    count = 0
    with open(path, "w") as f:
        for r in best.values():
            example = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"[fixture: {r.get('fixture_id', 'unknown')}]"}]},
                    {"role": "model", "parts": [{"text": r.get("raw_response", "")}]},
                ]
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    logger.info("Wrote %d Vertex fine-tuning examples to %s", count, path)
    return count
