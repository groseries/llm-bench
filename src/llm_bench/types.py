"""Core protocols and dataclasses for llm-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Fixture(Protocol):
    """A test input for benchmarking a model on a specific task.

    Users implement this protocol to provide golden test cases.
    The simplest approach is to use SimpleFixture below.
    """

    @property
    def fixture_id(self) -> str: ...

    @property
    def system_prompt(self) -> str: ...

    @property
    def user_prompt(self) -> str: ...


@runtime_checkable
class QualityScorer(Protocol):
    """Scores an LLM response for a specific task.

    Users implement this protocol to define quality metrics.
    The score() method must return a dict with a 'score' key (float 0-1).
    Additional keys are stored as sub_scores for analysis.
    """

    def score(self, response: str, fixture: Fixture, **kwargs: Any) -> dict: ...


@dataclass
class SimpleFixture:
    """Convenience fixture implementation — use this if you don't need a custom class."""

    fixture_id: str
    system_prompt: str
    user_prompt: str


@dataclass
class TaskDefinition:
    """A benchmark task: a named LLM call type with its scorer, fixtures, and quality floor.

    This is the primary configuration object. Users define one TaskDefinition per
    distinct LLM use case (e.g., "summarize", "classify", "generate_code").
    """

    name: str
    scorer: QualityScorer
    fixtures: list[Fixture]
    quality_floor: float = 0.5
    models: list[str] | None = None  # subset of model configs to test; None = all


@dataclass
class BenchmarkResult:
    """Result of running one (task, model, fixture) benchmark."""

    task_name: str
    config_name: str
    fixture_id: str
    quality_score: float  # composite [0, 1]
    sub_scores: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    raw_response: str = ""
    error: str | None = None
    run_index: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "config_name": self.config_name,
            "fixture_id": self.fixture_id,
            "quality_score": self.quality_score,
            "sub_scores": self.sub_scores,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "raw_response": self.raw_response,
            "error": self.error,
            "run_index": self.run_index,
            "timestamp": self.timestamp,
        }
