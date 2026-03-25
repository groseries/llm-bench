"""llm-bench — Benchmark LLM models on quality-per-cost and auto-select the optimal model per task."""

from llm_bench.analyze import (
    aggregate,
    cost_per_success,
    format_table,
    get_recommendations,
    pareto_frontier,
    recommend,
)
from llm_bench.client import Client, LLMResponse
from llm_bench.configs import ALL_CONFIGS, CONFIG_BY_NAME, ModelConfig, ModelRegistry
from llm_bench.runner import BenchmarkRunner
from llm_bench.scheduler import ContinuousRunner
from llm_bench.selector import AutoSelector
from llm_bench.storage import JSONFileStorage, SQLiteStorage
from llm_bench.types import BenchmarkResult, Fixture, QualityScorer, SimpleFixture, TaskDefinition

__all__ = [
    # Core types
    "BenchmarkResult",
    "Fixture",
    "QualityScorer",
    "SimpleFixture",
    "TaskDefinition",
    # Configs
    "ALL_CONFIGS",
    "CONFIG_BY_NAME",
    "ModelConfig",
    "ModelRegistry",
    # Client
    "Client",
    "LLMResponse",
    # Runner
    "BenchmarkRunner",
    # Analysis
    "aggregate",
    "cost_per_success",
    "format_table",
    "get_recommendations",
    "pareto_frontier",
    "recommend",
    # Auto-selection
    "AutoSelector",
    # Storage
    "JSONFileStorage",
    "SQLiteStorage",
    # Scheduler
    "ContinuousRunner",
]
