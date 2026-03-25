"""Model configurations for benchmarking.

Ships with built-in configs for popular models accessible via OpenRouter.
Users can extend with custom configs via ModelRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """A model configuration to benchmark.

    Attributes:
        name: Short identifier used in reports (e.g., "haiku", "gemini-2.5-flash").
        model: OpenRouter model slug (e.g., "anthropic/claude-haiku-4.5").
        thinking_budget: Extended thinking token budget. 0 = disabled.
    """

    name: str
    model: str  # OpenRouter model slug
    thinking_budget: int = 0


# Built-in model configurations — all accessible via OpenRouter
ALL_CONFIGS: list[ModelConfig] = [
    # Anthropic
    ModelConfig("haiku", "anthropic/claude-haiku-4.5"),
    ModelConfig("haiku-think-1k", "anthropic/claude-haiku-4.5", 1024),
    ModelConfig("haiku-think-4k", "anthropic/claude-haiku-4.5", 4096),
    ModelConfig("sonnet", "anthropic/claude-sonnet-4"),
    ModelConfig("sonnet-think-1k", "anthropic/claude-sonnet-4", 1024),
    # Google Gemini
    ModelConfig("gemini-2.5-flash", "google/gemini-2.5-flash"),
    ModelConfig("gemini-2.5-flash-think-1k", "google/gemini-2.5-flash", 1024),
    ModelConfig("gemini-2.5-flash-think-4k", "google/gemini-2.5-flash", 4096),
    ModelConfig("gemini-2.5-flash-lite", "google/gemini-2.5-flash-lite"),
    ModelConfig("gemini-2.5-flash-lite-think-1k", "google/gemini-2.5-flash-lite", 1024),
    ModelConfig("gemini-3-flash", "google/gemini-3-flash-preview"),
    ModelConfig("gemini-3-flash-think-1k", "google/gemini-3-flash-preview", 1024),
    # OpenRouter free models
    ModelConfig("qwen3-coder-free", "qwen/qwen3-coder:free"),
    ModelConfig("llama-3.3-70b-free", "meta-llama/llama-3.3-70b-instruct:free"),
    ModelConfig("mistral-small-free", "mistralai/mistral-small-3.1-24b-instruct:free"),
    # OpenRouter cheap paid models
    ModelConfig("devstral-small", "mistralai/devstral-small-2507"),
    ModelConfig("qwen3-coder-30b", "qwen/qwen3-coder-30b-a3b-instruct"),
    ModelConfig("qwen3-235b", "qwen/qwen3-235b-a22b-07-25"),
    ModelConfig("llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct"),
    # OpenAI
    ModelConfig("gpt-4.1-mini", "openai/gpt-4.1-mini"),
    ModelConfig("gpt-4.1-nano", "openai/gpt-4.1-nano"),
]

# Lookup by short name
CONFIG_BY_NAME: dict[str, ModelConfig] = {c.name: c for c in ALL_CONFIGS}


class ModelRegistry:
    """Extensible registry for model configurations.

    Starts with built-in configs. Users can add custom models.
    """

    def __init__(self, include_defaults: bool = True) -> None:
        self._configs: dict[str, ModelConfig] = {}
        if include_defaults:
            for c in ALL_CONFIGS:
                self._configs[c.name] = c

    def add(self, config: ModelConfig) -> None:
        self._configs[config.name] = config

    def remove(self, name: str) -> None:
        self._configs.pop(name, None)

    def get(self, name: str) -> ModelConfig | None:
        return self._configs.get(name)

    @property
    def all_configs(self) -> list[ModelConfig]:
        return list(self._configs.values())

    @property
    def names(self) -> list[str]:
        return list(self._configs.keys())

    def subset(self, names: list[str]) -> list[ModelConfig]:
        """Return configs matching the given names, preserving order."""
        return [self._configs[n] for n in names if n in self._configs]
