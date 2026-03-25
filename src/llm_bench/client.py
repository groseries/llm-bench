"""Thin OpenRouter client via the OpenAI SDK.

All LLM calls go through OpenRouter. Users bring their own provider keys
(Anthropic, Google, Mistral, etc.) via OpenRouter's BYOK feature.

Cost is read from response.usage.cost (actual billed USD, not estimated).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    """Response from an LLM call with cost and token metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    model: str  # actual model used (may differ from requested if fallback)


class Client:
    """OpenRouter API client.

    Args:
        api_key: OpenRouter API key.
        base_url: Override for testing or custom endpoints.
        default_max_tokens: Default max output tokens.
        app_name: Your app name (sent in HTTP-Referer for OpenRouter rankings).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        default_max_tokens: int = 4096,
        app_name: str = "llm-bench",
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._default_max_tokens = default_max_tokens
        self._app_name = app_name

    def call(
        self,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int | None = None,
        thinking_budget: int = 0,
        provider_sort: str | None = None,
        provider_order: list[str] | None = None,
        fallback_models: list[str] | None = None,
    ) -> LLMResponse:
        """Make an LLM call via OpenRouter.

        Args:
            model: OpenRouter model slug (e.g., "anthropic/claude-haiku-4.5").
            system: System prompt.
            user_message: User message.
            max_tokens: Max output tokens (defaults to self._default_max_tokens).
            thinking_budget: Extended thinking token budget. 0 = disabled.
            provider_sort: Sort providers by "price", "latency", or "throughput".
            provider_order: Preferred provider ordering.
            fallback_models: Fallback model chain (tried in order on failure).

        Returns:
            LLMResponse with text, tokens, actual cost, and latency.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        extra_body: dict[str, Any] = {}

        # Provider routing preferences
        provider: dict[str, Any] = {}
        if provider_sort:
            provider["sort"] = provider_sort
        if provider_order:
            provider["order"] = provider_order
        if provider:
            extra_body["provider"] = provider

        # Model fallback chain
        if fallback_models:
            extra_body["models"] = [model] + fallback_models
            extra_body["route"] = "fallback"

        # Extended thinking
        if thinking_budget > 0:
            extra_body["reasoning"] = {"max_tokens": thinking_budget}

        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            extra_headers={"HTTP-Referer": self._app_name},
            extra_body=extra_body if extra_body else None,
        )
        latency = time.monotonic() - t0

        # Extract response text
        text = response.choices[0].message.content or "" if response.choices else ""

        # Extract token counts
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Extract actual cost from OpenRouter (key feature — not estimated)
        cost_usd = 0.0
        if usage:
            # OpenRouter puts cost in usage as an extra field
            cost_usd = getattr(usage, "cost", 0.0) or 0.0
            # Some versions put it in a different location
            if cost_usd == 0.0 and hasattr(response, "_raw_response"):
                raw = response._raw_response
                if hasattr(raw, "json"):
                    data = raw.json()
                    cost_usd = data.get("usage", {}).get("cost", 0.0) or 0.0

        # Actual model used (may differ from requested on fallback)
        actual_model = response.model or model

        logger.info(
            "LLM call [%s]: %d in + %d out tokens, $%.6f, %.1fs",
            actual_model,
            input_tokens,
            output_tokens,
            cost_usd,
            latency,
        )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_s=latency,
            model=actual_model,
        )
