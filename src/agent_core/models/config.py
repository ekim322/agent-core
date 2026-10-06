"""Translate caller model options into the selected provider's SDK settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.settings import ModelSettings

from agent_core.models.runtime import ModelRuntime

# Shared SDK effort names. False requests disabled thinking where supported.
ThinkingEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

# OpenAI Responses reasoning-summary setting. Actual reasoning events depend
# on the selected model and provider response.
ReasoningSummary = Literal["auto", "concise", "detailed"]

ToolChoice = Literal["none", "required", "auto"]
ServiceTier = Literal["auto", "default", "flex", "priority"]
DEFAULT_OPENAI_SERVICE_TIER: ServiceTier = "default"

# Responses reasoning requests exclude sampling controls before SDK dispatch.
_OPENAI_SAMPLING_DROPPED_WHEN_REASONING = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
)


@dataclass
class ModelConfig:
    """Configure reasoning, generation and tool selection for a model call.

    Pass this to BaseAgent or LLMClient. ``to_settings(model_name)`` selects
    the provider settings type and translates reasoning effort. ``thinking``
    is required; False requests disabled thinking where supported. Optional generation
    fields set to None are omitted, leaving their choice to the SDK/provider.

    Common fields are collected first and ``extra`` overrides them. Provider
    translation then applies its own rules: OpenAI supplies explicit storage,
    summary and service-tier choices and drops sampling fields when thinking
    is enabled. ``service_tier=None`` leaves an extra-supplied tier intact.


    ``extra`` supports SDK-specific settings; it is not a universal final
    override. The selected model and SDK still determine which settings work.
    """

    thinking: bool | ThinkingEffort

    reasoning_summary: ReasoningSummary = "auto"
    parallel_tool_calls: bool = True
    # Explicitly request that OpenAI Responses are not stored server-side.
    openai_store: bool = False
    # None skips this assignment; an extra-supplied tier can still be present.
    service_tier: ServiceTier | None = DEFAULT_OPENAI_SERVICE_TIER

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tool_choice: ToolChoice | list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_settings(self, model_name: str) -> ModelSettings:
        """Collect shared options and apply the selected provider's wire policy."""
        provider = ModelRuntime.infer_provider(model_name)
        common_names = (
            "thinking",
            "parallel_tool_calls",
            "temperature",
            "max_tokens",
            "top_p",
            "timeout",
            "seed",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "tool_choice",
        )
        options = {}
        for name in common_names:
            value = getattr(self, name)
            if value is not None:
                options[name] = value
        options.update(self.extra)
        if provider == "openai":
            from pydantic_ai.models.openai import OpenAIResponsesModelSettings

            explicit = {
                "openai_store": self.openai_store,
                "openai_reasoning_summary": self.reasoning_summary,
            }
            if self.service_tier is not None:
                explicit["service_tier"] = self.service_tier
            options.update(explicit)
            return OpenAIResponsesModelSettings(**self._without_sampling(options))
        if provider == "google":
            from pydantic_ai.models.google import GoogleModelSettings

            return GoogleModelSettings(**options)
        if provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModelSettings

            return AnthropicModelSettings(**options)
        return ModelSettings(**options)

    def _without_sampling(self, options: dict[str, Any]) -> dict[str, Any]:
        if self.thinking is False:
            return options
        excluded = set(_OPENAI_SAMPLING_DROPPED_WHEN_REASONING)
        return {name: value for name, value in options.items() if name not in excluded}
