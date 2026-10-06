"""Curated general-purpose model identifiers, reviewed through September 2026.

ModelId includes capability and cost tiers for experimentation. Identifiers may
be provider aliases; membership does not guarantee availability. Applications
choose their own model and reasoning defaults. See docs/model-catalog.md.
"""

from enum import StrEnum, unique

__all__ = ["ModelId", "ReasoningEffort"]


@unique
class ModelId(StrEnum):
    """Use a curated member as a model-name string, or supply another explicit ID."""

    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    CLAUDE_SONNET_5_5 = "claude-sonnet-5-5"
    CLAUDE_OPUS_5_5 = "claude-opus-5-5"
    CLAUDE_FABLE_5_1 = "claude-fable-5-1"

    GPT_6_LUNA = "gpt-6-luna"
    GPT_6_1_SOL = "gpt-6.1-sol"
    GPT_6_ASTRA = "gpt-6-astra"

    GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_3_8_FLASH = "gemini-3.8-flash"
    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"


@unique
class ReasoningEffort(StrEnum):
    """Reasoning labels; accepted levels depend on the selected provider/model."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
