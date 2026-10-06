"""Inputs callers prepare for one agent invocation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic

from typing_extensions import TypeVar
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import Model
from pydantic_ai.tools import Tool
from pydantic_ai.usage import UsageLimits

from agent_core.models.config import ModelConfig

Deps = TypeVar("Deps")
Req = TypeVar("Req", default=Any)
UserPrompt = str | Sequence[UserContent]


@dataclass
class PreparedRequest(Generic[Deps]):
    """Supply agent inputs derived from an application request.

    Return this from ``_prepare_run`` when callers use ``request=``. Explicit
    non-None arguments to run or stream override the corresponding fields.
    The caller owns authorization and supplies the dependencies used by tools.
    """

    user_msg: UserPrompt | None
    message_history: list[ModelMessage] | None = None
    instructions: str | Sequence[str] | None = None
    deps: Deps | None = None
    model: Model | str | None = None
    config: ModelConfig | None = None
    tools: Sequence[Tool[Deps]] | None = None
    max_hops: int | None = None
    usage_limits: UsageLimits | None = None
    session_id: str | None = None
    message_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] | None = None
