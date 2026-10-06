"""Mutable execution and cleanup state owned by one invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.run import AgentRun

from agent_core.agent.request import Deps, PreparedRequest
from agent_core.utils.telemetry.run_state import RunState


@dataclass
class Invocation(Generic[Deps]):
    """Keep one caller's execution and cleanup data together across phases."""

    inputs: PreparedRequest[Deps]
    started_at: float
    state: RunState = field(default_factory=RunState)
    run: AgentRun[Deps, Any] | None = None
    result: AgentRunResult[Any] | None = None
    error: BaseException | None = None
    text_prefix: str = ""
    finalized: bool = False
    terminal_sent: bool = False
    history_boundary: int = field(init=False)

    def __post_init__(self) -> None:
        # Recovery histories contain new work. Keep the caller's original
        # boundary fixed throughout retries and finalization.
        self.history_boundary = len(self.inputs.message_history or [])
