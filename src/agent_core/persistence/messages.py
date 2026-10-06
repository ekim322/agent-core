"""Choose the history owned by one invocation, including interrupted output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.run import AgentRun, AgentRunResult

if TYPE_CHECKING:
    from agent_core.utils.telemetry.run_state import RunState

logger = logging.getLogger(__name__)


def messages_for_persistence(
    *,
    result: AgentRunResult[Any] | None,
    agent_run: AgentRun[Any, Any] | None,
    start_index: int,
    agent_name: str,
    state: RunState | None = None,
) -> list[ModelMessage]:
    """Select this invocation's recorded history, then append otherwise lost output.

    The caller's history length bounds the prefix to discard. A measured run
    boundary can shorten it when the SDK merges history requests. Invalid
    boundaries fall back to SDK new_messages; unrecorded output survives that
    fallback too. Equal messages already in the history are not appended again.
    """
    source = agent_run
    if result is not None:
        source = result
    history = [] if source is None else list(source.all_messages())
    boundary = start_index
    if state is not None:
        measured = state.turn_start_index(history)
        if measured is not None and measured < boundary:
            boundary = measured
    if boundary < 0 or boundary > len(history):
        selected = []
        if source is not None:
            logger.warning(
                "History boundary is out of range; selecting the SDK's new messages",
                extra=dict(
                    event_name="agent.persistence_slice_invalid",
                    agent_name=agent_name,
                    start_index=boundary,
                    message_count=len(history),
                ),
            )
            selected.extend(source.new_messages())
    else:
        selected = history[boundary:]
    if state is not None:
        for emitted in state.unrecorded_messages:
            if emitted not in history and emitted not in selected:
                selected.append(emitted)
    return selected
