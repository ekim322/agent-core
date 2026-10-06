"""Project invocation history and execute the selected storage policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.run import AgentRun

from agent_core.agent.request import Deps
from agent_core.persistence import ChatWriter, RunRecords, messages_for_persistence
from agent_core.persistence.background import BackgroundPersistence, write_records
from agent_core.utils.telemetry.run_state import RunState

logger = logging.getLogger("agent_core._execution.storage")


@dataclass(frozen=True)
class _InvocationHistory:
    """Select the SDK and interrupted output belonging to this invocation."""

    result: AgentRunResult[Any] | None
    run: AgentRun[Deps, Any] | None
    timing: RunState
    start_index: int

    def populate(self, batch: RunRecords) -> None:
        owned_messages = messages_for_persistence(
            state=self.timing,
            start_index=self.start_index,
            agent_name=batch.agent_name,
            result=self.result,
            agent_run=self.run,
        )
        batch.add_run(owned_messages, self.timing)


async def save_messages(
    writer: ChatWriter | None,
    agent_name: str,
    agent_id: str,
    *,
    result: AgentRunResult[Any] | None,
    agent_run: AgentRun[Deps, Any] | None,
    session_id: str | None,
    message_id: str | None,
    user_id: str | None,
    metadata: dict[str, Any] | None,
    state: RunState,
    run_error: BaseException | None,
    total_latency_ms: int | None,
    partial_message_start_index: int = 0,
    persist_in_background: bool = False,
    background_persistence: BackgroundPersistence | None = None,
    max_hops_finalized: bool = False,
) -> None:
    """Write this invocation's records, awaiting writer completion by default.

    Projection failures skip storage; awaited writer errors and cancellation
    propagate. Background mode uses the agent-owned bounded dispatcher. Full queues
    reject submission; accepted writes retain execution context and can be
    drained at shutdown. Storage remains best effort. Storage adapters
    own transactions and retries.
    """
    if writer is None:
        return
    if persist_in_background and background_persistence is None:
        raise RuntimeError("Background persistence requires an execution owner")

    history = _InvocationHistory(
        result, agent_run, state, partial_message_start_index
    )
    try:
        tags = dict(metadata or {})
        tags["agent_id"] = agent_id
        batch = RunRecords(
            agent_name=agent_name,
            metadata=tags,
            user_id=user_id,
            message_id=message_id,
            session_id=session_id,
        )
        history.populate(batch)
    except Exception:
        logger.exception(
            "Invocation history could not be projected for storage",
            extra={"event_name": "agent.persistence_build_failed"},
        )
        return

    if run_error is None:
        if max_hops_finalized:
            batch.add_max_hops_finalized()
    else:
        failure_detail = str(run_error)
        batch.add_run_failed(failure_detail or type(run_error).__name__)
    batch.stamp_total_latency(total_latency_ms)
    if not batch.is_empty():
        if persist_in_background:
            await background_persistence.submit(batch)
        else:
            await write_records(writer, batch)
