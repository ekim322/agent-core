"""Scope SDK node streams and bind response timing before cleanup."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    ModelResponseStreamEvent,
    HandleResponseEvent,
)
from pydantic_ai.run import AgentRun

from agent_core.streaming import StreamEvent
from agent_core.utils.telemetry.model_trace import trace_model_request
from agent_core.utils.telemetry.run_state import RunState
from agent_core.agent.request import Deps


async def stream_node(
    node: Any,
    agent_run: AgentRun[Deps, Any],
    state: RunState,
    map_request: Callable[[ModelResponseStreamEvent], StreamEvent | None],
    map_tool: Callable[[HandleResponseEvent], StreamEvent | None],
) -> AsyncGenerator[StreamEvent, None]:
    """Stream a node through event callbacks within its timing scope.

    Tool timing comes from the execution probe rather than event delivery,
    so a slow stream consumer does not inflate reported tool duration.
    """
    if Agent.is_call_tools_node(node):
        async with node.stream(agent_run.ctx) as events:
            async for raw in events:
                event = map_tool(raw)
                if event is not None:
                    yield event
        return
    if not Agent.is_model_request_node(node):
        return
    state.start_model_call()
    with trace_model_request(
        len(state.calls), agent_run.ctx.deps.model.model_name
    ) as trace:
        try:
            async with node.stream(agent_run.ctx) as events:
                state.mark_model_stream_opened()
                async for raw in events:
                    state.observe_model_event(raw)
                    if trace is not None:
                        trace.observe(raw, state.current_call)
                    event = map_request(raw)
                    if event is not None:
                        yield event
        finally:
            # Bind timing to the response object persisted by the SDK, not
            # its index: retries and pending requests can alter positions.
            history = agent_run.ctx.state.message_history
            response = (
                history[-1]
                if history and isinstance(history[-1], ModelResponse)
                else None
            )
            state.finish_model_call(response)
            if trace is not None:
                trace.finish(response, state.current_call)
