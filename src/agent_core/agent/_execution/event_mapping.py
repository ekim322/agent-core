"""Convert SDK events and invocation outcomes into StreamEvent contracts."""

from __future__ import annotations

import logging

from observability import bind_observability_context
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from agent_core.streaming import DoneReason, StreamEvent
from agent_core.persistence import messages_for_persistence
from agent_core.models.runtime import (
    ModelProviderUnavailable,
    ModelRouteSaturated,
    ModelTransportUnavailable,
)
from agent_core.tools.outcomes import has_tool_error_metadata
from agent_core.agent.request import Deps
from agent_core.agent._execution.invocation import Invocation

logger = logging.getLogger("agent_core._execution.events")


def map_request_event(event: ModelResponseStreamEvent) -> StreamEvent | None:
    """Expose answer/reasoning text; complete tool arguments come from tool events."""
    if isinstance(event, PartStartEvent):
        part = event.part
    elif isinstance(event, PartDeltaEvent):
        part = event.delta
    else:
        return None
    if isinstance(part, (TextPart, ThinkingPart)):
        content = part.content
    elif isinstance(part, (TextPartDelta, ThinkingPartDelta)):
        content = part.content_delta
    else:
        return None
    if not content:
        return None
    if isinstance(part, (ThinkingPart, ThinkingPartDelta)):
        return StreamEvent.reasoning_delta(content)
    return StreamEvent.text_delta(content)


def map_tool_event(event: HandleResponseEvent) -> StreamEvent | None:
    """Expose resolved calls and results, retaining safe rejection diagnostics."""
    if isinstance(event, FunctionToolCallEvent):
        call = event.part
        return StreamEvent.tool_call(call.tool_name, call.tool_call_id, call.args or "")
    if not isinstance(event, FunctionToolResultEvent):
        return None
    returned = event.part
    name = getattr(returned, "tool_name", "")
    call_id = getattr(returned, "tool_call_id", "")
    output = getattr(returned, "content", None)
    outcome = getattr(returned, "outcome", "success")
    retry = isinstance(returned, RetryPromptPart)
    if retry or outcome != "success":
        if retry:
            reason = (
                "argument_validation" if isinstance(output, list) else "retry_requested"
            )
        else:
            reason = (
                outcome if outcome in {"failed", "denied", "interrupted"} else "unknown"
            )
        with bind_observability_context(tool_name=name, tool_call_id=call_id):
            logger.info(
                "Agent tool call declined",
                extra={
                    "event_name": "agent.tool.declined",
                    "outcome": "rejected",
                    "reason": reason,
                },
            )
    declined = (
        retry
        or outcome != "success"
        or has_tool_error_metadata(getattr(returned, "metadata", None))
    )
    return StreamEvent.tool_result(name, call_id, output, declined)


def success_event(invocation: Invocation[Deps], agent_name: str) -> StreamEvent:
    result = invocation.result
    if result is not None and isinstance(result.output, str) and invocation.text_prefix:
        result.output = invocation.text_prefix + result.output
    history = None
    new_messages = None
    if result is not None:
        history = messages_for_persistence(
            result=result,
            agent_run=invocation.run,
            state=invocation.state,
            start_index=0,
            agent_name=agent_name,
        )
        new_messages = messages_for_persistence(
            result=result,
            agent_run=invocation.run,
            state=invocation.state,
            start_index=invocation.history_boundary,
            agent_name=agent_name,
        )
    event = StreamEvent.done(
        DoneReason.COMPLETE,
        output=result.output if result else None,
        messages=history,
        usage=result.usage if result else None,
    )
    event.data.update(
        result=result,
        new_messages=new_messages,
        max_hops_finalized=invocation.finalized,
    )
    return event


def failure_event(failure: Exception) -> StreamEvent:
    event = StreamEvent.done(DoneReason.ERROR, detail=str(failure))
    event.data["exception"] = failure
    categories = (
        (ModelRouteSaturated, "MODEL_CAPACITY_EXCEEDED"),
        (ModelTransportUnavailable, "MODEL_TRANSPORT_UNAVAILABLE"),
        (ModelProviderUnavailable, "MODEL_PROVIDER_UNAVAILABLE"),
    )
    for kind, code in categories:
        if isinstance(failure, kind):
            event.data.update(
                error_code=code,
                retryable=True,
                retry_after_seconds=failure.retry_after_seconds,
            )
            if isinstance(failure, ModelProviderUnavailable):
                event.data["status_code"] = failure.status_code
            break
    return event
