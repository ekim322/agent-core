"""Measure agent and validated tool execution through the shared observability API.

RunProbe connects SDK dispatch and tool hooks to invocation-local timing state.
The optional LiveTrace receives visible payloads within the same execution scopes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import aclosing, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any
from uuid import uuid4

from observability import bind_observability_context, observe_operation
from observability.correlation import current_observability_context
from observability.operation_telemetry import ObservedOperation
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition

from agent_core.streaming import DoneReason, EventType, StreamEvent
from agent_core.tools.outcomes import has_tool_error_metadata
from agent_core.utils.telemetry.live_trace import current_trace
from agent_core.utils.telemetry.model_trace import capture_model_request
from agent_core.utils.telemetry.run_state import RunState

_ACTIVE_INVOCATION: ContextVar[_AgentInvocation | None] = ContextVar(
    "agent_probe_invocation", default=None
)
# Retain the logger name used by existing log queries after moving the probes.
_LOG = logging.getLogger("agent_core.utils.telemetry.live_trace")


def _invocation_prompt(args, kwargs):
    if args:
        return args[0]
    for key in ("user_msg", "prompt"):
        if kwargs.get(key) is not None:
            return kwargs[key]
    request = kwargs.get("request")
    if request is not None:
        return getattr(request, "prompt", request)
    context = kwargs.get("context")
    return context.prompt() if context is not None else None


class _AgentInvocation:
    """Own execution identity, observed events and the eventual stream outcome."""

    def __init__(self, agent, prompt):
        self.agent = agent
        self.task = asyncio.current_task()
        self.prompt = prompt
        self.parent = current_observability_context()
        binding = current_trace()
        self.inspector = binding[0] if binding is not None else None
        self.row = None
        self.operation: ObservedOperation | None = None
        self.terminal_reason: str | None = None
        self.terminal_seen = False

    def is_forwarding(self, agent) -> bool:
        # A tool call starts a distinct invocation even for recursive use of
        # this agent. Subclass forwarding has no active delegation tool.
        delegation = current_observability_context().get("tool_call_id")
        return (
            self.agent is agent
            and self.task is asyncio.current_task()
            and delegation is None
        )

    @contextmanager
    def measure(self):
        inspector_scope = nullcontext()
        if self.inspector is not None:
            inspector_scope = self.inspector.span(
                "agent", self.agent.agent_name, self.prompt
            )
        try:
            with inspector_scope as self.row:
                with bind_observability_context(
                    agent_execution_id=uuid4().hex,
                    parent_agent_execution_id=self.parent.get("agent_execution_id"),
                    delegation_tool_call_id=self.parent.get("tool_call_id"),
                    agent_name=self.agent.agent_name,
                    execution_span_id=self.row.id if self.row is not None else None,
                    tool_call_id=None,
                    tool_name=None,
                ), observe_operation("agent.run") as self.operation:
                    token = _ACTIVE_INVOCATION.set(self)
                    try:
                        _LOG.info(
                            "Agent invocation entered",
                            extra={"event_name": "agent.execution.started"},
                        )
                        yield self
                    except GeneratorExit:
                        if not self.terminal_seen:
                            self._end_without_answer("cancelled", "consumer_closed")
                        raise
                    except asyncio.CancelledError:
                        self._end_without_answer("cancelled", "cancelled")
                        raise
                    else:
                        if not self.terminal_seen:
                            self._end_without_answer("error", "missing_terminal")
                    finally:
                        _ACTIVE_INVOCATION.reset(token)
        except GeneratorExit:
            # Closing after DONE still runs awaited persistence. A successful
            # closure must not turn the inspector's terminal result into a
            # cancellation merely because Python used GeneratorExit to close.
            if self.row is not None and self.terminal_seen:
                self.row.status = self.terminal_reason or "complete"
            raise

    def _end_without_answer(self, outcome: str, reason: str) -> None:
        self.operation.set_attribute("terminal_reason", reason)
        self.operation.set_outcome(outcome)
        if self.row is not None:
            self.row.status = outcome

    def observe(self, event: StreamEvent) -> None:
        if self.inspector is not None:
            self.inspector.observe(self.row, event)
        if event.type == EventType.DONE:
            self.terminal_seen = True
            self.terminal_reason = event.data.get("reason")
            self.operation.set_attribute("terminal_reason", self.terminal_reason)
            if self.terminal_reason == DoneReason.ERROR:
                exception = event.data.get("exception")
                if isinstance(exception, BaseException):
                    self.operation.record_exception(exception)
                else:
                    self.operation.set_outcome("error")
            elif self.terminal_reason == DoneReason.CANCELLED:
                self.operation.set_outcome("cancelled")
        elif event.type == EventType.RUN_STATUS:
            if event.data.get("status") == "max_hops_finalizing":
                self.operation.set_attribute("max_hops_finalized", True)


def traced_agent_stream(method):
    """Observe one invocation through iterator cleanup and awaited storage."""

    @wraps(method)
    async def measured(agent, *args, **kwargs):
        current = _ACTIVE_INVOCATION.get()
        invocation = None
        scope = nullcontext()
        if current is None or not current.is_forwarding(agent):
            invocation = _AgentInvocation(agent, _invocation_prompt(args, kwargs))
            scope = invocation.measure()
        with scope:
            async with aclosing(method(agent, *args, **kwargs)) as source:
                async for event in source:
                    if invocation is not None:
                        invocation.observe(event)
                    yield event

    return measured


async def trace_tool(call, handler, args):
    """Observe validated execution; visible payloads stay in the optional trace."""
    binding = current_trace()
    trace = binding[0] if binding else None
    inputs = call.args
    if trace and isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except ValueError:
            pass
    with trace.span("tool", call.tool_name, inputs) if trace else nullcontext() as row:
        fields = dict(tool_name=call.tool_name, tool_call_id=call.tool_call_id)
        if row is not None:
            row.call_id = call.tool_call_id
            fields["execution_span_id"] = row.id
        with (
            bind_observability_context(**fields),
            observe_operation("agent.tool") as operation,
        ):
            _LOG.info(
                "Tool execution entered", extra={"event_name": "agent.tool.started"}
            )
            result = await handler(args)
            failed = has_tool_error_metadata(getattr(result, "metadata", None))
            if failed:
                operation.set_outcome("error")
                operation.set_attribute("tool_result_error", True)
            if row is not None:
                row.output = trace.capture(getattr(result, "return_value", result))
                if failed:
                    row.status = "error"
            return result


class RunProbe(AbstractCapability[Any]):
    """Start inference timing at dispatch and tool timing after argument validation.

    BaseAgent installs this as the innermost SDK capability so preparation and
    outer capability work complete before the measured operations begin.
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if state := RunState.for_run(ctx.run_id):
            state.mark_inference_dispatched()
        capture_model_request(request_context)
        return await handler(request_context)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        state = RunState.for_run(ctx.run_id)
        if state is not None:
            with state.measure_tool(call.tool_call_id):
                return await trace_tool(call, handler, args)
        return await trace_tool(call, handler, args)
