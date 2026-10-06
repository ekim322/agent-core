"""Visible model inputs, response phases and usage for optional inspection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestContext

from agent_core.streaming import StreamEvent
from agent_core.persistence.records import ExecutionSpan, ModelDiagnostics
from agent_core.utils.telemetry.live_trace import LiveTrace, current_trace

if TYPE_CHECKING:
    from agent_core.utils.telemetry.run_state import ModelCallTiming


def _tool_arguments(part: ToolCallPart) -> Any:
    arguments = part.args
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _visible_message(message: ModelMessage) -> dict[str, Any]:
    """Inspect visible parts while excluding reasoning, signatures and internals."""
    visible = []
    for part in message.parts:
        item: dict[str, Any] = {"part_kind": part.part_kind}
        if part.part_kind == "thinking":
            item["omitted"] = True
        elif isinstance(part, ToolCallPart):
            item.update(
                tool_name=part.tool_name,
                tool_call_id=part.tool_call_id,
                args=_tool_arguments(part),
            )
        else:
            for key in ("content", "tool_name", "tool_call_id", "args"):
                if hasattr(part, key):
                    item[key] = getattr(part, key)
        visible.append(item)
    return {"kind": message.kind, "parts": visible}


class ModelTrace:
    """Inspect a prepared request and its response without private reasoning.

    Provider options and visible inputs are projected separately, so expanding
    diagnostics does not accidentally retain arbitrary SDK settings. LiveTrace
    owns redaction and retention for every captured payload.
    """

    def __init__(self, trace: LiveTrace, row: ExecutionSpan | None):
        self.trace = trace
        self.row = row

    def capture_request(self, request: ModelRequestContext) -> None:
        row = self.row
        if row is None or row.model is None:
            return
        for field, value in self._provider_options(request).items():
            setattr(row.model, field, value)
        tool_names = []
        for definition in request.model_request_parameters.function_tools:
            tool_names.append(self.trace.redact_text(definition.name))
        row.model.available_tools = tool_names
        row.model.phase = "waiting_for_output"
        row.input = self.trace.capture(self._request_input(request, tool_names))

    @staticmethod
    def _provider_options(request: ModelRequestContext) -> dict[str, Any]:
        """Interpret reasoning options with provider-body effort taking precedence."""
        settings = request.model_settings or {}
        route = getattr(request.model, "route", None)
        if route is None:
            provider = request.model.system
        else:
            provider = route.provider

        candidates = []
        body = settings.get("extra_body")
        if isinstance(body, dict):
            candidates.append(body.get("reasoning_effort"))
        candidates.append(settings.get("openai_reasoning_effort"))
        effort = next((value for value in candidates if value), None)
        summary = settings.get("openai_reasoning_summary")
        options = {"provider": provider, "thinking": settings.get("thinking")}
        for name, value in (
            ("reasoning_effort", effort),
            ("reasoning_summary", summary),
        ):
            options[name] = value if isinstance(value, str) else None
        return options

    @staticmethod
    def _request_input(
        request: ModelRequestContext, tool_names: list[str]
    ) -> dict[str, Any]:
        """Project inspectable inputs, selecting only approved generation settings."""
        settings = request.model_settings or {}
        generation = {}
        for name in ("temperature", "max_tokens", "top_p", "seed"):
            if name in settings:
                generation[name] = settings[name]

        parameters = request.model_request_parameters
        payload: dict[str, Any] = {
            "model": request.model.model_name,
            "streaming": request.streaming,
            "message_count": len(request.messages),
            "settings": generation,
            "available_tools": tool_names,
        }
        payload["output_tools"] = [tool.name for tool in parameters.output_tools]
        payload["instructions"] = [
            instruction.content for instruction in parameters.instruction_parts or ()
        ]
        payload["messages"] = list(map(_visible_message, request.messages))
        return payload

    def observe(
        self, event: ModelResponseStreamEvent, timing: ModelCallTiming | None = None
    ) -> None:
        row = self.row
        if row is None or row.model is None:
            return
        diagnostics = row.model
        if timing is not None:
            diagnostics.ttft_ms = timing.ttft_ms
        if isinstance(event, PartStartEvent):
            part = event.part
        elif isinstance(event, PartDeltaEvent):
            part = event.delta
        else:
            return
        if isinstance(part, (ThinkingPart, ThinkingPartDelta)):
            diagnostics.reasoning_observed = True
            diagnostics.phase = "thinking"
        elif isinstance(part, (ToolCallPart, ToolCallPartDelta)):
            diagnostics.phase = "preparing_tools"
            name = part.tool_name if isinstance(part, ToolCallPart) else None
            if name is not None:
                diagnostics.pending_tools.append(self.trace.redact_text(name))
        elif isinstance(part, (TextPart, TextPartDelta)):
            diagnostics.phase = "generating_text"
            content = part.content if isinstance(part, TextPart) else part.content_delta
            self.trace.observe(row, StreamEvent.text_delta(content))

    def finish(
        self, response: ModelResponse | None, timing: ModelCallTiming | None
    ) -> None:
        row = self.row
        if row is None or row.model is None:
            return
        row.model.ttft_ms = None if timing is None else timing.ttft_ms
        if response is not None:
            row.model.name = self.trace.redact_text(
                response.model_name or row.model.name
            )
            usage = response.usage
            for field in ("input_tokens", "output_tokens", "cache_read_tokens"):
                setattr(row.model, field, getattr(usage, field))
            text = []
            calls = []
            for part in response.parts:
                if isinstance(part, TextPart):
                    text.append(part.content)
                elif isinstance(part, ToolCallPart):
                    calls.append(
                        {
                            "name": part.tool_name,
                            "arguments": _tool_arguments(part),
                            "call_id": part.tool_call_id,
                        }
                    )
            row.output = self.trace.capture(
                {
                    "text": "".join(text),
                    "tool_calls": calls,
                    "finish_reason": response.finish_reason,
                }
            )


_ACTIVE_MODEL_TRACE: ContextVar[ModelTrace | None] = ContextVar(
    "agent_core_active_model_trace", default=None
)


@contextmanager
def trace_model_request(
    number: int, model_name: str
) -> Iterator[ModelTrace | None]:
    """Bind one inspector for the request, including dispatch and stream cleanup."""
    binding = current_trace()
    inspection = None
    with ExitStack() as lifetime:
        if binding is not None:
            trace, _ = binding
            row = lifetime.enter_context(
                trace.span("model", f"Model request {number}", None)
            )
            inspection = ModelTrace(trace, row)
            if row is not None:
                row.model = ModelDiagnostics(name=trace.redact_text(model_name))
        token = _ACTIVE_MODEL_TRACE.set(inspection)
        try:
            yield inspection
        finally:
            _ACTIVE_MODEL_TRACE.reset(token)


def capture_model_request(context: ModelRequestContext) -> None:
    """Capture prepared inputs only while their model span is the active scope."""
    inspection = _ACTIVE_MODEL_TRACE.get()
    if inspection is None or inspection.row is None:
        return
    binding = current_trace()
    if binding != (inspection.trace, inspection.row.id):
        return
    inspection.capture_request(context)
