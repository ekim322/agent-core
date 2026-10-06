"""Shared stream events and optional console output for agents and model clients."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, TextIO

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.usage import RunUsage


class EventType(str, Enum):
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RUN_STATUS = "run_status"
    DONE = "done"


class DoneReason(str, Enum):
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class _TextPayload:
    delta: str


@dataclass(frozen=True)
class _ToolInvocationPayload:
    name: str
    call_id: str
    arguments: str | dict[str, Any]


@dataclass(frozen=True)
class _ToolOutputPayload:
    name: str
    call_id: str
    output: Any
    declined: bool


@dataclass(frozen=True)
class _StatusPayload:
    status: str


@dataclass(frozen=True)
class _CompletionPayload:
    reason: str
    output: Any
    messages: list[ModelMessage] | None
    usage: RunUsage | None
    detail: str | None


_EventPayload = (
    _TextPayload
    | _ToolInvocationPayload
    | _ToolOutputPayload
    | _StatusPayload
    | _CompletionPayload
)


class StreamEvent(BaseModel):
    """Carry a typed event payload in a mutable, extensible in-process envelope.

    Constructors retain payload objects and explicit null fields. BaseAgent
    may add SDK results or exceptions to terminal data. Callers preparing wire
    output must choose how those objects are serialized.
    """

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _from_payload(
        cls, kind: EventType, payload: _EventPayload, extra: dict[str, Any] | None = None
    ) -> StreamEvent:
        # Read fields shallowly: recursive dataclass serialization would copy or
        # convert SDK messages, arbitrary tool results and usage objects.
        attributes = {
            column.name: getattr(payload, column.name) for column in fields(payload)
        }
        if extra is not None:
            attributes.update(extra)
        return cls(type=kind, data=attributes)

    @classmethod
    def text_delta(cls, delta: str) -> StreamEvent:
        return cls._from_payload(EventType.TEXT_DELTA, _TextPayload(delta))

    @classmethod
    def reasoning_delta(cls, delta: str) -> StreamEvent:
        return cls._from_payload(EventType.REASONING_DELTA, _TextPayload(delta))

    @classmethod
    def tool_call(
        cls, name: str, call_id: str, arguments: str | dict[str, Any]
    ) -> StreamEvent:
        payload = _ToolInvocationPayload(name, call_id, arguments)
        return cls._from_payload(EventType.TOOL_CALL, payload)

    @classmethod
    def tool_result(
        cls,
        name: str,
        call_id: str,
        output: Any,
        declined: bool = False,
    ) -> StreamEvent:
        payload = _ToolOutputPayload(name, call_id, output, declined)
        return cls._from_payload(EventType.TOOL_RESULT, payload)

    @classmethod
    def run_status(cls, status: str, **data: Any) -> StreamEvent:
        return cls._from_payload(EventType.RUN_STATUS, _StatusPayload(status), data)

    @classmethod
    def done(
        cls,
        reason: DoneReason,
        *,
        output: Any | None = None,
        messages: list[ModelMessage] | None = None,
        usage: RunUsage | None = None,
        detail: str | None = None,
    ) -> StreamEvent:
        payload = _CompletionPayload(reason.value, output, messages, usage, detail)
        return cls._from_payload(EventType.DONE, payload)


_COLORS = {
    EventType.TEXT_DELTA: 36,
    EventType.REASONING_DELTA: 90,
    EventType.TOOL_CALL: 33,
    EventType.TOOL_RESULT: 32,
    EventType.RUN_STATUS: 90,
    EventType.DONE: 35,
}
_DELTAS = {EventType.TEXT_DELTA, EventType.REASONING_DELTA}


def _arguments(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        value = {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value.strip()
    return json.dumps(value, indent=2, default=str)


class StreamPrinter:
    """Print ordered deltas and tool blocks, with colors only on TTYs by default."""

    def __init__(
        self, output: TextIO | None = None, *, color: bool | None = None
    ) -> None:
        self.output = output if output is not None else sys.stdout
        self.color = (
            bool(getattr(self.output, "isatty", lambda: False)())
            if color is None
            else color
        )
        self._delta_kind: EventType | None = None

    def _paint(self, text: str, kind: EventType, *, error: bool = False) -> str:
        if not self.color:
            return text
        code = 31 if error else _COLORS[kind]
        return f"\x1b[{code}m{text}\x1b[0m"

    def write(self, event: StreamEvent) -> None:
        kind, data = event.type, event.data
        if self._delta_kind is not None and kind != self._delta_kind:
            self.output.write("\n")
        if kind in _DELTAS:
            self.output.write(self._paint(str(data.get("delta", "")), kind))
            self._delta_kind = kind
        else:
            self._delta_kind = None
            lines = self._block(event)
            for line in lines:
                self.output.write(
                    self._paint(line, kind, error=data.get("reason") == "error") + "\n"
                )
        self.output.flush()

    @staticmethod
    def _block(event: StreamEvent) -> list[str]:
        data = event.data
        if event.type in {EventType.TOOL_CALL, EventType.TOOL_RESULT}:
            if event.type == EventType.TOOL_CALL:
                body = _arguments(data.get("arguments"))
            else:
                result = data.get("output")
                body = result if isinstance(result, str) else repr(result)
            lines = [data.get("name") or "", body]
            if data.get("call_id"):
                lines.append(f"  call_id: {data['call_id']}")
            return lines
        if event.type == EventType.RUN_STATUS:
            return [f"[run_status] status={data.get('status')}"]
        if event.type == EventType.DONE:
            line = f"[done] reason={data.get('reason')}"
            if data.get("detail"):
                line += f" detail={data['detail']!r}"
            return [line]
        return []
