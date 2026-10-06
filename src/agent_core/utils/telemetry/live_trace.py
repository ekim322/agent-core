"""Collect optional execution rows and redact their retained payloads."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from uuid import uuid4

from pydantic_core import to_jsonable_python

from agent_core.streaming import EventType, StreamEvent
from agent_core.persistence.records import ExecutionSpan
from agent_core.utils.telemetry._redaction import RedactedText, TraceRedactor

_TRACE: ContextVar[tuple[LiveTrace, str | None] | None] = ContextVar(
    "agent_inspector", default=None
)


def current_trace() -> tuple[LiveTrace, str | None] | None:
    return _TRACE.get()


class LiveTrace:
    """Collect execution rows with redacted visible payloads.

    Limits are opt-in: max_spans bounds retained rows, max_capture_chars bounds
    each serialized payload, and max_text_chars bounds each span's streamed
    text. Any omission sets truncated. Limits never interrupt agent execution.
    """

    def __init__(
        self,
        secrets: tuple[str, ...] = (),
        *,
        max_spans: int | None = None,
        max_capture_chars: int | None = None,
        max_text_chars: int | None = None,
    ):
        for value in (max_spans, max_capture_chars, max_text_chars):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("trace limits must be positive integers or None")
        self.spans: list[ExecutionSpan] = []
        self.started = time.monotonic()
        self.truncated = False
        self._max_spans = max_spans
        self._max_capture_chars = max_capture_chars
        self._max_text_chars = max_text_chars
        self._redactor = TraceRedactor(secrets)
        self._rows: dict[str, ExecutionSpan] = {}
        self._text_streams: dict[str, RedactedText] = {}
        self._announcements: dict[tuple[str, str], Any] = {}

    def now(self) -> int:
        return max(0, round(1000 * (time.monotonic() - self.started)))

    def find_span(self, span_id: str | None) -> ExecutionSpan | None:
        return self._rows.get(span_id) if span_id is not None else None

    def redact_text(self, value: str) -> str:
        return self._redactor.text(value)

    def capture(self, value: Any) -> Any:
        try:
            result = self._redactor.payload(
                to_jsonable_python(value, fallback=lambda item: "[unavailable]")
            )
            cap = self._max_capture_chars
            if cap is not None and len(json.dumps(result, ensure_ascii=False)) > cap:
                self.truncated = True
                return "[truncated]"
            return result
        except (TypeError, ValueError, RecursionError):
            return "[unavailable]"

    @contextmanager
    def bind(self):
        """Capture executions in this task and its children; restore context on exit."""
        token = _TRACE.set((self, None))
        try:
            yield self
        finally:
            _TRACE.reset(token)

    @contextmanager
    def span(self, kind: Literal["agent", "tool", "model"], name: str, inputs: Any):
        """Retain a child invocation and close its status/timing with this scope.

        Yield None when the span budget is exhausted. Inputs are captured here;
        callers must use capture for outputs they assign directly to a span.
        """
        cap = self._max_spans
        if cap is not None and len(self.spans) >= cap:
            self.truncated = True
            yield None
            return
        binding = current_trace()
        parent = binding[1] if binding is not None and binding[0] is self else None
        row = ExecutionSpan(
            id=uuid4().hex,
            parent_id=parent,
            kind=kind,
            name=self.redact_text(name),
            started_ms=self.now(),
            input=self.capture(inputs),
        )
        self.spans.append(row)
        self._rows[row.id] = row
        token = _TRACE.set((self, row.id))
        try:
            yield row
        except (GeneratorExit, asyncio.CancelledError):
            row.status = "cancelled"
            raise
        except BaseException as error:
            row.status = "error"
            row.output = {"error": type(error).__name__}
            raise
        finally:
            self._finish_text(row)
            for key in tuple(self._announcements):
                if key[0] == row.id:
                    del self._announcements[key]
            row.ended_ms = self.now()
            if row.status == "running":
                row.status = "complete"
            _TRACE.reset(token)

    def _append_text(self, row: ExecutionSpan, text: str) -> None:
        cap = self._max_text_chars
        if cap is not None and len(row.text) + len(text) > cap:
            self.truncated = True
            text = text[: max(0, cap - len(row.text))]
        row.text += text

    def _stream_text(self, row: ExecutionSpan, delta: str) -> None:
        stream = self._text_streams.get(row.id)
        if stream is None:
            stream = self._redactor.stream()
            self._text_streams[row.id] = stream
        self._append_text(row, stream.feed(delta))

    def _finish_text(self, row: ExecutionSpan) -> None:
        stream = self._text_streams.pop(row.id, None)
        if stream is not None:
            self._append_text(row, stream.finish())

    def observe(self, row: ExecutionSpan | None, event: StreamEvent) -> None:
        if row is None:
            return
        data = event.data
        if event.type == EventType.TEXT_DELTA:
            self._stream_text(row, str(data.get("delta", "")))
        elif event.type == EventType.TOOL_CALL:
            self._announcements[row.id, data["call_id"]] = self.capture(
                data.get("arguments")
            )
        elif event.type == EventType.TOOL_RESULT:
            self._tool_result(row, data)
        elif event.type == EventType.RUN_STATUS:
            # Event retention follows the row budget when a cap is configured.
            cap = self._max_spans
            if cap is None or len(row.events) < cap:
                captured = self.capture(data)
                event_data = (
                    captured if isinstance(captured, dict) else {"payload": captured}
                )
                row.events.append({"at_ms": self.now(), **event_data})
            else:
                self.truncated = True
        elif event.type == EventType.DONE:
            self._finish_text(row)
            row.status = data.get("reason", "complete")
            row.output = self.capture(data.get("output"))

    def _tool_result(self, parent: ExecutionSpan, data: dict) -> None:
        call_id = data["call_id"]
        inputs = self._announcements.pop((parent.id, call_id), None)
        executed = next(
            (
                row
                for row in reversed(self.spans)
                if row.parent_id == parent.id and row.call_id == call_id
            ),
            None,
        )
        if executed is not None:
            if data.get("declined"):
                executed.status = "error"
                if executed.output is None:
                    executed.output = self.capture(data.get("output"))
            return
        with self.span("tool", data.get("name") or "Rejected tool", inputs) as row:
            if row is not None:
                row.parent_id = parent.id
                row.call_id = call_id
                row.status = "declined"
                row.output = self.capture(data.get("output"))
