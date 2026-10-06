"""Track model and tool timing within one invocation and its inherited tasks."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
)

_BOUND_RUNS: ContextVar[dict[str, RunState]] = ContextVar(
    "agent_run_timings", default={}
)


def _milliseconds(start: float, end: float) -> int:
    return max(0, int(1000 * (end - start)))


@dataclass
class ModelCallTiming:
    """One model dispatch, its sequential parts, and its overlapping tool batch."""

    request_prep_ms: int | None = None
    ttft_ms: int | None = None
    tool_wait_ms: int | None = None
    part_starts: dict[int, datetime] = field(default_factory=dict)
    part_durations: dict[int, int] = field(default_factory=dict)
    tool_durations: dict[str, int] = field(default_factory=dict)
    response: ModelResponse | None = None
    _created: float = field(default_factory=time.perf_counter)
    _dispatch: float | None = None
    _ended: float | None = None
    _parts: dict[int, float] = field(default_factory=dict)
    _first_tool: float | None = None
    _last_tool: float | None = None

    @property
    def in_flight(self) -> bool:
        return self._ended is None

    def mark_dispatched(self) -> None:
        if self.in_flight and self._dispatch is None:
            self._dispatch = time.perf_counter()
            self.request_prep_ms = _milliseconds(self._created, self._dispatch)

    def observe_event(self, event: ModelResponseStreamEvent) -> None:
        if not self.in_flight or not isinstance(
            event, (PartStartEvent, PartDeltaEvent)
        ):
            return
        tick = time.perf_counter()
        if self._dispatch is not None and self.ttft_ms is None:
            self.ttft_ms = _milliseconds(self._dispatch, tick)
        if isinstance(event, PartStartEvent) and event.index not in self._parts:
            self._parts[event.index] = tick
            self.part_starts[event.index] = datetime.now(timezone.utc)

    def finish(self, response: ModelResponse | None) -> None:
        if not self.in_flight:
            return
        self._ended = time.perf_counter()
        previous = None
        for index, tick in sorted(self._parts.items(), key=lambda item: item[1]):
            if previous is not None:
                self.part_durations[previous[0]] = _milliseconds(previous[1], tick)
            previous = (index, tick)
        if previous is not None:
            self.part_durations[previous[0]] = _milliseconds(previous[1], self._ended)
        self.response = response

    def note_tool_started(self, at: float) -> None:
        self._first_tool = at if self._first_tool is None else min(at, self._first_tool)

    def note_tool_finished(self, at: float) -> None:
        self._last_tool = at if self._last_tool is None else max(at, self._last_tool)
        if self._first_tool is not None:
            self.tool_wait_ms = _milliseconds(self._first_tool, self._last_tool)

    def start_of_part(self, index: int, fallback: datetime) -> datetime:
        return self.part_starts.get(index, fallback)

    def duration_of_part(self, index: int) -> int | None:
        return self.part_durations.get(index)


@dataclass
class RunState:
    """Track one invocation's model calls, tool durations and unsaved messages.

    Bind the state to each SDK run so dispatch and tool hooks can find it in
    inherited task context. Nested bindings restore the previous lookup when
    they exit; concurrent invocations keep independent bindings.
    """

    calls: list[ModelCallTiming] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    unrecorded_messages: list[ModelMessage] = field(default_factory=list)
    _model_stream_opened: bool = False

    @contextmanager
    def bind_to_run(self, run_id: str) -> Iterator[None]:
        if run_id not in self.run_ids:
            self.run_ids.append(run_id)
        token = _BOUND_RUNS.set({**_BOUND_RUNS.get(), run_id: self})
        try:
            yield
        finally:
            _BOUND_RUNS.reset(token)

    @classmethod
    def for_run(cls, run_id: str) -> RunState | None:
        return _BOUND_RUNS.get().get(run_id)

    @property
    def current_call(self) -> ModelCallTiming | None:
        if self.calls:
            return self.calls[-1]
        return None

    @property
    def model_stream_opened(self) -> bool:
        return self._model_stream_opened

    def begin_node_stream(self) -> None:
        self._model_stream_opened = False

    def mark_model_stream_opened(self) -> None:
        self._model_stream_opened = True

    def start_model_call(self) -> None:
        self.calls.append(ModelCallTiming())

    def mark_inference_dispatched(self) -> None:
        if call := self.current_call:
            call.mark_dispatched()

    def observe_model_event(self, event: ModelResponseStreamEvent) -> None:
        if call := self.current_call:
            call.observe_event(event)

    def finish_model_call(self, response: ModelResponse | None) -> None:
        if call := self.current_call:
            call.finish(response)

    def timing_for_response(self, response: ModelResponse) -> ModelCallTiming | None:
        for call in reversed(self.calls):
            if call.response is response:
                return call
        return None

    @contextmanager
    def measure_tool(self, tool_call_id: str) -> Iterator[None]:
        call = self.current_call
        start = time.perf_counter()
        if call is not None:
            call.note_tool_started(start)
        try:
            yield
        finally:
            if call is not None:
                end = time.perf_counter()
                call.tool_durations[tool_call_id] = _milliseconds(start, end)
                call.note_tool_finished(end)

    def turn_start_index(self, messages: list[ModelMessage]) -> int | None:
        identities = set(self.run_ids)
        for index, message in enumerate(messages):
            if getattr(message, "run_id", None) in identities:
                return index
        return None
