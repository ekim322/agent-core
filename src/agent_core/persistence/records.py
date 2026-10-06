"""Project SDK history into storage rows, then read usage and timings by call.

Schemas stay independent of concrete storage. RunRecords creates ordered batches
for application writers; CallTrace and TurnTrace read those batches without
counting repeated call-level measurements more than once.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Literal
from uuid import UUID, uuid4

import ulid
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_serializer
from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ThinkingPart,
)

from agent_core.tools.outcomes import has_tool_error_metadata

if TYPE_CHECKING:
    from agent_core.utils.telemetry.run_state import ModelCallTiming, RunState

logger = logging.getLogger(__name__)
RESPONSE_KINDS = frozenset(
    {
        "text",
        "thinking",
        "tool_call",
        "builtin-tool-call",
        "builtin-tool-return",
    }
)


class ChatMessage(BaseModel):
    """Visible dialogue with caller-owned conversation identity."""

    model_config = ConfigDict(extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: AwareDatetime


class ModelDiagnostics(BaseModel):
    """Live progress and provider usage for one model invocation."""

    name: str
    provider: str | None = None
    thinking: bool | str | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    reasoning_observed: bool = False
    phase: Literal[
        "preparing_request",
        "waiting_for_output",
        "thinking",
        "preparing_tools",
        "generating_text",
    ] = "preparing_request"
    pending_tools: list[str] = Field(default_factory=list)
    ttft_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None


class ExecutionSpan(BaseModel):
    """One parent-linked invocation in the live inspector.

    Summary projections keep supplied word counts when detail_loaded is False."""

    id: str = Field(min_length=1)
    parent_id: str | None = None
    kind: Literal["agent", "tool", "model"]
    name: str
    started_ms: int = Field(ge=0)
    ended_ms: int | None = Field(default=None, ge=0)
    status: str = "running"
    input: Any = None
    output: Any = None
    text: str = ""
    events: list[dict[str, Any]] = Field(default_factory=list)
    detail_loaded: bool = True
    word_counts: dict[str, int | None] | None = None
    call_id: str | None = None
    model: ModelDiagnostics | None = None

    @model_serializer(mode="wrap")
    def serialize_with_word_counts(self, handler):
        projection = handler(self)
        if self.detail_loaded and self.kind == "tool":
            counts = {}
            for label in ("input", "output"):
                counts[label] = payload_word_count(getattr(self, label))
            projection.update(word_counts=counts)
        return projection


def payload_word_count(value: Any) -> int | None:
    """Count payload keys and values, ignoring recursive container references.

    JSON object/array strings are counted as structures. Other scalar strings
    split on whitespace and underscores; punctuation alone contributes no words.
    """
    if value is None:
        return None
    pending = [(value, frozenset())]
    total = 0
    while pending:
        item, ancestors = pending.pop()
        if isinstance(item, str):
            try:
                decoded = json.loads(item)
            except ValueError:
                decoded = None
            if isinstance(decoded, (dict, list)):
                item = decoded
        if isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity in ancestors:
                continue
            path = ancestors | {identity}
            if isinstance(item, dict):
                for key, child in item.items():
                    pending.append((str(key), path))
                    pending.append((child, path))
            else:
                pending.extend((child, path) for child in item)
        elif item is not None:
            words = re.split(r"[\s_]+", str(item))
            total += len([word for word in words if re.search(r"[^\W_]", word)])
    return total


def validate_execution_spans(spans: list[ExecutionSpan]) -> None:
    """Require unique IDs and parents already present in emission order."""
    known_ids: set[str] = set()
    for entry in spans:
        if entry.id in known_ids:
            raise ValueError(f"Duplicate execution span ID: {entry.id!r}")
        parent = entry.parent_id
        if parent is not None and parent not in known_ids:
            raise ValueError(
                f"Span {entry.id!r} references parent {parent!r} before it is recorded"
            )
        known_ids.add(entry.id)


@dataclass
class EventRecord:
    """One ordered storage row; call-level usage repeats across response rows.

    to_doc retains private metadata. to_api_dict omits user_id and metadata.
    Readers tolerate unknown document fields and accept ISO timestamps.
    """

    id: str = ""
    session_id: str | None = None
    message_id: str | None = None
    user_id: str | None = None
    agent_name: str | None = None
    kind: str = ""
    timestamp: datetime | str | None = None
    seq: int = 0

    content: Any = None
    tool_args: Any = None
    metadata: dict[str, Any] | None = None

    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_error: bool | None = None

    signature: str | None = None

    latency_ms: int | None = None
    total_latency_ms: int | None = None

    call_id: str | None = None
    model: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    ttft_ms: int | None = None
    request_prep_ms: int | None = None
    tool_wait_ms: int | None = None

    _PRIVATE_FIELDS: ClassVar[frozenset[str]] = frozenset({"user_id", "metadata"})

    def to_doc(self) -> dict[str, Any]:
        values = asdict(self)
        return {"_id" if key == "id" else key: value for key, value in values.items()}

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> EventRecord:
        values = {}
        for column in fields(cls):
            source = "_id" if column.name == "id" and "_id" in doc else column.name
            if source in doc:
                values[column.name] = doc[source]
        return cls(**values)

    def to_api_dict(self) -> dict[str, Any]:
        projection = asdict(self)
        for private in self._PRIVATE_FIELDS:
            projection.pop(private, None)
        return projection

    @property
    def is_response(self) -> bool:
        return self.kind in RESPONSE_KINDS


def _storage_value(value: Any, ancestors: frozenset[int] = frozenset()) -> Any:
    """Snapshot payloads for storage without retaining binary bytes or cycles."""
    if isinstance(value, BinaryContent):
        return dict(
            kind="binary",
            media_type=value.media_type,
            size=len(value.data or b""),
            identifier=getattr(value, "identifier", None),
        )
    if isinstance(value, (dict, list, tuple)):
        if id(value) in ancestors:
            return "[circular]"
        path = ancestors | {id(value)}
        if isinstance(value, dict):
            converted = {}
            for key, child in value.items():
                converted[str(key)] = _storage_value(child, path)
            return converted
        return [_storage_value(child, path) for child in value]
    if value is None or isinstance(value, (str, bool, int, float, datetime)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _storage_value(dump(mode="json"), ancestors | {id(value)})
        except Exception:
            pass
    return str(value)


@dataclass
class RunRecords:
    """Build a writer batch in history order, committing one message at a time.

    A successful projection commits both its rows and tool-call links. Conversion
    failures are logged and skip that message; subsequent messages can still be
    recorded. Tool returns use the most recent preceding response's call ID and
    duration, even when the SDK reuses a tool-call ID on a later hop.
    """

    session_id: str | None = None
    message_id: str | None = None
    user_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, Any] | None = None
    events: list[EventRecord] = field(default_factory=list)
    _tool_links: dict[str, tuple[str, int | None]] = field(
        default_factory=dict, repr=False
    )
    _output_started: bool = field(default=False, repr=False)
    _instructions_stored: bool = field(default=False, repr=False)
    _PART_KINDS: ClassVar[dict[str, str]] = {
        "user-prompt": "user_msg",
        "system-prompt": "system_prompt",
        "tool-call": "tool_call",
        "tool-return": "tool_return",
        "retry-prompt": "retry_prompt",
    }

    def __post_init__(self) -> None:
        self.metadata = _storage_value(self.metadata)

    def is_empty(self) -> bool:
        return len(self.events) == 0

    def add_run(
        self, messages: Iterable[ModelMessage], state: RunState | None = None
    ) -> None:
        """Append messages with their local timings; omit later recovery inputs."""
        for position, message in enumerate(messages):
            if position == 0 and isinstance(message, ModelResponse):
                logger.warning(
                    "Record batch begins with model output rather than a request",
                    extra={
                        "event_name": "agent.persistence_request_missing",
                        "message_id": str(self.message_id),
                    },
                )
            try:
                if isinstance(message, ModelRequest):
                    rows, instructions_stored = self._request_rows(message)
                    self._commit(rows)
                    self._instructions_stored = instructions_stored
                elif isinstance(message, ModelResponse):
                    timing = (
                        None if state is None else state.timing_for_response(message)
                    )
                    rows, links = self._response_rows(message, timing)
                    self._commit(rows)
                    self._tool_links.update(links)
                    self._output_started = True
            except Exception:
                logger.exception(
                    "Message projection failed; its rows were omitted from the batch",
                    extra={"event_name": "agent.persistence_record_failed"},
                )

    def _commit(self, rows: list[EventRecord]) -> None:
        offset = len(self.events)
        for index, row in enumerate(rows, start=offset):
            row.seq = index
        self.events.extend(rows)

    def _new_record(
        self, kind: str, timestamp: datetime, **payload: Any
    ) -> EventRecord:
        identity = dict(
            session_id=self.session_id,
            message_id=self.message_id,
            user_id=self.user_id,
            agent_name=self.agent_name,
            metadata=self.metadata,
        )
        return EventRecord(
            id=str(ulid.new()), kind=kind, timestamp=timestamp, **identity, **payload
        )

    def _request_rows(self, request: ModelRequest) -> tuple[list[EventRecord], bool]:
        rows = []
        has_instructions = self._instructions_stored
        timestamp = request.timestamp or datetime.now(timezone.utc)
        if not self._output_started and not has_instructions and request.instructions:
            rows.append(
                self._new_record(
                    "system_prompt",
                    timestamp,
                    content=_storage_value(request.instructions),
                )
            )
            has_instructions = True
        for part in request.parts:
            discriminator = getattr(part, "part_kind", None)
            if discriminator is None:
                continue
            is_followup = isinstance(part, (BaseToolReturnPart, RetryPromptPart))
            if self._output_started and not is_followup:
                continue
            if isinstance(part, SystemPromptPart):
                if has_instructions:
                    continue
                has_instructions = True
            payload = dict(content=_storage_value(getattr(part, "content", None)))
            if is_followup:
                tool_id = part.tool_call_id
                tool_name = part.tool_name
                if isinstance(part, RetryPromptPart):
                    tool_id = tool_id or None
                    tool_name = tool_name or None
                payload.update(tool_name=tool_name, tool_call_id=tool_id)
                link = self._tool_links.get(tool_id)
                if link is not None:
                    payload["call_id"] = link[0]
                if isinstance(part, BaseToolReturnPart):
                    payload["latency_ms"] = None if link is None else link[1]
                    if has_tool_error_metadata(getattr(part, "metadata", None)):
                        payload["tool_error"] = True
            kind = self._PART_KINDS.get(discriminator, discriminator)
            part_time = getattr(part, "timestamp", None) or timestamp
            rows.append(self._new_record(kind, part_time, **payload))
        return rows, has_instructions

    def _response_rows(
        self,
        response: ModelResponse,
        timing: ModelCallTiming | None,
    ) -> tuple[list[EventRecord], dict[str, tuple[str, int | None]]]:
        rows = []
        links = {}
        call_id = str(ulid.new())
        timestamp = response.timestamp or datetime.now(timezone.utc)
        usage = response.usage
        call_fields = {
            "call_id": call_id,
            "model": response.model_name or "",
            "provider": response.provider_name or "",
        }
        for metric in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            call_fields[metric] = getattr(usage, metric) or None
        for index, part in enumerate(response.parts):
            discriminator = getattr(part, "part_kind", None)
            if discriminator is None:
                continue
            payload = dict(call_fields)
            if timing is not None:
                payload["latency_ms"] = timing.duration_of_part(index)
                if index == 0:
                    for metric in ("ttft_ms", "request_prep_ms", "tool_wait_ms"):
                        payload[metric] = getattr(timing, metric)
            if isinstance(part, BaseToolCallPart):
                payload.update(
                    tool_args=_storage_value(part.args),
                    tool_name=part.tool_name,
                    tool_call_id=part.tool_call_id,
                )
                if part.tool_call_id:
                    duration = (
                        None
                        if timing is None
                        else timing.tool_durations.get(part.tool_call_id)
                    )
                    links[part.tool_call_id] = (call_id, duration)
            else:
                payload["content"] = _storage_value(getattr(part, "content", None))
                if isinstance(part, ThinkingPart) and part.signature:
                    payload["signature"] = part.signature
            started = (
                timestamp if timing is None else timing.start_of_part(index, timestamp)
            )
            part_time = getattr(part, "timestamp", None) or started
            rows.append(
                self._new_record(
                    self._PART_KINDS.get(discriminator, discriminator),
                    part_time,
                    **payload,
                )
            )
        return rows, links

    def add_run_failed(self, detail: str | None) -> None:
        record = self._new_record(
            "run_failed",
            datetime.now(timezone.utc),
            content={"detail": detail or "unknown error"},
        )
        self._commit([record])

    def add_max_hops_finalized(self) -> None:
        record = self._new_record(
            "run_status",
            datetime.now(timezone.utc),
            content={"status": "max_hops_finalizing"},
        )
        self._commit([record])

    def stamp_total_latency(self, total_latency_ms: int | None) -> None:
        if total_latency_ms is not None and self.events:
            priority = {"user_msg": 0, "run_failed": 1}
            representative = min(self.events, key=lambda row: priority.get(row.kind, 2))
            representative.total_latency_ms = total_latency_ms


def _first_recorded_value(events: list[EventRecord], field_name: str) -> Any:
    for row in events:
        value = getattr(row, field_name)
        if value is not None:
            return value
    return None


@dataclass
class CallTrace:
    """Read one call's output and tool returns without summing repeated usage."""

    call_id: str
    events: list[EventRecord] = field(default_factory=list)

    @property
    def is_hop(self) -> bool:
        return any(row.is_response for row in self.events)

    @property
    def model(self) -> str | None:
        return _first_recorded_value(self.events, "model")

    @property
    def provider(self) -> str | None:
        return _first_recorded_value(self.events, "provider")

    @property
    def request_prep_ms(self) -> int | None:
        return _first_recorded_value(self.events, "request_prep_ms")

    @property
    def ttft_ms(self) -> int | None:
        return _first_recorded_value(self.events, "ttft_ms")

    @property
    def tool_wait_ms(self) -> int | None:
        return _first_recorded_value(self.events, "tool_wait_ms")

    @property
    def generation_ms(self) -> int:
        measured = [row.latency_ms for row in self.events if row.is_response]
        return sum(duration for duration in measured if duration is not None)

    @property
    def total_ms(self) -> int:
        phases = (
            self.request_prep_ms,
            self.ttft_ms,
            self.generation_ms,
            self.tool_wait_ms,
        )
        return sum(duration for duration in phases if duration is not None)

    @property
    def tool_call_count(self) -> int:
        return len([row for row in self.events if row.kind == "tool_call"])

    @property
    def input_tokens(self) -> int | None:
        return _first_recorded_value(self.events, "input_tokens")

    @property
    def output_tokens(self) -> int | None:
        return _first_recorded_value(self.events, "output_tokens")

    @property
    def cache_read_tokens(self) -> int | None:
        return _first_recorded_value(self.events, "cache_read_tokens")

    @property
    def cache_write_tokens(self) -> int | None:
        return _first_recorded_value(self.events, "cache_write_tokens")


@dataclass
class TurnTrace:
    """Group stored rows into model calls and turn-level or orphaned events."""

    message_id: str | None = None
    session_id: str | None = None
    agent_name: str | None = None
    events: list[EventRecord] = field(default_factory=list)
    hops: list[CallTrace] = field(default_factory=list)
    loose_events: list[EventRecord] = field(default_factory=list)

    @classmethod
    def from_docs(cls, docs: list[dict[str, Any]]) -> TurnTrace:
        records = []
        for document in docs:
            records.append(EventRecord.from_doc(document))
        return cls.from_events(records)

    @classmethod
    def from_events(cls, events: list[EventRecord]) -> TurnTrace:
        grouped: dict[str, list[EventRecord]] = {}
        for row in events:
            if row.call_id:
                grouped.setdefault(row.call_id, []).append(row)
        calls = [CallTrace(identifier, rows) for identifier, rows in grouped.items()]
        hops = [call for call in calls if call.is_hop]
        admitted = {call.call_id for call in hops}
        loose = [row for row in events if row.call_id not in admitted]
        if any(row.call_id for row in loose):
            loose.sort(key=lambda row: row.seq)
        identity = {}
        for column in ("message_id", "session_id", "agent_name"):
            identity[column] = _first_recorded_value(events, column)
        return cls(events=list(events), hops=hops, loose_events=loose, **identity)

    def is_empty(self) -> bool:
        return len(self.events) == 0

    @property
    def wall_latency_ms(self) -> int | None:
        roots = [
            row
            for row in self.events
            if row.metadata and row.metadata.get("root") is True
        ]
        return _first_recorded_value(
            roots if roots else self.events, "total_latency_ms"
        )

    @property
    def retry_count(self) -> int:
        return len([row for row in self.events if row.kind == "retry_prompt"])

    @property
    def hit_max_hops(self) -> bool:
        statuses = [
            row.content
            for row in self.events
            if row.kind == "run_status" and isinstance(row.content, dict)
        ]
        return any(status.get("status") == "max_hops_finalizing" for status in statuses)

    def tools_used(self) -> dict[str, list[EventRecord]]:
        tool_rows = [
            row
            for row in self.events
            if row.tool_name and row.kind in {"tool_call", "tool_return"}
        ]
        grouped = {}
        for row in tool_rows:
            if row.tool_name not in grouped:
                grouped[row.tool_name] = []
            grouped[row.tool_name].append(row)
        return grouped
