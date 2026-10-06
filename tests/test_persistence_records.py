"""Stored call linkage and trace reads preserve order and measured values."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agent_core.persistence import CallTrace, EventRecord, RunRecords, TurnTrace
from agent_core.utils.telemetry.run_state import ModelCallTiming, RunState


def test_span_validation_reports_duplicate_and_unrecorded_parent_identities():
    import pytest
    from agent_core.persistence.records import ExecutionSpan, validate_execution_spans

    def span(identity, parent=None):
        return ExecutionSpan(
            id=identity, parent_id=parent, kind="agent", name="Run", started_ms=0
        )

    validate_execution_spans([span("root"), span("child", "root")])
    with pytest.raises(ValueError, match="Duplicate.*root"):
        validate_execution_spans([span("root"), span("root")])
    with pytest.raises(ValueError, match="child.*root"):
        validate_execution_spans([span("child", "root"), span("root")])
    with pytest.raises(ValueError, match="child.*absent"):
        validate_execution_spans([span("child", "absent")])


def test_reused_tool_ids_keep_response_linkage_and_do_not_inherit_old_timings():
    first = ModelResponse(parts=[ToolCallPart("lookup", {}, "reused")])
    second = ModelResponse(parts=[ToolCallPart("lookup", {}, "reused")])
    state = RunState(
        calls=[ModelCallTiming(response=first, tool_durations={"reused": 25})]
    )
    records = RunRecords(message_id="message")
    records.add_run(
        [
            ModelRequest(parts=[UserPromptPart("question")]),
            first,
            ModelRequest(parts=[ToolReturnPart("lookup", "first", "reused")]),
            second,
            ModelRequest(parts=[ToolReturnPart("lookup", "second", "reused")]),
        ],
        state,
    )
    calls = [row for row in records.events if row.kind == "tool_call"]
    returns = [row for row in records.events if row.kind == "tool_return"]
    assert calls[0].call_id != calls[1].call_id
    assert [row.call_id for row in returns] == [row.call_id for row in calls]
    assert [row.latency_ms for row in returns] == [25, None]
    trace = TurnTrace.from_docs([row.to_doc() for row in records.events])
    assert [hop.call_id for hop in trace.hops] == [row.call_id for row in calls]
    assert trace.tools_used()["lookup"] == [calls[0], returns[0], calls[1], returns[1]]


def test_trace_reads_first_non_null_values_without_double_counting_usage():
    events = [
        EventRecord(kind="user_msg", seq=0, message_id="", total_latency_ms=0),
        EventRecord(
            kind="text",
            call_id="call",
            seq=1,
            model="",
            input_tokens=8,
            ttft_ms=0,
            latency_ms=3,
        ),
        EventRecord(
            kind="tool_call",
            call_id="call",
            seq=2,
            model="later",
            input_tokens=8,
            ttft_ms=20,
            latency_ms=5,
        ),
        EventRecord(kind="tool_return", call_id="call", seq=3, latency_ms=25),
        EventRecord(kind="retry_prompt", call_id="orphan", seq=4),
    ]
    trace = TurnTrace.from_events(events)
    assert trace.message_id == "" and trace.wall_latency_ms == 0
    assert trace.retry_count == 1
    assert [event.seq for event in trace.loose_events] == [0, 4]
    call = trace.hops[0]
    assert call.model == "" and call.ttft_ms == 0
    assert call.input_tokens == 8 and call.generation_ms == call.total_ms == 8
    assert CallTrace("empty").input_tokens is None


def test_failed_message_projection_commits_neither_rows_nor_tool_links(caplog):
    from pydantic_ai.messages import TextPart

    class Unstorable:
        def __str__(self):
            raise ValueError("cannot convert")

    good = ModelResponse(parts=[ToolCallPart("lookup", {}, "same")])
    broken = ModelResponse(
        parts=[
            ToolCallPart("lookup", {}, "same"),
            ToolCallPart("other", Unstorable(), "broken"),
        ]
    )
    records = RunRecords()
    records.add_run(
        [
            ModelRequest(parts=[UserPromptPart("question")]),
            good,
            broken,
            ModelRequest(parts=[ToolReturnPart("lookup", "answer", "same")]),
            ModelResponse(parts=[TextPart("finished")]),
        ]
    )
    assert [row.kind for row in records.events] == [
        "user_msg",
        "tool_call",
        "tool_return",
        "text",
    ]
    assert records.events[1].call_id == records.events[2].call_id
    assert [row.seq for row in records.events] == list(range(4))
    assert any(
        getattr(record, "event_name", None) == "agent.persistence_record_failed"
        for record in caplog.records
    )


def test_storage_snapshots_recursive_payloads_and_summarizes_binary_content():
    from pydantic_ai.messages import BinaryContent
    from agent_core.persistence.records import payload_word_count

    payload = {"two_words": ["hello world"]}
    payload["loop"] = payload
    records = RunRecords(metadata=payload)
    records.add_run(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [
                            "question",
                            BinaryContent(
                                data=b"private bytes", media_type="image/png"
                            ),
                        ]
                    )
                ]
            )
        ]
    )
    assert records.metadata == {"two_words": ["hello world"], "loop": "[circular]"}
    assert payload_word_count(payload) == 5
    assert payload_word_count('[{"two_words": "hello world"}]') == 4
    assert payload_word_count(None) is None
    assert payload_word_count(["a", "a"]) == 2
    summary = records.events[0].content[1]
    assert summary["kind"] == "binary" and summary["size"] == 13
    assert "data" not in summary
    payload["two_words"].append("later")
    assert records.metadata["two_words"] == ["hello world"]


def test_request_projection_keeps_first_instructions_and_only_followups_after_output():
    from pydantic_ai.messages import SystemPromptPart, TextPart, RetryPromptPart

    records = RunRecords()
    records.add_run(
        [
            ModelRequest(
                instructions="instructions",
                parts=[SystemPromptPart("duplicate"), UserPromptPart("question")],
            ),
            ModelResponse(parts=[TextPart("answer")]),
            ModelRequest(
                instructions="recovery",
                parts=[UserPromptPart("continue"), RetryPromptPart("correct")],
            ),
        ]
    )
    assert [row.kind for row in records.events] == [
        "system_prompt",
        "user_msg",
        "text",
        "retry_prompt",
    ]
    assert records.events[0].content == "instructions"
    records.stamp_total_latency(0)
    assert records.events[1].total_latency_ms == 0
    assert records.events[0].total_latency_ms is None


def test_message_selection_retains_unrecorded_output_after_invalid_boundary(caplog):
    from types import SimpleNamespace
    from pydantic_ai.messages import TextPart
    from agent_core.persistence.messages import messages_for_persistence

    recorded = ModelResponse(parts=[TextPart("recorded")])
    emitted = ModelResponse(parts=[TextPart("partial")])
    source = SimpleNamespace(
        all_messages=lambda: [recorded], new_messages=lambda: [recorded]
    )
    state = SimpleNamespace(
        turn_start_index=lambda messages: None,
        unrecorded_messages=[recorded, emitted, emitted],
    )
    selected = messages_for_persistence(
        result=source, agent_run=None, start_index=10, agent_name="example", state=state
    )
    assert selected == [recorded, emitted]
    assert any(
        getattr(record, "event_name", None) == "agent.persistence_slice_invalid"
        for record in caplog.records
    )
    assert messages_for_persistence(
        result=None, agent_run=None, start_index=10, agent_name="example", state=state
    ) == [recorded, emitted]


def test_message_selection_uses_measured_boundary_when_sdk_merges_history():
    from types import SimpleNamespace
    from agent_core.persistence.messages import messages_for_persistence

    history = [
        ModelRequest(parts=[UserPromptPart("old")]),
        ModelRequest(parts=[UserPromptPart("current")]),
    ]
    source = SimpleNamespace(all_messages=lambda: history)
    state = SimpleNamespace(turn_start_index=lambda messages: 1, unrecorded_messages=[])
    assert (
        messages_for_persistence(
            result=source,
            agent_run=None,
            start_index=2,
            agent_name="example",
            state=state,
        )
        == history[1:]
    )


def test_event_documents_preserve_storage_fields_and_hide_private_api_fields():
    row = EventRecord(
        id="row", user_id="private", metadata={"secret": "value"}, input_tokens=0
    )
    document = row.to_doc()
    assert document["_id"] == "row" and "id" not in document
    assert EventRecord.from_doc({**document, "id": "ignored", "unknown": 42}) == row
    projection = row.to_api_dict()
    assert projection["id"] == "row" and projection["input_tokens"] == 0
    assert "user_id" not in projection and "metadata" not in projection
