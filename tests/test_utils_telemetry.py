"""Task-local timing scopes and opt-in inspector retention contracts."""

import asyncio
import json

import pytest
from pydantic_ai.models.test import TestModel

from agent_core import BaseAgent
from agent_core.streaming import DoneReason, StreamEvent
from agent_core.persistence.records import validate_execution_spans
from agent_core.utils.telemetry.live_trace import LiveTrace, current_trace
from agent_core.utils.telemetry.run_state import RunState


def test_nested_timing_scope_restores_previous_state_and_tasks_are_isolated():
    async def check():
        first, second = RunState(), RunState()
        ready, inspect = asyncio.Event(), asyncio.Event()

        async def other():
            with second.bind_to_run("same"):
                ready.set()
                await inspect.wait()
                assert RunState.for_run("same") is second

        with first.bind_to_run("same"):
            with second.bind_to_run("same"):
                assert RunState.for_run("same") is second
            assert RunState.for_run("same") is first
            task = asyncio.create_task(other())
            await ready.wait()
            assert RunState.for_run("same") is first
            inspect.set()
            await task
        assert RunState.for_run("same") is None

    asyncio.run(check())


def test_retention_caps_do_not_interrupt_execution_or_break_tree_validity():
    trace = LiveTrace(max_spans=2, max_capture_chars=8, max_text_chars=4)
    with trace.bind(), trace.span("agent", "Root", {"large": "payload"}) as root:
        trace.observe(root, StreamEvent.run_status("preparing", extra="oversized"))
        trace.observe(root, StreamEvent.text_delta("abcdefgh"))
        with trace.span("tool", "Child", None) as child:
            with trace.span("model", "Omitted", None) as skipped:
                assert skipped is None
        trace.observe(
            root, StreamEvent.done(DoneReason.COMPLETE, output="oversized output")
        )
    assert root.text == "abcd" and root.output == "[truncated]"
    assert root.events[0]["payload"] == "[truncated]"
    assert len(trace.spans) == 2 and trace.truncated
    assert current_trace() is None
    validate_execution_spans(trace.spans)


def test_small_capture_budget_is_safe_in_real_agent_status_events():
    async def check():
        trace = LiveTrace(max_capture_chars=1)
        with trace.bind():
            result = await BaseAgent(TestModel(custom_output_text="answer")).run(
                "hello", persist=False
            )
        assert result.output == "answer" and trace.truncated
        assert all(row.status == "complete" for row in trace.spans)

    asyncio.run(check())


def test_secrets_redact_keys_short_values_and_unfinished_stream_prefixes():
    trace = LiveTrace(("short", "credential-12345"))
    assert trace.capture({"short": "short", "api_key": "anything"}) == {
        "[redacted]": "[redacted]",
        "api_key": "[redacted]",
    }
    with pytest.raises(ValueError):
        with trace.bind(), trace.span("agent", "short", None) as row:
            trace.observe(row, StreamEvent.text_delta("Visible credential-"))
            raise ValueError("secret details must not be captured")
    assert row.text == "Visible [redacted]" and row.status == "error"
    assert "credential-" not in json.dumps(row.model_dump(mode="json"))
    assert row.name == "[redacted]" and row.output == {"error": "ValueError"}


def test_default_capture_remains_complete_and_tool_batch_timing_uses_extremes():
    trace = LiveTrace()
    payload = "x" * 200000
    assert trace.capture(payload) == payload and not trace.truncated
    state = RunState()
    state.start_model_call()
    call = state.current_call
    call.note_tool_started(10)
    call.note_tool_started(8)
    call.note_tool_finished(15)
    call.note_tool_finished(12)
    assert call.tool_wait_ms == 7000


def test_shared_prefix_secrets_are_redacted_independently_of_chunk_boundaries():
    text = "Before credential-12345 after"
    for boundary in range(len(text) + 1):
        trace = LiveTrace(("credential", "credential-12345"))
        with trace.bind(), trace.span("agent", "Root", None) as row:
            trace.observe(row, StreamEvent.text_delta(text[:boundary]))
            trace.observe(row, StreamEvent.text_delta(text[boundary:]))
        assert row.text == "Before [redacted] after", boundary


def test_redaction_does_not_reprocess_markers_as_secret_values():
    trace = LiveTrace(("private", "red"))
    assert trace.redact_text("private red") == "[redacted] [redacted]"
    with trace.bind(), trace.span("agent", "Root", None) as row:
        trace.observe(row, StreamEvent.text_delta("private "))
        trace.observe(row, StreamEvent.text_delta("red"))
    assert row.text == "[redacted] [redacted]"


def test_sensitive_payload_keys_accept_common_separator_and_case_variants():
    keys = ["API Key", "api.key", "Access Token", "refresh-token", "private_key"]
    trace = LiveTrace()
    payload = {key: {"nested": "credential"} for key in keys}
    payload["description"] = "visible"
    captured = trace.capture(payload)
    assert captured == {**{key: "[redacted]" for key in keys}, "description": "visible"}
    assert payload["API Key"] == {"nested": "credential"}


def test_nested_text_streams_keep_separate_prefixes_and_finish_once():
    trace = LiveTrace(("credential-12345",))
    with trace.bind(), trace.span("agent", "Root", None) as parent:
        trace.observe(parent, StreamEvent.text_delta("credential-"))
        with trace.span("tool", "Child", None) as child:
            trace.observe(child, StreamEvent.text_delta("12345"))
        assert child.text == "12345"
        trace.observe(parent, StreamEvent.text_delta("12345"))
        trace.observe(parent, StreamEvent.text_delta(" then credential-"))
        trace.observe(parent, StreamEvent.done(DoneReason.COMPLETE))
        assert parent.text == "[redacted] then [redacted]"
    assert parent.text == "[redacted] then [redacted]"


@pytest.mark.parametrize("secret", ["x", "[key].*", "\\token", "密钥"])
def test_secret_values_are_literal_and_safe_when_streamed_character_by_character(secret):
    trace = LiveTrace((secret,))
    assert trace.redact_text(f"Before {secret} end") == "Before [redacted] end"
    with trace.bind(), trace.span("agent", "Root", None) as row:
        for character in f"Before {secret} end":
            trace.observe(row, StreamEvent.text_delta(character))
    assert row.text == "Before [redacted] end"
