"""Typed payload construction preserves the extensible public event envelope."""

import json

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RunUsage

from agent_core.models.catalog import ModelId, ReasoningEffort
from agent_core.streaming import DoneReason, EventType, StreamEvent


def test_terminal_payload_retains_sdk_objects_and_explicit_nulls():
    output = {}
    output["recursive"] = output
    messages = [ModelResponse([TextPart("answer")])]
    usage = RunUsage()
    event = StreamEvent.done(DoneReason.COMPLETE, output=output, messages=messages, usage=usage)
    assert event.data["output"] is output
    assert event.data["messages"] is messages
    assert event.data["usage"] is usage
    assert event.data["detail"] is None
    assert StreamEvent.done(DoneReason.CANCELLED).model_dump(mode="json") == {
        "type": "done",
        "data": {"reason": "cancelled", "output": None, "messages": None, "usage": None, "detail": None},
    }


def test_tool_payloads_keep_objects_and_status_allows_arbitrary_keys():
    arguments = {"query": "example"}
    output = object()
    assert StreamEvent.tool_call("search", "call", arguments).data["arguments"] is arguments
    event = StreamEvent.tool_result("search", "call", output, declined=True)
    assert event.data == {"name": "search", "call_id": "call", "output": output, "declined": True}
    status = StreamEvent.run_status("waiting", kind="provider", payload={"retry": 1}, extra="x")
    assert status.data == {"status": "waiting", "kind": "provider", "payload": {"retry": 1}, "extra": "x"}
    status.data["annotation"] = "caller extension"
    assert status.type == EventType.RUN_STATUS


def test_event_factories_construct_subclasses_and_keep_delta_wire_shape():
    class CustomEvent(StreamEvent):
        origin: str = "custom"

    event = CustomEvent.text_delta("visible")
    assert isinstance(event, CustomEvent)
    assert event.model_dump(mode="json") == {
        "type": "text_delta", "data": {"delta": "visible"}, "origin": "custom",
    }
    assert StreamEvent.reasoning_delta("thinking").model_dump(mode="json") == {
        "type": "reasoning_delta", "data": {"delta": "thinking"},
    }


def test_catalog_members_are_strings_with_unambiguous_roundtrips():
    assert ModelId("gpt-6-luna") is ModelId.GPT_6_LUNA
    assert json.loads(json.dumps({"model": ModelId.GPT_6_LUNA, "effort": ReasoningEffort.LOW})) == {
        "model": "gpt-6-luna", "effort": "low",
    }
    assert len(ModelId) == len({member.value for member in ModelId})
