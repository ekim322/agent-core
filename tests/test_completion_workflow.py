"""Completion results, correction history and capacity survive storage boundaries."""

import asyncio
from contextlib import aclosing

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from agent_core import EventType, InMemoryChatWriter, LLMClient, SubCallTrace


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("structured", [False, True])
def test_completion_stores_answer_and_structured_corrections(structured):
    attempts = []

    def respond(messages, info):
        attempts.append(list(messages))
        if structured:
            value = "invalid" if len(attempts) == 1 else 7
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name, {"value": value}, "answer-call"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("answer")])

    async def check():
        writer = InMemoryChatWriter()
        client = LLMClient(FunctionModel(respond), writer=writer)
        messages = [ModelRequest(parts=[UserPromptPart("question")])]
        output = await client.run(
            messages,
            output_type=Answer if structured else None,
            session_id="session",
            message_id="message",
            user_id="user",
            trace_metadata=SubCallTrace(parent_tool_call_id="parent"),
        )
        if structured:
            assert output == Answer(value=7)
            assert len(attempts) == 2
            assert any(row.kind == "retry_prompt" for row in writer.all_events)
        else:
            assert isinstance(output, ModelResponse)
            assert output.parts[0].content == "answer"
            assert len(attempts) == 1
        assert len(writer.batches) == 1
        assert writer.all_events[0].content == "question"
        assert all(
            row.session_id == "session"
            and row.message_id == "message"
            and row.user_id == "user"
            for row in writer.all_events
        )
        assert writer.batches[0].metadata["parent_tool_call_id"] == "parent"
        assert len(messages) == 1

    asyncio.run(check())


@pytest.mark.parametrize("structured", [False, True])
def test_completion_writer_failure_releases_client_capacity(structured):
    failure = RuntimeError("writer unavailable")

    class FailedWriter(InMemoryChatWriter):
        async def write(self, records):
            raise failure

    def respond(messages, info):
        if structured:
            return ModelResponse(
                parts=[
                    ToolCallPart(info.output_tools[0].name, {"value": 7}, "answer-call")
                ]
            )
        return ModelResponse(parts=[TextPart("answer")])

    async def check():
        client = LLMClient(
            FunctionModel(respond), writer=FailedWriter(), max_concurrency=1
        )
        messages = [ModelRequest(parts=[UserPromptPart("question")])]
        with pytest.raises(RuntimeError) as caught:
            await client.run(messages, output_type=Answer if structured else None)
        assert caught.value is failure
        output = await asyncio.wait_for(
            client.run(
                messages, output_type=Answer if structured else None, persist=False
            ),
            2,
        )
        assert isinstance(output, Answer if structured else ModelResponse)

    asyncio.run(check())


def test_completion_capacity_includes_storage_and_cancellation_releases_it():
    async def check():
        storing = asyncio.Event()
        model_calls = []

        class BlockingWriter(InMemoryChatWriter):
            async def write(self, records):
                storing.set()
                await asyncio.Event().wait()

        def respond(messages, info):
            model_calls.append("requested")
            return ModelResponse(parts=[TextPart("answer")])

        client = LLMClient(
            FunctionModel(respond), writer=BlockingWriter(), max_concurrency=1
        )
        messages = [ModelRequest(parts=[UserPromptPart("question")])]
        first = asyncio.create_task(client.run(messages))
        second = None
        try:
            await asyncio.wait_for(storing.wait(), 2)
            second = asyncio.create_task(client.run(messages, persist=False))
            await asyncio.sleep(0)
            assert model_calls == ["requested"]
            assert not first.done() and not second.done()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert isinstance(await asyncio.wait_for(second, 2), ModelResponse)
            assert model_calls == ["requested", "requested"]
        finally:
            first.cancel()
            if second is not None:
                second.cancel()
            await asyncio.gather(
                first, *([second] if second else []), return_exceptions=True
            )

    asyncio.run(check())


def test_completion_stream_closure_releases_capacity_without_storage():
    async def check():
        closed = []

        async def respond(messages, info):
            try:
                yield "first"
                yield "second"
            finally:
                closed.append("closed")

        writer = InMemoryChatWriter()
        client = LLMClient(
            FunctionModel(stream_function=respond), writer=writer, max_concurrency=1
        )
        messages = [ModelRequest(parts=[UserPromptPart("question")])]
        async with aclosing(client.stream(messages)) as stream:
            event = await anext(stream)
            assert event.type is EventType.TEXT_DELTA
        assert closed == ["closed"]

        async def consume():
            return [event async for event in client.stream(messages)]

        events = await asyncio.wait_for(consume(), 2)
        assert "".join(event.data["delta"] for event in events) == "firstsecond"
        assert all(event.type is EventType.TEXT_DELTA for event in events)
        assert closed == ["closed", "closed"]
        assert writer.batches == []

    asyncio.run(check())


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_completion_rejects_invalid_explicit_concurrency(limit):
    with pytest.raises(ValueError, match="max_concurrency"):
        LLMClient(
            FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")])),
            max_concurrency=limit,
        )


def test_completion_override_uses_matching_provider_configuration(monkeypatch):
    from agent_core import ModelConfig
    from agent_core.models.config import ModelRuntime

    selected_settings = []
    translated_names = []
    original_translation = ModelConfig.to_settings

    def translate(config, name):
        translated_names.append(name)
        return original_translation(config, name)

    monkeypatch.setattr(ModelConfig, "to_settings", translate)

    def respond(messages, info):
        selected_settings.append(info.model_settings)
        return ModelResponse(parts=[TextPart("override")])

    override = FunctionModel(respond, model_name="override")
    monkeypatch.setattr(
        ModelRuntime,
        "infer_provider",
        lambda name: "openai" if name == "override" else "custom",
    )
    client = LLMClient(
        FunctionModel(respond, model_name="default"), config=ModelConfig(thinking=False)
    )
    output = asyncio.run(
        client.run(
            [ModelRequest(parts=[UserPromptPart("question")])],
            model=override,
            config=ModelConfig(thinking="high"),
        )
    )
    assert output.parts[0].content == "override"
    assert translated_names == ["override"]
    assert selected_settings[0]["openai_store"] is False
