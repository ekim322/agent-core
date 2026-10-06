"""Execution contracts across tool phases, finalization and caller cleanup."""

import asyncio
from contextlib import aclosing

import httpx2
import pytest
from pydantic_ai import Tool
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from agent_core import BaseAgent, EventType, InMemoryChatWriter
from agent_core.agent._execution.recovery import ModelRequestRetryPolicy


def test_hop_finalization_resolves_tool_history_and_combines_usage():
    requests = []
    calls = []

    async def lookup():
        calls.append("lookup")
        return "captured fact"

    async def respond(messages, info):
        requests.append((messages, [tool.name for tool in info.function_tools]))
        if info.function_tools:
            yield {0: DeltaToolCall(name="lookup", json_args="{}")}
        else:
            assert any(
                part.part_kind == "tool-return" and part.content == "captured fact"
                for message in messages
                for part in message.parts
            )
            yield "Final answer"

    async def check():
        writer = InMemoryChatWriter()
        agent = BaseAgent(
            FunctionModel(stream_function=respond), tools=[Tool(lookup)], writer=writer
        )
        events = [event async for event in agent.stream("Research", max_hops=1)]
        assert calls == ["lookup"]
        assert [names for _, names in requests] == [["lookup"], []]
        assert sum(event.type == EventType.DONE for event in events) == 1
        terminal = events[-1]
        assert terminal.data["output"] == "Final answer"
        assert terminal.data["usage"].requests == 2
        assert terminal.data["max_hops_finalized"] is True
        assert [row.kind for row in writer.all_events].count("run_status") == 1
        assert not any(row.kind == "run_failed" for row in writer.all_events)

    asyncio.run(check())


@pytest.mark.parametrize("recover", [True, False])
def test_final_phase_recovery_preserves_tool_work_and_original_failure(
    monkeypatch, recover
):
    attempts = 0
    calls = 0
    original = httpx2.ReadTimeout("final response interrupted")
    monkeypatch.setattr(ModelRequestRetryPolicy, "_delay_seconds", lambda *args: 0)

    async def lookup():
        nonlocal calls
        calls += 1
        return "fact"

    async def respond(messages, info):
        nonlocal attempts
        if info.function_tools:
            yield {0: DeltaToolCall(name="lookup", json_args="{}")}
            return
        attempts += 1
        if recover and attempts == 2:
            yield "complete"
            return
        yield "partial "
        raise original

    async def check():
        writer = InMemoryChatWriter()
        agent = BaseAgent(
            FunctionModel(stream_function=respond),
            tools=[Tool(lookup)],
            writer=writer,
            model_request_retries=1,
        )
        if recover:
            result = await agent.run("Research", max_hops=1)
            assert result.output == "partial complete"
        else:
            with pytest.raises(httpx2.ReadTimeout) as caught:
                await agent.run("Research", max_hops=1)
            assert caught.value is original
        assert attempts == 2
        assert calls == 1
        assert any(row.kind == "tool_return" for row in writer.all_events)
        assert sum(row.kind == "run_failed" for row in writer.all_events) == (
            0 if recover else 1
        )

    asyncio.run(check())


def test_concurrent_replacement_and_additive_tools_are_isolated():
    entered = 0
    both_ready = asyncio.Event()
    seen = []

    def tool():
        return "value"

    async def respond(messages, info):
        nonlocal entered
        names = sorted(item.name for item in info.function_tools)
        seen.append((names, info.instructions))
        if names != ["base"]:
            entered += 1
            if entered == 2:
                both_ready.set()
            await asyncio.wait_for(both_ready.wait(), 2)
        yield ",".join(names)

    async def check():
        base = Tool(tool, name="base")
        agent = BaseAgent(
            FunctionModel(stream_function=respond), tools=[base], system_prompt="common"
        )
        additive, replacement = await asyncio.gather(
            agent.run(
                "A",
                extra_tools=[Tool(tool, name="alpha"), Tool(tool, name="base")],
                instructions="alpha guidance",
                persist=False,
            ),
            agent.run(
                "B",
                tools=[Tool(tool, name="beta")],
                instructions="beta guidance",
                persist=False,
            ),
        )
        assert additive.output == "alpha,base"
        assert replacement.output == "beta"
        assert (await agent.run("C", persist=False)).output == "base"
        for names, instructions in seen:
            if "alpha" in names:
                assert (
                    "alpha guidance" in instructions
                    and "beta guidance" not in instructions
                )
            elif "beta" in names:
                assert (
                    "beta guidance" in instructions
                    and "alpha guidance" not in instructions
                )
            else:
                assert instructions == "common"
        assert base.function is tool

    asyncio.run(check())


def test_reasoning_only_interruption_is_not_replayed():
    attempts = 0
    failure = httpx2.ReadTimeout("reasoning interrupted")

    async def respond(messages, info):
        nonlocal attempts
        attempts += 1
        yield {0: DeltaThinkingPart(content="thinking")}
        raise failure

    async def check():
        agent = BaseAgent(FunctionModel(stream_function=respond))
        events = [event async for event in agent.stream("Question", persist=False)]
        assert [event.type for event in events] == [
            EventType.REASONING_DELTA,
            EventType.DONE,
        ]
        assert events[-1].data["exception"] is failure
        assert not any(event.type == EventType.RUN_STATUS for event in events)
        assert attempts == 1

    asyncio.run(check())


def test_awaited_storage_failure_follows_terminal_event():
    failure = RuntimeError("storage unavailable")

    class FailedWriter(InMemoryChatWriter):
        async def write(self, records):
            raise failure

    async def check():
        agent = BaseAgent(TestModel(custom_output_text="answer"), writer=FailedWriter())
        events = []
        with pytest.raises(RuntimeError) as caught:
            async with aclosing(agent.stream("Question")) as stream:
                async for event in stream:
                    events.append(event)
        assert caught.value is failure
        assert sum(event.type == EventType.DONE for event in events) == 1
        assert events[-1].data["output"] == "answer"
        with pytest.raises(RuntimeError) as caught:
            await agent.run("Question")
        assert caught.value is failure

    asyncio.run(check())
