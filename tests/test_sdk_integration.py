"""Public SDK configuration and budgets across agent execution phases."""

import asyncio
from contextlib import aclosing

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import ModelRetry, Tool
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import UsageLimits

from agent_core import BaseAgent, EventType, PreparedRequest
from agent_core.agent._execution.recovery import ModelRequestRetryPolicy


def test_constructor_toolsets_close_before_storage_and_allow_replacement():
    lifecycle = []

    def lookup() -> str:
        return "fact"

    class ConnectedTools(FunctionToolset):
        async def __aenter__(self):
            lifecycle.append("opened")
            return self

        async def __aexit__(self, *args):
            lifecycle.append("closed")

    class StoredAgent(BaseAgent):
        async def _save_messages(self, **kwargs):
            assert lifecycle[-1] == "closed"
            lifecycle.append("stored")

    async def check():
        agent = StoredAgent(
            TestModel(call_tools=["lookup"], custom_output_text="answer"),
            toolsets=[ConnectedTools([lookup])],
            agent_name="Research",
        )
        assert agent.agent.name == "Research"
        result = await agent.run("Question")
        assert result.output == "answer"
        assert lifecycle == ["opened", "closed", "stored"]
        result = await agent.run(
            "Question",
            toolsets=[],
            model=TestModel(custom_output_text="no tools"),
            persist=False,
        )
        assert result.output == "no tools"
        assert lifecycle == ["opened", "closed", "stored"]

    asyncio.run(check())


def test_separate_tool_and_output_validation_retry_budgets():
    attempts = 0
    executed = []

    def lookup(value: int) -> str:
        executed.append(value)
        return "fact"

    class Answer(BaseModel):
        value: int

    async def respond(messages, info):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield {0: DeltaToolCall(name="lookup", json_args='{"value":"invalid"}')}
        elif attempts == 2:
            yield {0: DeltaToolCall(name="lookup", json_args='{"value":1}')}
        else:
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name, json_args='{"value":"invalid"}'
                )
            }

    async def check():
        agent = BaseAgent(
            FunctionModel(stream_function=respond),
            tools=[Tool(lookup)],
            retries={"tools": 1, "output": 0},
        )
        with pytest.raises(UnexpectedModelBehavior):
            await agent.run("Question", output_type=Answer, persist=False)
        assert attempts == 3
        assert executed == [1]

    asyncio.run(check())


@pytest.mark.parametrize("tool_retries", [0, 1])
def test_tool_requested_retry_reaches_sdk_budget(tool_retries):
    calls = 0

    def lookup() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry("Try the lookup again")
        return "fact"

    async def check():
        agent = BaseAgent(
            TestModel(call_tools=["lookup"], custom_output_text="answer"),
            tools=[Tool(lookup)],
            retries={"tools": tool_retries, "output": 0},
        )
        if tool_retries == 0:
            with pytest.raises(UnexpectedModelBehavior):
                await agent.run("Question", persist=False)
            assert calls == 1
        else:
            result = await agent.run("Question", persist=False)
            assert result.output == "answer"
            assert calls == 2

    asyncio.run(check())


@pytest.mark.parametrize("request_limit", [1, 2])
def test_request_budget_includes_final_answer(request_limit):
    requests = []
    executed = []

    def lookup() -> str:
        executed.append("lookup")
        return "fact"

    async def respond(messages, info):
        requests.append(bool(info.function_tools))
        if info.function_tools:
            yield {0: DeltaToolCall(name="lookup", json_args="{}")}
        else:
            yield "answer"

    async def check():
        agent = BaseAgent(
            FunctionModel(stream_function=respond),
            tools=[Tool(lookup)],
            max_concurrency=1,
        )
        call = agent.run(
            "Question",
            max_hops=1,
            usage_limits=UsageLimits(request_limit=request_limit),
            persist=False,
        )
        if request_limit == 1:
            with pytest.raises(UsageLimitExceeded):
                await call
            assert requests == [True]
        else:
            result = await call
            assert result.output == "answer"
            assert result.usage.requests == 2
            assert requests == [True, False]
        assert executed == ["lookup"]

    asyncio.run(check())


def test_default_hop_limit_can_finalize_after_fifty_model_turns():
    requests = 0

    def lookup() -> str:
        return "fact"

    async def respond(messages, info):
        nonlocal requests
        requests += 1
        if info.function_tools:
            yield {0: DeltaToolCall(name="lookup", json_args="{}")}
        else:
            yield "answer"

    async def check():
        agent = BaseAgent(FunctionModel(stream_function=respond), tools=[Tool(lookup)])
        result = await agent.run("Question", persist=False)
        assert result.output == "answer"
        assert result.usage.requests == requests == 51

    asyncio.run(check())


def test_recovery_keeps_request_budget_and_counts_interrupted_request(monkeypatch):
    monkeypatch.setattr(ModelRequestRetryPolicy, "_delay_seconds", lambda *args: 0)
    requests = 0

    async def respond(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield "partial "
            raise httpx2.ReadTimeout("interrupted")
        yield "answer"

    async def check():
        agent = BaseAgent(FunctionModel(stream_function=respond))
        with pytest.raises(UsageLimitExceeded):
            await agent.run(
                "Question",
                usage_limits=UsageLimits(request_limit=1),
                persist=False,
            )
        assert requests == 1

    asyncio.run(check())


def test_recovery_keeps_hop_count_and_success_usage(monkeypatch):
    monkeypatch.setattr(ModelRequestRetryPolicy, "_delay_seconds", lambda *args: 0)
    requests = []
    calls = 0

    def lookup() -> str:
        nonlocal calls
        calls += 1
        return "fact"

    async def respond(messages, info):
        requests.append(bool(info.function_tools))
        if len(requests) == 1:
            yield {0: DeltaToolCall(name="lookup", json_args="{}")}
        elif len(requests) == 2:
            yield "partial "
            raise httpx2.ReadTimeout("interrupted")
        else:
            yield "answer"

    async def check():
        agent = BaseAgent(
            FunctionModel(stream_function=respond),
            tools=[Tool(lookup)],
            max_hops=2,
            max_concurrency=1,
        )
        result = await agent.run("Question", persist=False)
        assert result.output == "partial answer"
        assert result.usage.requests == 3
        assert requests == [True, True, False]
        assert calls == 1

    asyncio.run(check())


def test_prepared_token_budget_can_be_overridden_and_isolated_between_calls():
    class RequestAgent(BaseAgent):
        async def _prepare_run(self, request):
            return PreparedRequest(
                user_msg=request,
                usage_limits=UsageLimits(output_tokens_limit=0),
            )

    async def check():
        agent = RequestAgent(TestModel(custom_output_text="answer"))
        events = []
        async with aclosing(agent.stream(request="Question", persist=False)) as stream:
            async for event in stream:
                events.append(event)
        assert events[-1].type is EventType.DONE
        assert isinstance(events[-1].data["exception"], UsageLimitExceeded)
        first, second = await asyncio.gather(
            *(
                agent.run(
                    request="Question",
                    usage_limits=UsageLimits(request_limit=1),
                    persist=False,
                )
                for _ in range(2)
            )
        )
        assert first.output == second.output == "answer"
        assert first.usage.requests == second.usage.requests == 1
        assert first.usage is not second.usage

    asyncio.run(check())
