"""Consumers can extend the public agent without application infrastructure."""
import asyncio
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool

from agent_core import BaseAgent, EventType, InMemoryChatWriter, PreparedRequest


@dataclass
class Dependencies:
    prefix: str


@dataclass
class Request:
    prompt: str
    prefix: str


def test_subclass_request_tools_and_persistence_work_for_run_and_stream():
    received = []

    async def greet(ctx: RunContext[Dependencies], name: str) -> str:
        received.append(ctx.deps.prefix)
        return f'{ctx.deps.prefix} {name}'

    class GreetingAgent(BaseAgent[Dependencies, Request]):
        def __init__(self, writer):
            super().__init__(model=TestModel(custom_output_text='Finished'),
                deps_type=Dependencies, tools=[Tool(greet)], writer=writer)

        async def _prepare_run(self, request):
            return PreparedRequest(user_msg=request.prompt,
                deps=Dependencies(request.prefix), session_id='conversation',
                message_id='message')

    async def check():
        writer = InMemoryChatWriter()
        agent = GreetingAgent(writer)
        result = await agent.run(request=Request('Greet someone', 'Hello'),
            persist_in_background=False)
        assert result.output == 'Finished'
        assert received == ['Hello']
        assert writer.batches and writer.all_events
        events = [event async for event in agent.stream(
            request=Request('Greet someone', 'Welcome'), persist_in_background=False)]
        assert received == ['Hello', 'Welcome']
        assert any(event.type == EventType.TOOL_CALL for event in events)
        assert events[-1].type == EventType.DONE
        assert events[-1].data['output'] == 'Finished'
        assert len(writer.batches) == 2

    asyncio.run(check())


def test_request_overrides_inherit_only_when_none():
    """An empty value still overrides a request's prepared value."""
    from agent_core import ModelConfig
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    prepared_model = TestModel(custom_output_text='Prepared')
    override_model = TestModel(custom_output_text='Override')
    prepared_config = ModelConfig(thinking='high')
    override_config = ModelConfig(thinking=False)
    prepared = PreparedRequest(
        user_msg='prepared prompt', session_id='session', message_id='message',
        user_id='user', metadata={'source': 'request'}, instructions='guidance',
        deps=Dependencies('prepared'), model=prepared_model, config=prepared_config,
        max_hops=3, message_history=[ModelRequest(parts=[UserPromptPart('earlier')])],
        tools=[Tool(lambda: 'prepared tool', name='prepared_tool')],
    )

    class RequestAgent(BaseAgent):
        async def _prepare_run(self, request):
            assert request == 'request'
            return prepared

    async def check():
        agent = RequestAgent(TestModel())
        inherited = await agent._merge_request('request', PreparedRequest(user_msg=None))
        assert inherited == prepared
        overrides = PreparedRequest(
            user_msg='', session_id='', message_id='', user_id='', metadata={},
            instructions=[], deps=Dependencies(''), model=override_model,
            config=override_config, max_hops=0, message_history=[], tools=[],
        )
        merged = await agent._merge_request('request', overrides)
        assert merged == overrides
        assert merged.model is override_model
        assert merged.config is override_config
        assert prepared.user_msg == 'prepared prompt'
        assert prepared.tools
        unchanged = await agent._merge_request(None, overrides)
        assert unchanged == overrides

    asyncio.run(check())
