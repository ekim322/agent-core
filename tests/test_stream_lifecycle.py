"""Closing a public stream releases node observation before persistence."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from agent_core.agent.base import BaseAgent
from agent_core.streaming import StreamEvent
from agent_core.utils.telemetry.run_state import RunState


@pytest.mark.parametrize('max_hops', [0, 2], ids=['finalizer', 'ordinary'])
def test_close_during_text_releases_run_binding_before_persistence(monkeypatch, max_hops):
    lifecycle = []

    class TransportRun:
        def __init__(self, run_id):
            self.ctx = SimpleNamespace(state=SimpleNamespace(run_id=run_id))

        async def __aiter__(self):
            yield 'model-request'

        def all_messages(self):
            return []

    @asynccontextmanager
    async def iter_run(**kwargs):
        run_id = 'finalizer' if kwargs['user_prompt'] != 'Research' else 'ordinary'
        try:
            yield TransportRun(run_id)
        finally:
            lifecycle.append('transport closed')

    class ObservedAgent(BaseAgent):
        async def _handle_node(self, node, agent_run, state):
            try:
                yield StreamEvent.text_delta('Partial answer')
            finally:
                lifecycle.append('node closed')

        async def _save_messages(self, **kwargs):
            state = kwargs['state']
            expected_run = 'finalizer' if max_hops == 0 else 'ordinary'
            assert kwargs['agent_run'].ctx.state.run_id == expected_run
            assert state.run_ids
            assert all(RunState.for_run(run_id) is None for run_id in state.run_ids)
            assert 'node closed' in lifecycle
            lifecycle.append('persisted')

    async def run():
        agent = ObservedAgent(model=TestModel())
        monkeypatch.setattr(agent._agent, 'iter', iter_run)
        monkeypatch.setattr(Agent, 'is_model_request_node', lambda node: node == 'model-request')
        stream = agent.stream('Research', max_hops=max_hops, persist_in_background=False)
        try:
            async for event in stream:
                if event.type == 'text_delta':
                    break
            else:
                pytest.fail('Expected a streamed text delta')
        finally:
            await stream.aclose()
        expected = ['node closed', 'transport closed', 'persisted']
        if max_hops == 0:
            # Release the capped run before the final answer acquires resources.
            expected.insert(0, 'transport closed')
        assert lifecycle == expected

    asyncio.run(run())


@pytest.mark.parametrize('fails', [False, True])
def test_closing_after_done_preserves_terminal_persistence(fails):
    from pydantic_ai.models.function import FunctionModel
    from agent_core.persistence import InMemoryChatWriter

    async def check():
        async def model_stream(messages, info):
            if fails:
                raise ValueError('original model failure')
            yield 'Completed answer'
        writer = InMemoryChatWriter()
        agent = BaseAgent(FunctionModel(stream_function=model_stream), writer=writer)
        stream = agent.stream('Question', persist_in_background=False)
        try:
            async for event in stream:
                if event.type == 'done':
                    break
        finally:
            await stream.aclose()
        failures = [row for row in writer.all_events if row.kind == 'run_failed']
        if fails:
            assert len(failures) == 1
            assert failures[0].content == {'detail': 'original model failure'}
        else:
            assert failures == []
            assert any(row.kind == 'text' and row.content == 'Completed answer' for row in writer.all_events)
    asyncio.run(check())
