"""Common telemetry explains delegation independently of the inspector."""
import asyncio
from contextlib import nullcontext
import io
import json

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Tool
from pydantic_ai.models.test import TestModel

from agent_core.agent.base import BaseAgent
from agent_core.streaming import DoneReason, StreamEvent
from agent_core.utils.telemetry.execution import traced_agent_stream
from agent_core.utils.telemetry.live_trace import LiveTrace
from observability import Observability, ObservabilitySettings, bind_observability_context
from observability.correlation import current_observability_context
from observability.cli import main


@pytest.fixture
def telemetry():
    output, spans, metrics = io.StringIO(), InMemorySpanExporter(), InMemoryMetricReader()
    runtime = Observability(ObservabilitySettings(), stream=output,
        span_processors=[SimpleSpanProcessor(spans)], metric_readers=[metrics])
    runtime.start()
    try:
        yield output, spans, metrics
    finally:
        runtime.close()


@pytest.mark.parametrize('inspector', [False, True])
def test_delegated_agents_join_tools_and_query_emitted_logs(telemetry, inspector, tmp_path, capsys):
    output, exporter, metrics = telemetry
    trace = LiveTrace() if inspector else None

    class ForwardingAgent(BaseAgent):
        @traced_agent_stream
        async def stream(self, *args, **kwargs):
            async for event in super().stream(*args, **kwargs):
                yield event

    child = ForwardingAgent(TestModel(custom_output_text='private-child-result'), agent_name='Child')
    async def delegate():
        return (await child.run('private-child-prompt', persist=False)).output
    root = ForwardingAgent(TestModel(call_tools=['delegate']), agent_name='Root', tools=[Tool(delegate)])
    async def check():
        with bind_observability_context(run_id='saved-run', conversation_id='saved-chat'):
            with trace.bind() if trace else nullcontext():
                await root.run('private-root-prompt', persist=False)
        assert current_observability_context() == {}
    asyncio.run(check())
    spans = exporter.get_finished_spans()
    agents = [s for s in spans if s.name == 'agent.run']
    assert len(agents) == 2
    parent = next(s for s in agents if s.attributes['agent_name'] == 'Root')
    child_span = next(s for s in agents if s.attributes['agent_name'] == 'Child')
    tool = next(s for s in spans if s.name == 'agent.tool')
    assert tool.parent.span_id == parent.context.span_id
    assert child_span.parent.span_id == tool.context.span_id
    assert child_span.attributes['parent_agent_execution_id'] == parent.attributes['agent_execution_id']
    assert child_span.attributes['delegation_tool_call_id'] == tool.attributes['tool_call_id']
    assert 'tool_call_id' not in child_span.attributes
    assert all(s.attributes['run_id'] == 'saved-run' and s.attributes['outcome'] == 'ok' for s in agents)
    if trace:
        assert {s.attributes['execution_span_id'] for s in agents} == {s.id for s in trace.spans if s.kind == 'agent'}
    assert 'private-child-result' not in output.getvalue()
    assert 'private-root-prompt' not in output.getvalue()
    for metric in metrics.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics:
        assert all(set(p.attributes) == {'operation', 'outcome'} for p in metric.data.data_points)
    path = tmp_path / 'actual.jsonl'
    path.write_text(output.getvalue())
    assert main(['logs', '--file', str(path), '--field', 'run_id=saved-run',
                 '--field', f"agent_execution_id={child_span.attributes['agent_execution_id']}", '--limit', '1']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['truncated']
    assert result['records'][0]['agent_name'] == 'Child'
    assert result['records'][0]['outcome'] == 'ok'
    assert main(['logs', '--file', str(path), '--field', f"tool_call_id={tool.attributes['tool_call_id']}"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert all(r['agent_name'] == 'Root' for r in result['records'])


@pytest.mark.parametrize('mode,outcome', [
    ('terminal_error', 'error'), ('raised', 'error'), ('cancelled', 'cancelled'),
    ('close', 'cancelled'), ('cleanup_error', 'error'), ('missing_terminal', 'error'),
])
def test_terminal_and_cleanup_outcomes_are_truthful(telemetry, mode, outcome):
    output, exporter, _ = telemetry
    class Agent:
        agent_name = 'Controlled'
        @traced_agent_stream
        async def stream(self):
            try:
                if mode == 'raised':
                    raise ValueError('private-exception')
                if mode == 'cancelled':
                    raise asyncio.CancelledError()
                if mode == 'terminal_error':
                    yield StreamEvent.done(DoneReason.ERROR, detail='private-detail')
                elif mode == 'missing_terminal':
                    yield StreamEvent.text_delta('private-text')
                elif mode == 'close':
                    yield StreamEvent.text_delta('private-text')
                    await asyncio.Event().wait()
                else:
                    yield StreamEvent.done(DoneReason.COMPLETE)
            finally:
                if mode == 'cleanup_error':
                    raise ValueError('private-cleanup')
    async def check():
        stream = Agent().stream()
        if mode == 'close':
            await anext(stream)
            await stream.aclose()
        elif mode in ('raised', 'cleanup_error'):
            with pytest.raises(ValueError):
                async for _ in stream:
                    pass
        elif mode == 'cancelled':
            with pytest.raises(asyncio.CancelledError):
                async for _ in stream:
                    pass
        else:
            async for _ in stream:
                pass
        assert current_observability_context() == {}
    asyncio.run(check())
    span = exporter.get_finished_spans()[0]
    assert span.name == 'agent.run' and span.attributes['outcome'] == outcome
    logs = [json.loads(line) for line in output.getvalue().splitlines()]
    assert logs[0]['event_name'] == 'agent.execution.started'
    assert logs[-1]['outcome'] == outcome
    assert 'private-' not in output.getvalue()


def test_concurrent_invocations_of_same_agent_keep_distinct_executions(telemetry):
    _, exporter, _ = telemetry
    entered = 0
    ready = asyncio.Event()
    class Agent:
        agent_name = 'Shared'
        @traced_agent_stream
        async def stream(self):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 2)
            yield StreamEvent.done(DoneReason.COMPLETE)
    agent = Agent()
    async def run(identifier):
        with bind_observability_context(run_id=identifier):
            async for _ in agent.stream():
                pass
    async def check():
        await asyncio.gather(run('a'), run('b'))
    asyncio.run(check())
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert {s.attributes['run_id'] for s in spans} == {'a', 'b'}
    assert len({s.attributes['agent_execution_id'] for s in spans}) == 2
    assert len({s.context.trace_id for s in spans}) == 2


@pytest.mark.parametrize('reason', [DoneReason.COMPLETE, DoneReason.ERROR, DoneReason.CANCELLED])
def test_inspector_keeps_terminal_status_when_consumer_closes_after_done(telemetry, reason):
    _, exporter, _ = telemetry
    trace = LiveTrace()
    cleaned = []

    class Agent:
        agent_name = 'Terminal'

        @traced_agent_stream
        async def stream(self):
            try:
                yield StreamEvent.done(reason)
            finally:
                cleaned.append(True)

    async def check():
        with trace.bind():
            stream = Agent().stream()
            assert (await anext(stream)).type == 'done'
            await stream.aclose()
        assert current_observability_context() == {}

    asyncio.run(check())
    assert cleaned == [True]
    assert trace.spans[0].status == reason.value
    expected = 'ok' if reason == DoneReason.COMPLETE else reason.value
    assert exporter.get_finished_spans()[0].attributes['outcome'] == expected


def test_recursive_same_agent_call_from_tool_gets_separate_execution(telemetry):
    _, exporter, _ = telemetry

    class Agent:
        agent_name = 'Recursive'

        @traced_agent_stream
        async def stream(self, recurse=False):
            if recurse:
                with bind_observability_context(tool_call_id='recursive-tool'):
                    async for _ in self.stream():
                        pass
            yield StreamEvent.done(DoneReason.COMPLETE)

    async def check():
        async for _ in Agent().stream(recurse=True):
            pass
        assert current_observability_context() == {}

    asyncio.run(check())
    child, parent = exporter.get_finished_spans()
    assert child.attributes['parent_agent_execution_id'] == parent.attributes['agent_execution_id']
    assert child.attributes['delegation_tool_call_id'] == 'recursive-tool'
    assert child.attributes['agent_execution_id'] != parent.attributes['agent_execution_id']
    assert child.parent.span_id == parent.context.span_id


def test_child_task_using_same_agent_owns_its_execution(telemetry):
    _, exporter, _ = telemetry
    trace = LiveTrace()

    class Agent:
        agent_name = 'Shared'

        @traced_agent_stream
        async def stream(self, child=False):
            if not child:
                async def consume_child():
                    async for _ in self.stream(child=True):
                        pass
                await asyncio.create_task(consume_child())
            yield StreamEvent.done(DoneReason.COMPLETE)

    async def check():
        with trace.bind(), bind_observability_context(run_id='shared-run'):
            async for _ in Agent().stream():
                pass
        assert current_observability_context() == {}

    asyncio.run(check())
    child, parent = exporter.get_finished_spans()
    assert child.attributes['agent_execution_id'] != parent.attributes['agent_execution_id']
    assert child.attributes['parent_agent_execution_id'] == parent.attributes['agent_execution_id']
    assert child.parent.span_id == parent.context.span_id
    assert {span.attributes['run_id'] for span in (child, parent)} == {'shared-run'}
    assert len(trace.spans) == 2
    assert trace.spans[1].parent_id == trace.spans[0].id
    assert all(row.status == 'complete' for row in trace.spans)


def test_background_storage_failure_emits_origin_execution_ids(telemetry):
    output, exporter, _ = telemetry

    async def check():
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Writer:
            async def write(self, batch):
                started.set()
                try:
                    await release.wait()
                    raise LookupError('private-storage-error')
                finally:
                    finished.set()

        agent = BaseAgent(TestModel(custom_output_text='answer'), writer=Writer())
        with bind_observability_context(run_id='storage-origin'):
            await agent.run('Question', persist_in_background=True)
        await started.wait()
        release.set()
        await finished.wait()
        await agent.aclose()

    asyncio.run(check())
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    failure, = [r for r in records if r['event_name'] == 'agent.persistence_write_failed']
    agent_span, = [s for s in exporter.get_finished_spans() if s.name == 'agent.run']
    assert failure['run_id'] == 'storage-origin'
    assert failure['agent_execution_id'] == agent_span.attributes['agent_execution_id']
    assert failure['severity'] == 'ERROR'
    assert 'private-storage-error' not in output.getvalue()
