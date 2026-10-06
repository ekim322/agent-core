"""Operational tool traces remain useful without collecting model payloads."""
import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Tool, ToolReturn
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from agent_core.agent.base import BaseAgent
from agent_core.utils.telemetry.execution import trace_tool
from agent_core.utils.telemetry.live_trace import LiveTrace
from agent_core.tools.outcomes import tool_error_metadata
from observability import (
    Observability, ObservabilitySettings, bind_observability_context, observe_operation,
)


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


def records(output):
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_tool_failure_outcomes_without_live_trace_and_payloads(telemetry):
    output, exporter, metrics = telemetry
    private = 'private-argument-and-result'

    async def check():
        for call_id, expected in [('ok', 'ok'), ('handled', 'error'), ('raised', 'error'), ('cancelled', 'cancelled')]:
            async def handler(args):
                assert args == private
                if call_id == 'handled':
                    return ToolReturn(return_value=private, metadata=tool_error_metadata())
                if call_id == 'raised':
                    raise ValueError(private)
                if call_id == 'cancelled':
                    raise asyncio.CancelledError()
                return private
            call = SimpleNamespace(tool_name='lookup', tool_call_id=call_id, args=private)
            with bind_observability_context(run_id='run-1'):
                if call_id == 'raised':
                    with pytest.raises(ValueError):
                        await trace_tool(call, handler, private)
                elif call_id == 'cancelled':
                    with pytest.raises(asyncio.CancelledError):
                        await trace_tool(call, handler, private)
                else:
                    await trace_tool(call, handler, private)
            span = exporter.get_finished_spans()[-1]
            assert span.attributes['outcome'] == expected
            assert span.attributes['tool_call_id'] == call_id
            assert span.attributes['run_id'] == 'run-1'
        instruments = metrics.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics
        for metric in instruments:
            assert all(set(point.attributes) == {'operation', 'outcome'} for point in metric.data.data_points)
    asyncio.run(check())
    logs = records(output)
    starts = [row for row in logs if row['event_name'] == 'agent.tool.started']
    ends = [row for row in logs if row['event_name'] == 'operation.completed']
    assert len(starts) == len(ends) == 4
    assert all(row['tool_name'] == 'lookup' and row['run_id'] == 'run-1' for row in logs)
    assert ends[1]['outcome'] == 'error' and ends[1]['severity'] == 'ERROR'
    assert ends[2]['error_type'] == 'ValueError'
    assert private not in output.getvalue()
    assert private not in repr([dict(span.attributes) for span in exporter.get_finished_spans()])


def test_parallel_actual_tool_execution_keeps_call_identity_and_parent(telemetry):
    output, exporter, _ = telemetry

    async def check():
        both_entered = asyncio.Event()
        entered = 0
        async def lookup(value: int):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 2)
            with observe_operation('dependency.lookup'):
                return value
        async def stream(messages, info):
            if any(part.part_kind == 'tool-return' for msg in messages for part in msg.parts):
                yield 'Finished.'
            else:
                yield {0: DeltaToolCall(name='lookup', json_args='{"value":1}'),
                       1: DeltaToolCall(name='lookup', json_args='{"value":2}')}
        agent = BaseAgent(FunctionModel(stream_function=stream), tools=[Tool(lookup)])
        with bind_observability_context(run_id='parallel-run'):
            with observe_operation('caller'):
                await agent.run('Look up both', persist=False)
    asyncio.run(check())
    spans = exporter.get_finished_spans()
    tools = [span for span in spans if span.name == 'agent.tool']
    dependencies = [span for span in spans if span.name == 'dependency.lookup']
    assert len(tools) == len(dependencies) == 2
    assert tools[0].parent.span_id == tools[1].parent.span_id
    assert tools[0].context.trace_id == tools[1].context.trace_id
    assert tools[0].start_time <= tools[1].end_time and tools[1].start_time <= tools[0].end_time
    assert len({span.attributes['tool_call_id'] for span in tools}) == 2
    assert all(span.attributes['run_id'] == 'parallel-run' for span in tools)
    for dependency in dependencies:
        parent = next(span for span in tools if span.context.span_id == dependency.parent.span_id)
        assert dependency.attributes['tool_call_id'] == parent.attributes['tool_call_id']
    assert len([row for row in records(output) if row['event_name'] == 'agent.tool.started']) == 2


def test_optional_inspector_span_is_linked_without_exporting_its_payload(telemetry):
    output, exporter, _ = telemetry
    trace = LiveTrace()
    async def check():
        async def handler(args):
            return ToolReturn(return_value='private-output', metadata={'application_detail': {'private': 'detail'}})
        with trace.bind():
            result = await trace_tool(SimpleNamespace(tool_name='inspect_documents', tool_call_id='call-1',
                args='{"private":"input"}'), handler, None)
            assert result.return_value == 'private-output'
    asyncio.run(check())
    row = trace.spans[0]
    span = exporter.get_finished_spans()[0]
    assert row.input == {'private': 'input'}
    assert row.output == 'private-output'
    assert span.attributes['execution_span_id'] == row.id
    assert all(log['execution_span_id'] == row.id for log in records(output))
    assert 'private-output' not in output.getvalue()
    assert 'application_detail' not in repr(dict(span.attributes))


def test_started_event_is_available_while_tool_is_still_running(telemetry):
    output, exporter, _ = telemetry
    async def check():
        entered = asyncio.Event()
        async def handler(args):
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(trace_tool(SimpleNamespace(tool_name='lookup', tool_call_id='hung-1',
            args=None), handler, None))
        await asyncio.wait_for(entered.wait(), 2)
        assert [row['event_name'] for row in records(output)] == ['agent.tool.started']
        assert not exporter.get_finished_spans()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(check())
    assert records(output)[-1]['outcome'] == 'cancelled'


def test_argument_rejection_is_queryable_without_execution_or_private_errors(telemetry):
    output, exporter, _ = telemetry
    called = False
    private = 'private-invalid-argument'
    async def check():
        async def lookup(value: int):
            nonlocal called
            called = True
            return value
        async def stream(messages, info):
            if any(part.part_kind == 'retry-prompt' for msg in messages for part in msg.parts):
                yield 'The call was rejected.'
            else:
                yield {0: DeltaToolCall(name='lookup', json_args=json.dumps({'value': private}))}
        agent = BaseAgent(FunctionModel(stream_function=stream), tools=[Tool(lookup)])
        with bind_observability_context(run_id='declined-run'):
            result = await agent.run('Inspect', persist=False)
            assert result.output == 'The call was rejected.'
    asyncio.run(check())
    assert not called
    declined = [row for row in records(output) if row['event_name'] == 'agent.tool.declined']
    assert len(declined) == 1
    assert declined[0]['tool_name'] == 'lookup'
    assert declined[0]['tool_call_id']
    assert declined[0]['run_id'] == 'declined-run'
    assert declined[0]['outcome'] == 'rejected'
    assert declined[0]['reason'] == 'argument_validation'
    assert not [row for row in records(output) if row['event_name'] == 'agent.tool.started']
    assert not [span for span in exporter.get_finished_spans() if span.name == 'agent.tool']
    assert next(span for span in exporter.get_finished_spans() if span.name == 'agent.run').attributes['outcome'] == 'ok'
    assert private not in output.getvalue()


def test_returned_recoverable_tool_error_does_not_duplicate_declined_event(telemetry):
    output, _, _ = telemetry
    async def check():
        async def lookup():
            raise ValueError('private-tool-error')
        async def stream(messages, info):
            if any(part.part_kind == 'tool-return' for msg in messages for part in msg.parts):
                yield 'The tool failed.'
            else:
                yield {0: DeltaToolCall(name='lookup', json_args='{}')}
        agent = BaseAgent(FunctionModel(stream_function=stream), tools=[Tool(lookup)])
        await agent.run('Inspect', persist=False)
    asyncio.run(check())
    assert not [row for row in records(output) if row['event_name'] == 'agent.tool.declined']
    completions = [row for row in records(output) if row.get('operation') == 'agent.tool']
    assert len(completions) == 1 and completions[0]['outcome'] == 'error'
    assert 'private-tool-error' not in output.getvalue()
