"""Real runtime tracing with controlled model/tool boundaries, without providers."""
import asyncio
import json

import pytest

from pydantic_ai import Tool
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from agent_core.agent.base import BaseAgent
from agent_core.utils.telemetry.live_trace import LiveTrace


def test_nested_agents_and_parallel_tools_have_actual_execution_parents():
    async def check():
        both_entered = asyncio.Event()
        entered = 0
        async def lookup(value: int):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 2)
            return {'found': value}
        async def child_stream(messages, info):
            if any(part.part_kind == 'tool-return' for msg in messages for part in msg.parts):
                yield 'Both lookups finished.'
            else:
                yield {0: DeltaToolCall(name='lookup', json_args='{"value":1}'),
                       1: DeltaToolCall(name='lookup', json_args='{"value":2}')}
        child = BaseAgent(FunctionModel(stream_function=child_stream), agent_name='Child', tools=[Tool(lookup)])
        async def delegate():
            return (await child.run('Look up both', persist=False)).output
        root = BaseAgent(TestModel(call_tools=['delegate']), agent_name='Root', tools=[Tool(delegate)])
        trace = LiveTrace()
        with trace.bind():
            await root.run('Investigate', persist=False)
        spans = [s for s in trace.spans if s.kind != 'model']
        assert [s.kind for s in spans] == ['agent', 'tool', 'agent', 'tool', 'tool']
        assert spans[1].parent_id == spans[0].id
        assert spans[2].parent_id == spans[1].id
        first, second = spans[3:]
        assert first.parent_id == second.parent_id == spans[2].id
        assert first.input == {'value': 1} and second.input == {'value': 2}
        assert first.output == {'found': 1}
        assert first.started_ms <= second.ended_ms and second.started_ms <= first.ended_ms
        assert all(s.status == 'complete' for s in spans)
        assert spans[2].output == 'Both lookups finished.'
        models = [s for s in trace.spans if s.kind == 'model']
        assert len(models) == 4
        assert [s.parent_id for s in models] == [spans[0].id, spans[2].id, spans[2].id, spans[0].id]
        assert models[1].input['available_tools'] == ['lookup']
        assert models[1].model.available_tools == ['lookup']
        assert models[1].model.provider == 'function'
        assert models[1].output['tool_calls'][0]['name'] == 'lookup'
        assert models[2].output['text'] == 'Both lookups finished.'
        assert models[2].model.input_tokens > 0
        assert all(s.ended_ms is not None for s in models)
    asyncio.run(check())


def test_trace_isolation_errors_redaction_and_cancellation():
    async def check():
        async def broken():
            raise ValueError('source unavailable')
        agent = BaseAgent(TestModel(call_tools=['broken']), tools=[Tool(broken)])
        first, second = LiveTrace(), LiveTrace()
        async def run(trace):
            with trace.bind():
                await agent.run('Inspect', persist=False)
        await asyncio.gather(run(first), run(second))
        assert len(first.spans) == len(second.spans) == 4
        assert not {s.id for s in first.spans} & {s.id for s in second.spans}
        assert next(s for s in first.spans if s.kind == 'tool').status == 'error'
        safe = LiveTrace(('very-secret-value',))
        assert safe.capture({'api_key': 'hidden', 'data': 'very-secret-value'}) == {'api_key': '[redacted]', 'data': '[redacted]'}
        assert safe.capture('x' * 200000) == 'x' * 200000
        assert not safe.truncated
        trace = LiveTrace()
        started = asyncio.Event()
        async def blocking():
            started.set()
            await asyncio.Event().wait()
        agent = BaseAgent(TestModel(call_tools=['blocking']), tools=[Tool(blocking)])
        async def run_blocked():
            with trace.bind():
                await agent.run('Wait', persist=False)
        task = asyncio.create_task(run_blocked())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert trace.spans[0].status == next(s for s in trace.spans if s.kind == 'tool').status == 'cancelled'
    asyncio.run(check())


def test_stream_redacts_credentials_split_across_deltas():
    from agent_core.streaming import StreamEvent, DoneReason
    trace = LiveTrace(('sk-test-123456789',))
    with trace.bind(), trace.span('agent', 'Test', 'hello') as row:
        trace.observe(row, StreamEvent.text_delta('Value: sk-test-'))
        assert row.text == 'Value: '
        trace.observe(row, StreamEvent.text_delta('123456789 end'))
        assert row.text == 'Value: [redacted] end'
        trace.observe(row, StreamEvent.done(DoneReason.COMPLETE, output='sk-test-123456789'))
        assert row.output == '[redacted]'


def test_rejected_tool_arguments_remain_visible_in_trace():
    async def check():
        async def lookup(value: int):
            return value
        async def stream(messages, info):
            if any(part.part_kind in ('tool-return', 'retry-prompt') for msg in messages for part in msg.parts):
                yield 'The call was rejected.'
            else:
                yield {0: DeltaToolCall(name='lookup', json_args='{"value":"not-a-number"}')}
        agent = BaseAgent(FunctionModel(stream_function=stream), tools=[Tool(lookup)])
        trace = LiveTrace()
        with trace.bind():
            await agent.run('Inspect', persist=False)
        tool = next(s for s in trace.spans if s.kind == 'tool')
        assert tool.name == 'lookup'
        assert tool.status == 'declined'
        assert 'not-a-number' in str(tool.input)
        assert tool.output is not None
    asyncio.run(check())


def test_model_trace_records_visible_output_usage_and_thinking_presence_only():
    from pydantic_ai.models.function import DeltaThinkingPart

    async def check():
        async def stream(messages, info):
            yield {0: DeltaThinkingPart(content='private reasoning must stay private', signature='private-signature')}
            yield 'Visible sk-test-'
            yield '123456789 answer.'
        agent = BaseAgent(FunctionModel(stream_function=stream), system_prompt='private system instructions')
        trace = LiveTrace(('sk-test-123456789',))
        with trace.bind():
            await agent.run('Test model diagnostics', persist=False)
        model = next(s for s in trace.spans if s.kind == 'model')
        assert model.model.reasoning_observed is True
        assert model.model.output_tokens > 0
        assert model.input['streaming'] is True
        assert model.output['text'] == 'Visible [redacted] answer.'
        saved = json.dumps([span.model_dump(mode='json') for span in trace.spans])
        assert 'private reasoning' not in saved and 'private-signature' not in saved
        assert model.input['instructions'] == ['private system instructions']
        assert 'sk-test-123456789' not in saved
    asyncio.run(check())


def test_failed_and_cancelled_model_requests_close_their_spans():
    async def check():
        async def failed_stream(messages, info):
            yield 'Partial output'
            raise ValueError('private exception detail')
        trace = LiveTrace()
        agent = BaseAgent(FunctionModel(stream_function=failed_stream))
        with trace.bind(), pytest.raises(ValueError):
            await agent.run('Fail safely', persist=False)
        model = next(s for s in trace.spans if s.kind == 'model')
        assert model.status == 'error' and model.ended_ms is not None
        assert model.text == 'Partial output'
        assert 'private exception detail' not in json.dumps([span.model_dump(mode='json') for span in trace.spans])

        entered = asyncio.Event()
        async def blocking_stream(messages, info):
            yield 'Started'
            entered.set()
            await asyncio.Event().wait()
        trace = LiveTrace()
        agent = BaseAgent(FunctionModel(stream_function=blocking_stream))
        async def run():
            with trace.bind():
                await agent.run('Wait', persist=False)
        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert len(trace.spans) == 2
        assert all(s.status == 'cancelled' and s.ended_ms is not None for s in trace.spans)
    asyncio.run(check())


def test_request_messages_redact_tool_arguments_and_omit_reasoning():
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ThinkingPart
    from agent_core.utils.telemetry.model_trace import _visible_message
    response = ModelResponse(parts=[
        ThinkingPart(content='hidden reasoning', signature='hidden signature'),
        ToolCallPart(tool_name='lookup', args='{"password":"sensitive value","query":"visible"}'),
    ])
    captured = LiveTrace().capture(_visible_message(response))
    assert captured['parts'][0] == {'part_kind': 'thinking', 'omitted': True}
    assert captured['parts'][1]['args'] == {'password': '[redacted]', 'query': 'visible'}
    assert 'hidden' not in json.dumps(captured)


def test_capture_preserves_large_payloads_and_long_execution_traces():
    from agent_core.streaming import StreamEvent
    from agent_core.persistence.records import validate_execution_spans
    trace = LiveTrace()
    payload = 'é' * 2_100_000
    assert trace.capture({'result': payload}) == {'result': payload}
    with trace.bind(), trace.span('agent', 'Root', payload) as root:
        trace.observe(root, StreamEvent.text_delta(payload))
        for index in range(600):
            with trace.span('tool', str(index), {'index': index}) as tool:
                tool.output = trace.capture(payload[:101000])
    assert root.text == payload
    assert len(trace.spans) == 601
    assert trace.spans[-1].output == payload[:101000]
    assert not trace.truncated
    validate_execution_spans(trace.spans)




def test_model_trace_records_thinking_settings_sent_to_provider():
    from types import SimpleNamespace
    from pydantic_ai.models import ModelRequestParameters
    from agent_core.models.config import ModelConfig
    from agent_core.utils.telemetry.model_trace import capture_model_request, trace_model_request

    def captured(model_name, provider, thinking):
        context = SimpleNamespace(
            model=SimpleNamespace(model_name=model_name, system='openai', route=SimpleNamespace(provider=provider)),
            model_settings=ModelConfig(thinking=thinking).to_settings(model_name),
            model_request_parameters=ModelRequestParameters(), messages=[], streaming=True)
        trace = LiveTrace()
        with trace.bind(), trace_model_request(1, model_name) as model_trace:
            capture_model_request(context)
        return model_trace.row.model

    openai = captured('gpt-5', 'openai', 'high')
    assert openai.thinking == 'high' and openai.reasoning_summary == 'auto'


def test_model_trace_preserves_cached_usage_in_saved_span():
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.usage import RequestUsage
    from agent_core.persistence.records import ExecutionSpan, ModelDiagnostics
    from agent_core.utils.telemetry.model_trace import ModelTrace

    row = ExecutionSpan(id='model', kind='model', name='Request', started_ms=0,
                        model=ModelDiagnostics(name='test'))
    ModelTrace(LiveTrace(), row).finish(ModelResponse(
        parts=[TextPart('Done')],
        usage=RequestUsage(input_tokens=1000, output_tokens=50, cache_read_tokens=750),
    ), None)
    restored = ExecutionSpan.model_validate(row.model_dump(mode='json', exclude_none=True))
    assert restored.model.input_tokens == 1000
    assert restored.model.cache_read_tokens == 750
    assert restored.model.output_tokens == 50


@pytest.mark.parametrize(
    "body_effort,expected", [("max", "max"), ("", "high"), (42, None), (None, "high")]
)
def test_prepared_request_inspection_preserves_option_precedence_and_privacy(
    body_effort, expected
):
    from types import SimpleNamespace

    from pydantic_ai.messages import ModelResponse, ThinkingPart, ToolCallPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.tools import ToolDefinition

    from agent_core.utils.telemetry.model_trace import capture_model_request, trace_model_request

    secret = "inspection-secret"
    settings = {
        "thinking": False,
        "extra_body": {"reasoning_effort": body_effort, "private_option": secret},
        "openai_reasoning_effort": "high",
        "openai_reasoning_summary": 42,
        "temperature": None,
        "seed": 7,
        "unknown_sdk_option": secret,
    }
    context = SimpleNamespace(
        model=SimpleNamespace(model_name="prepared-model", system="fallback-provider"),
        model_settings=settings,
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name=f"lookup-{secret}")],
            output_tools=[ToolDefinition(name="structured_answer")],
            instruction_parts=[SimpleNamespace(content=f"Instruction {secret}")],
        ),
        messages=[ModelResponse(parts=[
            ThinkingPart(content="private reasoning", signature="private signature"),
            ToolCallPart(tool_name="lookup", args='{"password":"hidden","query":"visible"}'),
        ])],
        streaming=False,
    )
    trace = LiveTrace((secret,))
    with trace.bind(), trace_model_request(1, "prepared-model") as inspection:
        capture_model_request(context)
        row = inspection.row
        assert row.model.provider == "fallback-provider"
        assert row.model.reasoning_effort == expected
        assert row.model.reasoning_summary is None
        assert row.model.thinking is False
        assert row.model.phase == "waiting_for_output"
        assert row.model.available_tools == ["lookup-[redacted]"]
        assert row.input["settings"] == {"temperature": None, "seed": 7}
        assert row.input["instructions"] == ["Instruction [redacted]"]
        assert row.input["output_tools"] == ["structured_answer"]
        assert row.input["message_count"] == 1
        assert row.input["streaming"] is False
        snapshot = row.input
        with trace.span("tool", "nested tool", None):
            capture_model_request(context)
        assert row.input is snapshot
    saved = json.dumps([span.model_dump(mode="json") for span in trace.spans])
    assert secret not in saved
    assert "private reasoning" not in saved and "private signature" not in saved
    assert "unknown_sdk_option" not in saved and "private_option" not in saved
    assert context.model_settings is settings
    assert settings["unknown_sdk_option"] == secret


def test_prepared_request_capture_respects_payload_retention_limit():
    from types import SimpleNamespace

    from pydantic_ai.models import ModelRequestParameters
    from agent_core.utils.telemetry.model_trace import capture_model_request, trace_model_request

    context = SimpleNamespace(
        model=SimpleNamespace(model_name="test", system="test"),
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        messages=[],
        streaming=True,
    )
    trace = LiveTrace(max_capture_chars=1)
    with trace.bind(), trace_model_request(1, "test") as inspection:
        capture_model_request(context)
        assert inspection.row.input == "[truncated]"
        assert inspection.row.model.provider == "test"
        assert inspection.row.model.phase == "waiting_for_output"
    assert trace.truncated


def test_model_inspection_restores_outer_scope_after_nested_failure_and_budget_exhaustion():
    from types import SimpleNamespace
    from pydantic_ai.models import ModelRequestParameters
    from agent_core.utils.telemetry.model_trace import capture_model_request, trace_model_request

    def request(name):
        return SimpleNamespace(
            model=SimpleNamespace(model_name=name, system="test"),
            model_settings=None, model_request_parameters=ModelRequestParameters(),
            messages=[], streaming=True,
        )

    trace = LiveTrace(max_spans=2)
    with trace.bind(), trace_model_request(1, "outer") as outer:
        capture_model_request(request("outer-before"))
        with pytest.raises(ValueError):
            with trace_model_request(2, "inner") as inner:
                capture_model_request(request("inner"))
                raise ValueError("inner failed")
        assert inner.row.status == "error"
        assert outer.row.input["model"] == "outer-before"
        with trace_model_request(3, "discarded") as discarded:
            assert discarded.row is None
            capture_model_request(request("must-not-leak"))
        assert outer.row.input["model"] == "outer-before"
        capture_model_request(request("outer-after"))
        assert outer.row.input["model"] == "outer-after"
    capture_model_request(request("outside"))
    assert outer.row.input["model"] == "outer-after"
    assert outer.row.status == "complete"
