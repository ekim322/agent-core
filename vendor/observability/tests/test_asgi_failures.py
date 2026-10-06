"""Optional request enrichment never replaces the application's outcome."""

import asyncio
import io
import json

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from observability import Observability, ObservabilitySettings
from observability.integrations.asgi import HTTPObservabilityMiddleware


@pytest.mark.parametrize('outcome', ['ok', 'error', 'cancelled'])
def test_failed_request_fields_preserves_response_exception_and_telemetry(outcome):
    output, spans, metrics = io.StringIO(), InMemorySpanExporter(), InMemoryMetricReader()
    runtime = Observability(
        ObservabilitySettings(), stream=output,
        span_processors=[SimpleSpanProcessor(spans)], metric_readers=[metrics],
    )
    original = ValueError('private application text') if outcome == 'error' else asyncio.CancelledError()
    sent = []

    async def app(scope, receive, send):
        if outcome != 'ok':
            raise original
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'first', 'more_body': True})
        await send({'type': 'http.response.body', 'body': b'last'})

    def request_fields(scope):
        raise RuntimeError('private callback text')

    async def receive():
        return {'type': 'http.request', 'body': b''}

    async def send(message):
        sent.append(message)

    middleware = HTTPObservabilityMiddleware(app, request_fields=request_fields)
    scope = {'type': 'http', 'method': 'GET'}
    runtime.start()
    try:
        if outcome == 'ok':
            asyncio.run(middleware(scope, receive, send))
            assert [message.get('body') for message in sent[1:]] == [b'first', b'last']
            assert (b'x-request-id', scope['state']['request_id'].encode()) in sent[0]['headers']
        else:
            with pytest.raises(type(original)) as caught:
                asyncio.run(middleware(scope, receive, send))
            assert caught.value is original
        instruments = [metric for resource in metrics.get_metrics_data().resource_metrics
                       for group in resource.scope_metrics for metric in group.metrics]
        duration = next(metric for metric in instruments if metric.name == 'http.server.request.duration')
        assert duration.data.data_points[0].attributes['outcome'] == outcome
        assert duration.data.data_points[0].count == 1
    finally:
        runtime.close()

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [row['event_name'] for row in records] == ['http.request_fields.failed', 'operation.completed']
    assert records[0]['error_type'] == 'RuntimeError'
    assert records[1]['outcome'] == outcome
    assert records[0]['request_id'] == records[1]['request_id'] == scope['state']['request_id']
    assert records[0]['trace_id'] == records[1]['trace_id']
    assert spans.get_finished_spans()[0].attributes['outcome'] == outcome
    assert 'private' not in output.getvalue()
