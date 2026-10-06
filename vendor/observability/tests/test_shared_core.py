import asyncio
import io
import json
import logging

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)

from observability import (
    Observability,
    ObservabilitySettings,
    bind_observability_context,
    current_trace_ids,
    observe_operation,
)


def test_correlated_errors_preserve_details_without_exception_payloads():
    output = io.StringIO()
    spans = InMemorySpanExporter()
    logs = InMemoryLogRecordExporter()
    runtime = Observability(
        ObservabilitySettings(),
        stream=output,
        span_processors=[SimpleSpanProcessor(spans)],
        log_processors=[SimpleLogRecordProcessor(logs)],
    )
    runtime.start()
    try:
        with bind_observability_context(conversation_id='chat1'):
            with pytest.raises(ValueError):
                with observe_operation('chat.execute'):
                    logging.getLogger('test').info(
                        'Checkpoint',
                        extra={
                            'event_name': 'chat.checkpoint',
                            'attempt': 2,
                            'api_key': 'secret',
                            'duration': float('nan'),
                        },
                    )
                    raise ValueError('private prompt text')
    finally:
        runtime.close()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r['severity'] for r in records] == ['INFO', 'ERROR']
    assert records[0]['attempt'] == 2
    assert records[0]['api_key'] == '[REDACTED]'
    assert records[0]['duration'] is None
    assert records[1]['error_type'] == 'ValueError'
    assert records[1]['error_frames'][0]['function']
    assert records[0]['trace_id'] == records[1]['trace_id']
    assert records[0]['conversation_id'] == 'chat1'
    assert 'private prompt text' not in output.getvalue()
    assert spans.get_finished_spans()[0].status.status_code.name == 'ERROR'
    exported = logs.get_finished_logs()[0].log_record
    assert f'{exported.trace_id:032x}' == records[0]['trace_id']
    assert exported.attributes['api_key'] == '[REDACTED]'


def test_async_context_and_detached_job_links():
    output, exporter = io.StringIO(), InMemorySpanExporter()
    runtime = Observability(
        ObservabilitySettings(),
        stream=output,
        span_processors=[SimpleSpanProcessor(exporter)],
    )

    async def run():
        ready = asyncio.Event()

        async def worker(identifier):
            with bind_observability_context(conversation_id=identifier):
                with observe_operation('chat.request'):
                    parent = current_trace_ids()
                    ready.set()
                    await asyncio.sleep(0)
                    with observe_operation('chat.execute', new_root=True):
                        assert current_trace_ids()['trace_id'] != parent['trace_id']

        await asyncio.gather(worker('a'), worker('b'))

    runtime.start()
    try:
        asyncio.run(run())
    finally:
        runtime.close()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert sorted(r['conversation_id'] for r in records) == ['a', 'a', 'b', 'b']
    jobs = [s for s in exporter.get_finished_spans() if s.name == 'chat.execute']
    parents = [s for s in exporter.get_finished_spans() if s.name == 'chat.request']
    assert all(s.parent is None and len(s.links) == 1 for s in jobs)
    assert {s.links[0].context.trace_id for s in jobs} == {
        s.context.trace_id for s in parents
    }
    assert current_trace_ids() == {}


def test_metrics_only_use_bounded_dimensions_and_cancellation_is_distinct():
    reader = InMemoryMetricReader()
    runtime = Observability(
        ObservabilitySettings(), stream=io.StringIO(), metric_readers=[reader]
    )
    runtime.start()
    try:
        with bind_observability_context(run_id='unique-job'):
            with observe_operation('chat.execute'):
                pass
            with pytest.raises(asyncio.CancelledError):
                with observe_operation('chat.execute'):
                    raise asyncio.CancelledError()
        data = reader.get_metrics_data()
        instruments = data.resource_metrics[0].scope_metrics[0].metrics
        assert {m.name for m in instruments} == {
            'application.operation.count',
            'application.operation.duration',
        }
        for metric in instruments:
            assert {p.attributes['outcome'] for p in metric.data.data_points} == {
                'ok',
                'cancelled',
            }
            assert all(
                set(p.attributes) == {'operation', 'outcome'}
                for p in metric.data.data_points
            )
    finally:
        runtime.close()


def test_lifecycle_restores_logging_and_supports_next_application():
    root, access = logging.getLogger(), logging.getLogger('uvicorn.access')
    previous = (root.handlers[:], root.level, access.disabled)
    traces = []
    for _ in range(2):
        runtime = Observability(ObservabilitySettings(), stream=io.StringIO())
        runtime.start()
        runtime.start()
        assert access.disabled
        with observe_operation('test'):
            traces.append(current_trace_ids()['trace_id'])
        runtime.close()
        runtime.close()
        assert (root.handlers, root.level, access.disabled) == previous
    assert traces[0] != traces[1]


def test_configuration_headers_are_not_in_repr():
    settings = ObservabilitySettings.from_env(
        {
            'OTEL_EXPORTER_OTLP_HEADERS': 'Authorization=Bearer%20secret',
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318',
        }
    )
    assert settings.otlp_headers == {'Authorization': 'Bearer secret'}
    assert 'secret' not in repr(settings)
    with pytest.raises(ValueError):
        ObservabilitySettings(otlp_endpoint='https://secret:password@collector')


def test_log_level_and_explicit_trace_context_after_span_ends():
    output, logs = io.StringIO(), InMemoryLogRecordExporter()
    runtime = Observability(
        ObservabilitySettings(log_level='WARNING'),
        stream=output,
        log_processors=[SimpleLogRecordProcessor(logs)],
    )
    runtime.start()
    try:
        with observe_operation('test'):
            ids = current_trace_ids()
            logging.getLogger('test').debug('Should be hidden')
            logging.getLogger('test').info('Should also be hidden')
        with bind_observability_context(**ids):
            logging.getLogger('test').error('Error after span exit')
    finally:
        runtime.close()
    records = output.getvalue().splitlines()
    assert len(records) == 1
    assert logs.get_finished_logs()[0].log_record.trace_id == int(ids['trace_id'], 16)


def test_otlp_http_exports_all_signals_to_configured_endpoint():
    # Capture actual protobuf requests without any network socket/provider account.
    # This exercises exporter encoding, paths, headers and flush behavior.
    import requests
    from unittest.mock import patch
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    received = {}

    def capture(session, url, data=None, **kwargs):
        assert session.headers['Authorization'] == 'Bearer test-only'
        received[url.rsplit('/', 1)[-1]] = data
        response = requests.Response()
        response.status_code = 200
        response._content = b''
        return response

    with patch.object(requests.Session, 'post', capture):
        runtime = Observability(
            ObservabilitySettings(
                otlp_endpoint='http://collector.test:4318',
                otlp_headers={'Authorization': 'Bearer test-only'},
            ),
            stream=io.StringIO(),
        )
        runtime.start()
        try:
            with bind_observability_context(run_id='example'):
                with observe_operation('chat.execute'):
                    ids = current_trace_ids()
            runtime.tracer_provider.force_flush()
            runtime.logger_provider.force_flush()
            runtime.meter_provider.force_flush()
        finally:
            runtime.close()
    assert set(received) == {'logs', 'metrics', 'traces'}
    spans = ExportTraceServiceRequest.FromString(received['traces'])
    logs = ExportLogsServiceRequest.FromString(received['logs'])
    metrics = ExportMetricsServiceRequest.FromString(received['metrics'])
    assert (
        spans.resource_spans[0].scope_spans[0].spans[0].trace_id.hex()
        == ids['trace_id']
    )
    assert (
        logs.resource_logs[0].scope_logs[0].log_records[0].trace_id.hex()
        == ids['trace_id']
    )
    assert {m.name for m in metrics.resource_metrics[0].scope_metrics[0].metrics} == {
        'application.operation.count',
        'application.operation.duration',
    }


def test_failed_exporter_setup_restores_lifecycle(monkeypatch):
    previous = logging.getLogger().handlers[:]
    runtime = Observability(
        ObservabilitySettings(otlp_endpoint='http://collector.test')
    )

    def fail():
        raise ValueError('Invalid exporter configuration')

    monkeypatch.setattr(runtime, '_configure_exporters', fail)
    with pytest.raises(ValueError):
        runtime.start()
    assert logging.getLogger().handlers == previous
    fresh = Observability(ObservabilitySettings(), stream=io.StringIO())
    fresh.start()
    fresh.close()


def test_quiet_operations_keep_spans_metrics_and_visible_errors():
    output, spans, reader = (
        io.StringIO(),
        InMemorySpanExporter(),
        InMemoryMetricReader(),
    )
    runtime = Observability(
        ObservabilitySettings(),
        stream=output,
        span_processors=[SimpleSpanProcessor(spans)],
        metric_readers=[reader],
    )
    runtime.start()
    try:
        with observe_operation('db.read', success_log_level=logging.DEBUG):
            pass
        with pytest.raises(ValueError):
            with observe_operation('db.read', success_log_level=logging.DEBUG):
                raise ValueError('private')
        with observe_operation('db.read', success_log_level=logging.DEBUG) as observed:
            observed.set_outcome('rejected')
        with pytest.raises(asyncio.CancelledError):
            with observe_operation('db.read', success_log_level=logging.DEBUG):
                raise asyncio.CancelledError
        instruments = [
            m
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for m in scope.metrics
        ]
        duration = next(
            m for m in instruments if m.name == 'application.operation.duration'
        )
        assert sum(point.count for point in duration.data.data_points) == 4
        assert duration.data.data_points[0].explicit_bounds[0] == 0.005
        assert 0.1 in duration.data.data_points[0].explicit_bounds
        assert len(spans.get_finished_spans()) == 4
    finally:
        runtime.close()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [row['outcome'] for row in records] == ['error', 'rejected', 'cancelled']
    assert [row['severity'] for row in records] == ['ERROR', 'INFO', 'INFO']


def test_handled_failure_completion_keeps_span_ids_and_restores_parent_context():
    output = io.StringIO()
    spans = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    runtime = Observability(
        ObservabilitySettings(),
        stream=output,
        span_processors=[SimpleSpanProcessor(spans)],
        metric_readers=[reader],
    )
    runtime.start()
    try:
        with bind_observability_context(run_id='run-a'):
            with observe_operation('parent'):
                parent_ids = current_trace_ids()
                with observe_operation('child') as operation:
                    child_ids = current_trace_ids()
                    operation.record_exception(ValueError('private error text'))
                assert current_trace_ids() == parent_ids
        assert current_trace_ids() == {}
        metrics = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics
        counter = next(
            metric for metric in metrics if metric.name == 'application.operation.count'
        )
        assert {
            (point.attributes['operation'], point.attributes['outcome']): point.value
            for point in counter.data.data_points
        } == {('child', 'error'): 1, ('parent', 'ok'): 1}
    finally:
        runtime.close()

    child_log, parent_log = [
        json.loads(line) for line in output.getvalue().splitlines()
    ]
    assert child_log['trace_id'] == child_ids['trace_id'] == parent_log['trace_id']
    assert child_log['span_id'] == child_ids['span_id']
    assert parent_log['span_id'] == parent_ids['span_id']
    assert child_log['run_id'] == parent_log['run_id'] == 'run-a'
    assert child_log['outcome'] == 'error'
    assert child_log['severity'] == 'ERROR'
    assert child_log['error_type'] == 'ValueError'
    assert 'private error text' not in output.getvalue()
    child_span = next(
        span for span in spans.get_finished_spans() if span.name == 'child'
    )
    assert child_span.attributes['outcome'] == 'error'
    assert child_span.status.status_code.name == 'ERROR'


def test_partial_logging_setup_failure_restores_all_loggers_and_closes_processors(
    monkeypatch,
):
    names = (
        '',
        'uvicorn',
        'uvicorn.error',
        'uvicorn.access',
        'httpx',
        'httpcore',
        'opentelemetry',
        'urllib3',
    )

    def logger_state(name):
        logger = logging.getLogger(name)
        return (logger.handlers[:], logger.level, logger.propagate, logger.disabled)

    previous = {name: logger_state(name) for name in names}
    shutdowns = []
    processor = SimpleSpanProcessor(InMemorySpanExporter())

    def fail_shutdown():
        shutdowns.append('span')
        raise RuntimeError('secondary shutdown failure')

    monkeypatch.setattr(processor, 'shutdown', fail_shutdown)
    log_processor = SimpleLogRecordProcessor(InMemoryLogRecordExporter())
    original_log_shutdown = log_processor.shutdown

    def shutdown_logs():
        shutdowns.append('log')
        original_log_shutdown()

    monkeypatch.setattr(log_processor, 'shutdown', shutdown_logs)
    runtime = Observability(
        ObservabilitySettings(),
        stream=io.StringIO(),
        span_processors=[processor],
        log_processors=[log_processor],
    )
    original_configure_logging = runtime._configure_logging

    def fail_after_configuring_logging():
        original_configure_logging()
        raise ValueError('original startup failure')

    monkeypatch.setattr(runtime, '_configure_logging', fail_after_configuring_logging)

    with pytest.raises(ValueError, match='original startup failure'):
        runtime.start()
    assert shutdowns == ['span', 'log']
    assert {name: logger_state(name) for name in names} == previous
    with pytest.raises(RuntimeError, match='Create a new Observability runtime'):
        runtime.start()
    fresh = Observability(ObservabilitySettings(), stream=io.StringIO())
    fresh.start()
    fresh.close()
