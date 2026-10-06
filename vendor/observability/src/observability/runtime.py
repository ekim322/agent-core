"""Own logging handlers and SDK providers for an application lifespan.

Providers are passed explicitly, not installed into OpenTelemetry's irreversible
process globals. Integrations can use the public providers or package helpers.
"""

import logging
import sys
from typing import TextIO

from opentelemetry import metrics, trace
from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .log_records import JsonFormatter
from .settings import ObservabilitySettings

_active: 'Observability | None' = None


def get_tracer(name: str = 'application'):
    """Get a span creator from the running runtime; before startup it is a no-op."""
    return (
        _active.tracer_provider if _active else trace.NoOpTracerProvider()
    ).get_tracer(name)


def get_meter(name: str = 'application'):
    """Get metric instruments from the running runtime; before startup it is a no-op."""
    return (
        _active.meter_provider if _active else metrics.NoOpMeterProvider()
    ).get_meter(name)


class _ExportHandler(logging.Handler):
    def __init__(self, provider: LoggerProvider, formatter: JsonFormatter):
        super().__init__()
        self.provider = provider
        self.record_formatter = formatter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_record(record)
        except Exception as exc:
            logging.getLogger('opentelemetry.export').warning(
                'Telemetry log export failed: %s',
                type(exc).__name__,
            )

    def _emit_record(self, record: logging.LogRecord) -> None:
        # Exporter diagnostics stay on stdout; never recursively export them.
        if record.name.startswith(('opentelemetry', 'urllib3', 'requests')):
            return
        fields = self.record_formatter.record_fields(record)
        severity = {
            10: SeverityNumber.DEBUG,
            20: SeverityNumber.INFO,
            30: SeverityNumber.WARN,
            40: SeverityNumber.ERROR,
            50: SeverityNumber.FATAL,
        }.get(record.levelno, SeverityNumber.UNSPECIFIED)
        self.provider.get_logger(record.name).emit(
            LogRecord(
                timestamp=int(record.created * 1_000_000_000),
                severity_text=record.levelname,
                severity_number=severity,
                trace_id=int(str(fields.get('trace_id', '0')), 16),
                span_id=int(str(fields.get('span_id', '0')), 16),
                body=fields.pop('message'),
                attributes=fields,
                event_name=str(fields.get('event_name', '')) or None,
            )
        )


class Observability:
    """Configure where application logs, spans, and metrics are sent.

    Start once during application startup. Standard Python logging then writes
    structured JSON to stdout (or the supplied stream), including bound fields
    and active trace/span IDs. A configured OTLP endpoint also receives logs,
    spans, and metrics through background exporters. The destination or container
    logging system owns retention; this class creates no local log database.

    Only one runtime may be active per process. Close it after application tasks
    finish to shut down exporters and restore the previous logging configuration.
    Package get_tracer/get_meter helpers use these runtime-owned providers;
    OpenTelemetry's global provider accessors are configured separately.

    Optional readers/processors support in-memory testing or custom integrations.
    They are owned and closed by this runtime just like configured exporters.
    Repeated starts while running and repeated closes are harmless. After close
    or failed startup, construct a new runtime. Avoid changing logging settings
    while active, because close restores the settings captured during startup.
    """

    def __init__(
        self,
        settings: ObservabilitySettings | None = None,
        *,
        stream: TextIO | None = None,
        span_processors=(),
        metric_readers=(),
        log_processors=(),
    ):
        self.settings = settings or ObservabilitySettings.from_env()
        self.stream = stream
        self._span_processors = span_processors
        self._metric_readers = metric_readers
        self._log_processors = log_processors
        self._started = False
        self._closed = False
        self._handlers: list[logging.Handler] = []
        self._logger_states = []

    def start(self) -> None:
        """Install JSON logging and telemetry providers; undo setup if it fails."""
        try:
            self._start()
        except Exception:
            # A bad exporter configuration must not leak worker threads or replace
            # application logging after startup reports a failure.
            self._restore_logging()
            self._shutdown_created_providers_after_start_failure()
            self._closed = True
            raise

    def _shutdown_created_providers_after_start_failure(self) -> None:
        """Release partial startup resources without masking the original failure."""
        for name in ('tracer_provider', 'meter_provider', 'logger_provider'):
            provider = getattr(self, name, None)
            if provider is not None:
                try:
                    provider.shutdown()
                except Exception:
                    pass

    def _start(self) -> None:
        global _active
        if self._started:
            return
        if self._closed:
            raise RuntimeError('Create a new Observability runtime after close')
        if _active is not None:
            raise RuntimeError('An Observability runtime is already active')
        self._configure_providers()
        self._configure_logging()
        _active = self
        self._started = True

    def _configure_providers(self) -> None:
        """Create owned providers and attach supplied or configured exporters."""
        resource = Resource.create(
            {
                'service.name': self.settings.service_name,
                'service.version': self.settings.service_version,
                'deployment.environment.name': self.settings.environment,
            }
        )
        readers = list(self._metric_readers)
        self.tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        self.logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
        for processor in self._span_processors:
            self.tracer_provider.add_span_processor(processor)
        for processor in self._log_processors:
            self.logger_provider.add_log_record_processor(processor)
        if self.settings.otlp_endpoint:
            readers.append(self._configure_exporters())
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers,
            shutdown_on_exit=False,
            views=_duration_histogram_views(),
        )

    def _configure_logging(self) -> None:
        """Save existing logger state before installing structured output."""
        formatter = JsonFormatter(self.settings)
        console = logging.StreamHandler(
            self.stream if self.stream is not None else sys.stdout
        )
        console.setFormatter(formatter)
        self._handlers.append(console)
        if self.settings.otlp_endpoint or self._log_processors:
            self._handlers.append(_ExportHandler(self.logger_provider, formatter))
        self._redirect_application_loggers()
        self._quiet_transport_loggers()
        logging.getLogger().handlers = self._handlers.copy()

    def _save_logger_state(self, logger: logging.Logger) -> None:
        self._logger_states.append(
            (
                logger,
                list(logger.handlers),
                logger.level,
                logger.propagate,
                logger.disabled,
            )
        )

    def _redirect_application_loggers(self) -> None:
        """Route root/Uvicorn output through our handlers and suppress raw access logs."""
        # Uvicorn otherwise keeps separate text handlers; its access logger embeds
        # raw paths/query strings. HTTP middleware emits safe route summaries.
        for name in ('', 'uvicorn', 'uvicorn.error', 'uvicorn.access'):
            logger = logging.getLogger(name)
            self._save_logger_state(logger)
            logger.handlers = []
            logger.propagate = bool(name)
            logger.disabled = name == 'uvicorn.access'
            logger.setLevel(self.settings.log_level if not name else logging.NOTSET)

    def _quiet_transport_loggers(self) -> None:
        """Reduce transport chatter while retaining each logger's previous settings."""
        for name in ('httpx', 'httpcore', 'opentelemetry', 'urllib3'):
            logger = logging.getLogger(name)
            self._save_logger_state(logger)
            logger.setLevel(logging.WARNING)

    def _restore_logging(self) -> None:
        """Restore the logger settings captured during startup."""
        for logger, handlers, level, propagate, disabled in self._logger_states:
            logger.handlers = handlers
            logger.propagate = propagate
            logger.disabled = disabled
            logger.setLevel(level)

    def _configure_exporters(self):
        """Attach batched trace/log exporters and return the periodic metric reader."""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        options = dict(
            headers=self.settings.otlp_headers,
            timeout=self.settings.export_timeout_seconds,
        )
        base = self.settings.otlp_endpoint
        self.tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f'{base}/v1/traces', **options),
                max_queue_size=2048,
                max_export_batch_size=256,
            )
        )
        self.logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=f'{base}/v1/logs', **options),
                max_queue_size=2048,
                max_export_batch_size=256,
            )
        )
        return PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f'{base}/v1/metrics', **options),
            export_interval_millis=self.settings.metric_interval_seconds * 1000,
            export_timeout_millis=self.settings.export_timeout_seconds * 1000,
        )

    def close(self) -> None:
        """Shut down telemetry and restore logging; repeated closes are harmless."""
        global _active
        if not self._started:
            return
        # Restore logging before shutting down exporters so their final diagnostics
        # cannot enqueue fresh records into the provider being closed.
        self._restore_logging()
        _active = None
        self._started = False
        self._closed = True
        for provider in (
            self.tracer_provider,
            self.meter_provider,
            self.logger_provider,
        ):
            try:
                provider.shutdown()
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    'Telemetry shutdown failed',
                    extra={
                        'event_name': 'observability.shutdown.failed',
                        'error_type': type(exc).__name__,
                    },
                )
        for handler in self._handlers:
            handler.close()


def _duration_histogram_views() -> list[View]:
    """Use second-scale buckets for the duration instruments emitted by this package."""
    # Duration instruments record seconds; SDK defaults assume much larger values.
    duration_bounds = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        30,
        60,
        120,
        300,
    )
    return [
        View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(duration_bounds),
        )
        for name in (
            'application.operation.duration',
            'http.server.request.duration',
            'chat.stream.duration',
        )
    ]
