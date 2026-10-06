"""Measure application boundaries, without capturing arguments or results."""

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import logging
from time import perf_counter

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Link, Status, StatusCode

from .correlation import current_observability_context
from .log_records import exception_details, safe_value
from .runtime import get_meter, get_tracer

# Keep instrumentation identity stable across the module rename.
_INSTRUMENTATION_NAME = 'observability.operations'
_logger = logging.getLogger(_INSTRUMENTATION_NAME)


class ObservedOperation:
    """Add span attributes or report handled failures inside an observed operation.

    The surrounding context manager emits the completion log and metrics when
    the block exits. Changes here describe that operation's eventual result.
    """

    def __init__(self, span):
        self.span = span
        self.outcome = 'ok'
        self.failure: dict[str, object] = {}

    def set_outcome(self, outcome: str) -> None:
        """Set the result used by the completion log, span, and metric labels."""
        if outcome not in {'ok', 'error', 'cancelled', 'rejected'}:
            raise ValueError(
                'Operation outcome must be ok, error, cancelled, or rejected'
            )
        self.outcome = outcome
        if outcome == 'error':
            self.span.set_status(Status(StatusCode.ERROR))

    def set_attribute(self, key: str, value: object) -> None:
        """Add a sanitized scalar to this span only, leaving log fields unchanged."""
        safe = safe_value(key, value)
        if isinstance(safe, (str, int, float, bool)):
            self.span.set_attribute(key, safe)

    def record_exception(self, exc: BaseException) -> None:
        """Mark failure and retain its type and code locations for diagnostics.

        Call this when catching an exception inside the observed block so its
        completion still reports an error. Exception text and locals are omitted.
        """
        self.set_outcome('error')
        self.failure = exception_details(exc)
        self.span.add_event('exception', {'exception.type': type(exc).__name__})


@contextmanager
def observe_operation(
    name: str,
    *,
    attributes: Mapping[str, object] | None = None,
    new_root: bool = False,
    success_log_level: int = logging.INFO,
) -> Iterator[ObservedOperation]:
    """Measure a block of work and report how long it took and how it ended.

    With an active Observability runtime, entering starts an OpenTelemetry span
    and copies the bound correlation fields plus supplied attributes onto it.
    Logs emitted inside the block include that active span's trace/span IDs.
    On exit, emit one completion log and record a count and duration metric.
    Metric labels use the operation name and outcome. Choose a stable name;
    request and run IDs belong in attributes, not operation names.

    Nested operations become child spans by default. Use new_root=True for a
    detached job: its new trace links to the scheduling span if one is active.
    The caller must propagate safe IDs when work crosses processes or durable
    queues; this helper only uses the context available in the current task.

    Unhandled exceptions and cancellation set the outcome and propagate. For
    failures caught inside the block, use the yielded object's record_exception
    or set_outcome method. Completion is recorded before the span closes, so
    its log shares the operation's trace and span IDs. Duration covers the
    yielded block, excluding attribute setup and completion emission.

    success_log_level controls successful completion logs; errors log at ERROR
    and cancelled/rejected work at INFO. Spans and metrics still record every
    outcome. Start the runtime to enable spans and metrics and configure output;
    the destination owns retention.
    """
    parent = trace.get_current_span().get_span_context()
    links = [Link(parent)] if new_root and parent.is_valid else []
    tracer = get_tracer(_INSTRUMENTATION_NAME)
    with tracer.start_as_current_span(
        name,
        context=Context() if new_root else None,
        links=links,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        observed = ObservedOperation(span)
        for key, value in {
            **current_observability_context(),
            **(attributes or {}),
        }.items():
            observed.set_attribute(key, value)
        started = perf_counter()
        try:
            yield observed
        except asyncio.CancelledError:
            observed.set_outcome('cancelled')
            raise
        except Exception as exc:
            observed.record_exception(exc)
            raise
        finally:
            duration = perf_counter() - started
            _record_operation_completion(
                name, span, observed, duration, success_log_level
            )


def _record_operation_completion(
    name: str,
    span: trace.Span,
    observed: ObservedOperation,
    duration_seconds: float,
    success_log_level: int,
) -> None:
    """Emit metrics, span outcome, and the completion log inside the active span."""
    labels = {'operation': name, 'outcome': observed.outcome}
    meter = get_meter(_INSTRUMENTATION_NAME)
    meter.create_counter('application.operation.count', unit='{operation}').add(
        1, labels
    )
    meter.create_histogram('application.operation.duration', unit='s').record(
        duration_seconds, labels
    )
    span.set_attribute('outcome', observed.outcome)
    level = success_log_level if observed.outcome == 'ok' else logging.INFO
    if observed.outcome == 'error':
        level = logging.ERROR
    _logger.log(
        level,
        'Application operation completed',
        extra={
            'event_name': 'operation.completed',
            **labels,
            'duration_ms': round(duration_seconds * 1000, 3),
            **observed.failure,
        },
    )
