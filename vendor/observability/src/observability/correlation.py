"""Keep chat/request IDs available to logging and tracing in the current task.

Binding stores fields in memory. The log formatter reads them when a log is
emitted; observe_operation copies them onto a span when that operation starts.
OpenTelemetry separately owns the active trace and span IDs.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from opentelemetry import trace

_context: ContextVar[dict[str, object]] = ContextVar(
    'observability_context', default={}
)


def current_observability_context() -> dict[str, object]:
    """Return a copy of the correlation fields attached to the current task."""
    return dict(_context.get())


@contextmanager
def bind_observability_context(**fields: object) -> Iterator[None]:
    """Make IDs available to logs and observed operations inside this block.

    For example, bind conversation_id once when starting a chat run. Logs from
    functions called inside the block then include it without passing that ID
    to every logger. observe_operation also copies these fields onto new spans;
    binding alone emits no logs, creates no spans, and changes no existing span.

    Fields survive ordinary calls and await. New asyncio tasks inherit the
    current fields unless created with an explicit fresh context. Nested blocks
    can add or override fields; exiting restores the caller's previous fields,
    including on failure. Independently created tasks keep their own fields.
    Use safe identifiers here, never message contents or credentials.
    """
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def current_trace_ids() -> dict[str, str]:
    """Read the active OpenTelemetry span's IDs, or return empty if none exists."""
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return {}
    return {
        'trace_id': format(context.trace_id, '032x'),
        'span_id': format(context.span_id, '016x'),
    }
