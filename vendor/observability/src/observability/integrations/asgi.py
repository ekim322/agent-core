"""Measure HTTP requests without buffering bodies or interrupting streaming.

Only route templates and bounded HTTP attributes become metric labels. Request
IDs are generated here; public callers cannot choose another request's identity
or trace parent. Requested resource identifiers are correlation hints, not proof
that the request was authorized to read that resource.
"""

import asyncio
import logging
from collections.abc import Callable
from time import monotonic
from uuid import uuid4

from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from observability import (
    bind_observability_context,
    current_trace_ids,
    get_meter,
    get_tracer,
)

logger = logging.getLogger(__name__)
_METHODS = frozenset(
    {'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS', 'CONNECT', 'TRACE'}
)


class HTTPObservabilityMiddleware:
    """Connect logs from one HTTP request and measure its full response lifetime.

    Generate and bind a request ID, return it in X-Request-ID, and start a server
    span so logs emitted during handling share request and trace identifiers.
    On completion, record duration, status, and outcome in a log and metric.
    Streaming requests are timed through the response body, not just headers.

    Non-HTTP scopes pass through unchanged. The request starts a fresh trace
    and ignores inbound request IDs. Successful GET requests on quiet_routes
    log at DEBUG while still recording spans and duration metrics.

    request_fields runs after handling, when route and application state are
    available. Return only safe fields: they are added directly to the span
    and completion log, and do not attach to earlier handler logs. Callback
    failures emit a diagnostic and omit those fields while preserving the
    application's result, exception, or cancellation and completion telemetry.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        request_fields: Callable[[Scope], dict[str, str]] | None = None,
        quiet_routes: frozenset[str] = frozenset(),
    ):
        self.app = app
        self.request_fields = request_fields
        self.quiet_routes = quiet_routes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault('state', {})['request_id'] = request_id
        method = scope['method'] if scope['method'] in _METHODS else '_OTHER'
        # Resolve instruments after startup configures the process providers.
        tracer = get_tracer(__name__)
        request_duration = get_meter(__name__).create_histogram(
            'http.server.request.duration',
            unit='s',
            description='HTTP response duration, including streamed response bodies.',
        )
        started = monotonic()
        status_code = 500
        disconnected = False
        response_completed = False
        error_type = None

        async def receive_with_disconnect_tracking() -> Message:
            nonlocal disconnected
            message = await receive()
            if message['type'] == 'http.disconnect':
                disconnected = True
            return message

        async def send_with_response_tracking(message: Message) -> None:
            nonlocal status_code, response_completed
            if message['type'] == 'http.response.start':
                status_code = message['status']
                message = {
                    **message,
                    'headers': [
                        (name, value)
                        for name, value in message.get('headers', [])
                        if name.lower() != b'x-request-id'
                    ]
                    + [(b'x-request-id', request_id.encode('ascii'))],
                }
            await send(message)
            if message['type'] == 'http.response.body' and not message.get(
                'more_body', False
            ):
                response_completed = True

        with (
            bind_observability_context(request_id=request_id),
            tracer.start_as_current_span(
                method,
                context=Context(),
                kind=SpanKind.SERVER,
                record_exception=False,
                set_status_on_exception=False,
            ) as span,
        ):
            scope['state']['telemetry_trace'] = current_trace_ids()
            try:
                await self.app(
                    scope, receive_with_disconnect_tracking, send_with_response_tracking
                )
            except asyncio.CancelledError:
                disconnected = True
                raise
            except Exception as exc:
                error_type = type(exc).__name__
                raise
            finally:
                route = getattr(scope.get('route'), 'path', 'unmatched')
                duration_seconds = monotonic() - started
                outcome = 'error' if error_type or status_code >= 500 else 'ok'
                if disconnected and not response_completed and not error_type:
                    outcome = 'cancelled'
                attributes = {
                    'http.request.method': method,
                    'http.route': route,
                    'http.response.status_code': status_code,
                }
                request_fields = self._request_fields_for(scope)
                span.update_name(f'{method} {route}')
                span.set_attributes(
                    {
                        **attributes,
                        **request_fields,
                        'request_id': request_id,
                        'outcome': outcome,
                    }
                )
                if outcome == 'error':
                    span.set_status(StatusCode.ERROR)
                if error_type:
                    span.set_attribute('error.type', error_type)
                request_duration.record(
                    duration_seconds, attributes={**attributes, 'outcome': outcome}
                )
                fields = {
                    'event_name': 'operation.completed',
                    'operation': 'http.request',
                    'method': method,
                    'route': route,
                    'status_code': status_code,
                    'duration_ms': round(duration_seconds * 1000, 3),
                    'outcome': outcome,
                    'request_id': request_id,
                    **request_fields,
                }
                if error_type:
                    fields['error_type'] = error_type
                level = logging.ERROR if outcome == 'error' else logging.INFO
                # Polls and health checks remain measurable without flooding logs.
                if (
                    outcome == 'ok'
                    and status_code < 400
                    and method == 'GET'
                    and route in self.quiet_routes
                ):
                    level = logging.DEBUG
                logger.log(level, 'HTTP request completed', extra=fields)

    def _request_fields_for(self, scope: Scope) -> dict[str, str]:
        if self.request_fields is None:
            return {}
        try:
            return self.request_fields(scope)
        except Exception as exc:
            # Optional diagnostic enrichment must preserve the request's outcome.
            # Callback exception text can contain private request data.
            logger.warning(
                'HTTP request correlation callback failed',
                extra={
                    'event_name': 'http.request_fields.failed',
                    'error_type': type(exc).__name__,
                },
            )
            return {}
