# Using observability

Start with the [application instrumentation example](../README.md#shared-workflow-follow-one-request-through-three-files) and
[console example](../examples/basic.py).

## Public API

Import `Observability`, `ObservabilitySettings`, `bind_observability_context`, and
`observe_operation` from `observability`. The package also exports
`ObservedOperation`, `current_trace_ids`, `get_tracer`, and `get_meter`.
Import the optional HTTP middleware from `observability.integrations.asgi`.

## Runtime and correlation

`Observability()` reads environment settings at construction. Passing an explicit
`ObservabilitySettings(...)` uses those settings instead. Call `start()` once during
application startup, finish application tasks, then call `close()` in cleanup.
Repeated starts while running and repeated closes are harmless. Create a new runtime
when starting again after close or failed startup. Supplied span/log processors and
metric readers are owned and shut down by the runtime.

Startup replaces root and Uvicorn logging, suppresses Uvicorn access logs, and quiets
transport loggers. Close restores captured logger settings before exporter shutdown.
Avoid reconfiguring logging while the runtime is active.

`bind_observability_context(run_id=..., request_id=...)` attaches fields to the current
context; it emits no records. Python logger calls emit events. `observe_operation`
starts a span and emits one completion log, counter update, and duration observation.
Use stable operation names because operation/outcome are metric dimensions.

Bound fields override log extras; standard fields and active trace IDs take precedence
in formatted records. Nested context blocks restore previous fields on exit, including
failure. Ordinary async calls and newly created tasks inherit context. Separate
processes and durable jobs need explicit propagation of safe IDs.

For errors caught inside an operation, call `record_exception(exc)` on the yielded
object, or `set_outcome('error')`. Unhandled exceptions and cancellation propagate
and record their outcomes. Other allowed outcomes are `ok`, `cancelled`, and `rejected`.
Use `new_root=True` for a detached operation whose new trace links to the active span.
It does not persist job context. `success_log_level=logging.DEBUG` reduces successful
completion logs while preserving spans and metrics; errors still log at ERROR.

Providers are runtime-owned. Use `get_tracer()`/`get_meter()` or explicitly pass
`runtime.tracer_provider` and `runtime.meter_provider` to third-party instrumentation.
Startup does not install these providers into OpenTelemetry's global accessors.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `OTEL_SERVICE_NAME` | `application` | Set a distinct application name. |
| `OBSERVABILITY_ENVIRONMENT` | `development` | Included in logs and provider resource attributes. |
| `OBSERVABILITY_SERVICE_VERSION` | `unknown` | Application version in logs and resource attributes. |
| `OBSERVABILITY_LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, or CRITICAL. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | HTTP(S) base URL; paths `/v1/logs`, `/v1/traces`, `/v1/metrics` are appended. |
| `OTEL_EXPORTER_OTLP_HEADERS` | empty | Comma-separated `key=value` pairs; keys/values are URL-decoded. |
| `OBSERVABILITY_EXPORT_TIMEOUT_SECONDS` | `5` | Greater than zero and at most 30 seconds. |
| `OBSERVABILITY_METRIC_INTERVAL_SECONDS` | `30` | At least one second between metric exports. |

A blank endpoint disables OTLP export. Logs go to stdout by default; supply `stream=`
to use another text stream. There is no local span/metric database. An endpoint must
have no embedded credentials, query, or fragment. Supply credentials through headers.
Exporters run in background workers; batch queues are bounded. Delivery and retention
depend on the collector/backend and are not guaranteed by completion logs.

## HTTP applications

The `asgi` extra adds Starlette support for automatically instrumenting incoming
HTTP requests in FastAPI/Starlette applications. The base package still supports
explicit operation instrumentation in web applications. Outgoing calls made with
`requests` or `httpx` do not require this extra.

Install the `asgi` extra and your server separately, then add the middleware.
For a Starlette application:

```python
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from observability import Observability
from observability.integrations.asgi import HTTPObservabilityMiddleware

@asynccontextmanager
async def lifespan(app):
    runtime = Observability()
    runtime.start()
    try:
        yield
    finally:
        runtime.close()

async def health(request):
    return JSONResponse({"status": "ok"})

app = Starlette(routes=[Route("/health", health)], lifespan=lifespan)
app.add_middleware(HTTPObservabilityMiddleware, quiet_routes=frozenset({"/health"}))
```

The consuming project launches `app` with its ASGI server and stops its own tasks
before lifespan cleanup. Middleware generates a fresh request ID, returns it as
`X-Request-ID`, and starts a new server trace. It ignores inbound request IDs and
trace parents. Durations cover the full streamed response. Non-HTTP scopes pass
through. Successful GET requests on quiet routes emit DEBUG completion logs while
still recording spans and metrics.

An optional `request_fields(scope)` callback adds safe resource identifiers to the
completion log and span after handling. Use established application state; those
fields do not automatically attach to logs emitted earlier in the handler. IDs are
correlation hints, not authorization. Route templates, method, status, and outcome
are metric dimensions; avoid dynamic route templates.
If this callback raises an ordinary exception, middleware emits
`http.request_fields.failed` with its type and omits the optional fields.
The application's response, exception or cancellation remains authoritative;
completion logs, spans and metrics still describe that outcome. Exception text
is omitted from the diagnostic.

## Investigate saved logs

```sh
python -m observability logs --file /tmp/observability-demo.jsonl --field run_id=example --pretty
python -m observability logs --file /tmp/observability-demo.jsonl --severity ERROR \
  --since 2026-10-05T00:00:00-04:00 --until 2026-10-06T00:00:00-04:00
python -m observability summary --file /tmp/observability-demo.jsonl --operation demo.work
python -m observability --help
```

The CLI reads a finite UTF-8 JSONL file or stdin until EOF. It skips malformed lines,
non-object JSON, and records without a timestamp. Field filters match string values exactly. Use repeated `--field NAME=VALUE`
for application-defined identifiers; all filters must match. Values may contain
`=` or be empty. Dedicated flags cover service, environment, severity, operation,
request/trace ID, event name, and outcome. Numeric/boolean fields are not coerced
to strings.
Time bounds require a timezone; `since` is inclusive and `until` exclusive.

`logs` retains the final 50 matches in source order by default. Set `--limit` from
1 to 10,000. `summary` retains the final 100,000 matching records by default; set
`--max-records` from 1 to 1,000,000. Both report `matched_records` and `truncated`.
Input is fully scanned even when output is bounded. A summary uses only valid,
nonnegative durations from retained `operation.completed` logs, grouped by service,
environment, operation, and outcome. Percentiles are nearest-rank values. Other
matching records can consume the retention window, so filtering by
`--event-name operation.completed` can be useful.

Start from service/environment and a time window, then filter a known run/request
ID. Use a returned trace ID to join to traces in your configured backend. The CLI
cannot query that backend; retention, sampling, collector health, and backend query
access must be established separately. Missing records alone cannot establish success.

## Privacy and limits

Keep message text static and fields safe. Sensitive field names are redacted,
strings/collections/nesting are bounded, and exception diagnostics keep types and
code locations while omitting messages and locals. Arbitrary message text is not
sanitized. Never put credentials or raw user/model payloads into messages or attributes.
IDs belong in logs/spans, not metric dimensions. Only instrumented boundaries have
individual timings.

## Share the library

Commit and tag releases in this standalone repository, then publish to your chosen
private remote or package index. For a private submodule, consumers need source
access, should install the checkout as a local dependency, and should pin its commit
and update dependency locks together. The current manifest declares version 0.1.0;
it does not establish that this version is published to a package index.
