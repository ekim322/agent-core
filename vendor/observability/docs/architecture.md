# Architecture

This package provides application instrumentation and a saved-log query CLI.
The consuming application owns business decisions, permissions, task scheduling,
HTTP endpoints, deployment, and telemetry storage.

## Source map

| Location | Responsibility |
| --- | --- |
| `src/observability/__init__.py` | Public runtime, settings, context, and operation exports. |
| `src/observability/settings.py` | Validated settings and environment parsing. |
| `src/observability/runtime.py` | One active runtime, providers, exporter workers, handlers, and logger restoration. |
| `src/observability/correlation.py` | Task-local fields and current OpenTelemetry trace IDs. |
| `src/observability/log_records.py` | Shared sanitized record format for console and OTLP logs. |
| `src/observability/operation_telemetry.py` | Timed application spans, completion events, and outcome metrics. |
| `src/observability/cli.py` | Terminal arguments, file/stdin handling, JSON output, and exit codes. |
| `src/observability/log_queries.py` | Finite-input JSONL filtering, retained-record summaries, and aggregation. |
| `src/observability/__main__.py` | Module CLI entry point; the console entry point calls `cli.main` directly. |
| `src/observability/integrations/asgi.py` | Optional ASGI request correlation and full-response telemetry. |
| `examples/basic.py` | Console-only runnable usage. |
| `tests/test_shared_core.py` | Runtime, correlation, privacy, outcomes, and intercepted exporter tests. |
| `tests/test_shared_environment.py` | Environment parsing contracts. |
| `tests/test_shared_query.py` | Saved-log query behavior. |
| `.agents/skills/` | Contributor workflows; excluded from the Python wheel. |

The package groups runtime configuration, correlation, telemetry, and log queries
in separate modules.
`operation_telemetry` uses runtime providers and record sanitization; runtime uses settings
and log records; log records use correlation. Core modules have no Starlette dependency.
ASGI depends on the public core API and Starlette types. Querying uses standard
library code and interprets the emitted record contract.

## Follow an operation

The application constructs and starts `Observability`. Startup creates providers,
attaches processors/readers, then saves and replaces logging configuration. Only
after setup succeeds does the runtime become active. Failed startup restores
logging and attempts to shut down created providers.

A bound context contributes correlation fields. `observe_operation` copies those
fields onto a span; logger calls format records using current context and active
span IDs. On block exit, completion logs and metrics are emitted before the span
closes. Close restores logging before shutting down providers to prevent final
exporter diagnostics from entering the closing log pipeline.

Console JSON and OTLP logs use the same formatter. Export-handler diagnostics stay
on console rather than recursively entering export. Backend storage owns delivery,
retention, and live queries. `log_queries.py` only scans logs that have already been saved.

For another framework adapter, add it beneath `integrations/`, use the public
context/provider helpers, and keep its dependency optional. A formatting/privacy
change belongs in `log_records.py`; an exporter/lifecycle change belongs in `runtime.py`.
Application field filters use repeatable `--field NAME=VALUE` arguments; consumers
define their fields without adding domain-specific options to this library.

## Integration boundaries

The consuming application owns its telemetry collector, backend, dashboards, and
query service. Development environments also own service identity, ports, storage,
and teardown. This library supplies instrumentation and queries over saved logs.

## Reliability and verification limits

- ASGI tests cover failed enrichment during streaming success, application error
  and cancellation. Disconnect and quiet-route policy behavior still require
  focused verification when changing the integration.
- Intercepted exporter tests verify protobuf encoding and HTTP paths. Live
  ingestion, retention, and retrieval depend on the consumer's telemetry stack
  and require separate integration verification.
- Runtime lifecycle requires application-controlled startup and shutdown. It has
  no synchronization for competing starts from different threads.
- CLI output is bounded by record count, but input lines are fully parsed; record
  size and total scan time are not bounded. Use trusted finite diagnostic files.

See [usage](usage.md) for telemetry privacy, propagation, and query constraints,
and [development](development.md) for contributor checks.
