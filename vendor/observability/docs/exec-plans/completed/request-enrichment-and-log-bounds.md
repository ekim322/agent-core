# Contain enrichment failures and bound log traversal

## Purpose / Big Picture

Preserve HTTP application outcomes when optional correlation callbacks fail,
and bound the work required to sanitize structured mapping fields.

## Progress

- [x] 2026-10-05: Review identified unguarded callback in HTTP completion cleanup
  and full mapping materialization before truncation.
- [x] 2026-10-05: Contain ordinary callback failures and use islice for mappings.
- [x] 2026-10-05: Add response/error/cancellation and traversal regression tests.
- [x] 2026-10-05: Shared regression suite: 33 passed. Whitespace check passed.

## Surprises & Discoveries

Callback execution in finally could replace the original exception or cancellation.
The previous list(value.items())[:30] visited every entry despite returning 30.

## Decision Log

2026-10-05: Keep application outcome authoritative. Callback failures log a fixed
diagnostic with exception type, omit optional fields, and retain completion logs,
metrics and spans. Catch Exception, preserving process-level control exceptions.

## Outcomes & Retrospective

Both fixes and their regression checks are complete. The shared suite passed all
33 tests using the existing Python 3.13 dependency environment. No live exporter
or provider calls were made. Remote publication remains separate.

## Context and Orientation

src/observability/integrations/asgi.py owns optional HTTP correlation enrichment.
src/observability/log_records.py owns safe_value record sanitization.

## Plan of Work

Apply the two scoped fixes, verify failure/cancellation ownership and bounded
mapping traversal, then run existing shared tests and retain API/privacy contracts.

## Concrete Steps

Run tests from this repository with a dependency-ready Python environment and
PYTHONPATH=src. Inspect git diff --check and update architecture/usage constraints.

## Validation and Acceptance

tests/test_asgi_failures.py verifies unchanged streamed responses, identical
application exceptions/cancellation, joined diagnostic/completion records,
span/metric outcomes and omitted private exception text. tests/test_log_bounds.py
checks exactly 30 entries are visited in a 10,000-entry mapping.

## Idempotence and Recovery

No persisted data or remote state changes. Edits can be reviewed against the
starting submodule commit. Do not discard unrelated checkout changes.

## Artifacts and Notes

This work is developed in agent-core/vendor/observability; retain a local commit
for the parent submodule pointer. Publication is separate from local validation.

## Interfaces and Dependencies

Public signatures and telemetry fields remain unchanged. A new diagnostic event,
http.request_fields.failed, identifies contained callback failures without text.
