# Design principles

Agent Core is application-independent infrastructure. Applications provide
their own agents, prompts, tools, authorization rules and concrete storage
adapters. The package owns reusable execution, model, persistence and telemetry
mechanisms.

## Interfaces and behavior

Keep public contracts explicit. Retain useful names and interfaces, and change
them when a clearer design justifies it. Update affected callers, exports,
examples and documentation together. Avoid compatibility layers that preserve
interfaces with no continuing use.

Preserve cancellation propagation, cleanup order, invocation isolation, result
ordering and truthful success or failure outcomes. Keep retry ownership clear:
the model runtime owns admission, transport and pre-response retries; agent
execution owns applying stream-recovery decisions. Recovery must not repeat
tool side effects.

Persistence adapters remain caller-owned. Awaited writes propagate failures;
optional background writes are best effort and have no process-loss durability
guarantee. Generic observability is provided by the pinned
`vendor/observability` package. Agent-specific execution tracing remains in
`src/agent_core/utils/telemetry`.

## Testing

Tests use fake models or local servers and do not require provider credentials.
Add contract coverage for changed public or lifecycle boundaries. Tests should
check observable behavior, cancellation and cleanup rather than internal helper
structure alone. Record the checks that actually ran and any outstanding work
in the relevant change summary.
