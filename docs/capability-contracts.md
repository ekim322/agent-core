# Capability contracts

Each capability owns its configuration and resource lifetime. Applications own their tools,
request dependencies, authorization, prompts and storage adapters.

## Tool support

`tools.validation` checks exact ASCII calendar strings and two/three-letter
region-code shapes. None and non-string arguments are left for the tool schema
or caller to validate. Date-window limits count both endpoint calendar days.

`LoadTool` validates and snapshots a catalog, selects options in declared order,
filters admission and updates caller-owned `opened_tool_names` additively.
Options and guidance must reference the registered Tool object, not another tool
with the same name. Do not rename registered SDK tools. Admission of None means
no filter; an empty set permits no data tools.

`tools.visibility` composes sync/async preparation callbacks in order, carries schema
edits forward, and stops at the first denial. Preparation preview copies schema
and metadata so gates cannot alter the registered definition. Source catalogs
are snapshotted when creating a gate. Any-source and all-source rules are
alternative ways to enable a tool; caller authorization still belongs inside
actual execution.

## SQL filtering

Install the `sql` extra to import `agent_core.sql`. `VirtualTable` snapshots
records, puts declared columns first, fills absent values with null, and rejects
case-insensitive column collisions. A query may reference only the registered
relation and its lexically visible CTEs. Table functions, qualified relations,
mutations and multiple statements are rejected. DuckDB disables external access,
extension loading and disk spill, with a 256 MB query memory limit.

SQLQueryExecutor owns dedicated worker threads and bounded admission. Tables
may receive an explicit executor; otherwise they share a default with four
workers and 32 queued jobs. Excess submissions raise SQLQueryCapacityExceeded.
Close an owned executor with aclose or async with; close the shared default with
SQLQueryExecutor.close_default after stopping query callers. Closing rejects new
work and drains accepted jobs. Cancelling a close waiter leaves shutdown running.

One monotonic deadline starts before parsing and includes executor queue wait,
Arrow conversion, connection setup, execution and fetch. Queued expiration or
cancellation removes the job and its captured inputs before worker execution.
Cancellation interrupts a running connection and joins cleanup before returning;
Arrow conversion itself cannot be interrupted, so cleanup can outlast the budget.
SQL workers and shutdown do not depend on the default asyncio executor.
The sql.query operation reports outcome/duration and queue_wait_ms; worker
execution preserves the submission's observability context. The row
cap rejects oversized results rather than silently
truncating them. Omitting SQL returns all source rows. `sql_filter_notice`
distinguishes source emptiness from a query selecting no rows.

## Storage telemetry

Actual writes through BaseAgent and `LLMClient.run` are measured by
`persistence.write`. Its `operation.completed` log and span record `ok`, `error`
or `cancelled`, with duration covering the writer call. `ok` means the supplied
`ChatWriter.write` returned; the adapter owns commit and retry semantics.
Existing `agent.persistence_write_failed` and
`agent.persistence_write_cancelled` diagnostics remain available.

Write scopes inherit request/run/execution context and add available
`agent_name`, `session_id`, `message_id` and `persistence_mode`. Adapter logs
inherit these fields and the write span; scope exit restores the previous
context. IDs are log/span fields, while operation metrics use only operation
and outcome dimensions.

Background submissions emit `persistence.enqueue` completions separately.
An `ok` outcome records admission; queue overflow records `rejected` and still
raises `PersistenceQueueFull`. Calls after dispatcher closure raise before
admission. Accepted work retains the submitting context, including the active
trace parent, before the admission span starts. The worker's write span uses
that parent rather than the finished admission span. Its `queue_wait_ms` field
measures from queue insertion until the write task starts, separately from
write duration. Admission and awaited writes use null for this field so an
outer operation's queue timing cannot be mistaken for their own. Background
execution may fail after an invocation returns and
accepted records can still be lost on process exit.

With the shared runtime started and JSON stdout retained, query by service,
environment and time window, then `--field run_id=VALUE` or
`--field message_id=VALUE`. Filter `--operation persistence.write` for writer
outcomes, or `--operation persistence.enqueue` for admission. Returned trace
IDs join to exported traces when a configured backend retains them. Saved-log
CLI checks verify local retrieval; they do not establish live backend retention.

## Execution tracing

`RunState` binds timing lookup through task context; nested bindings restore
previous state and concurrent invocations cannot overwrite another task's run.
Child SDK tasks inherit the binding. Model preparation, dispatch, first output,
part durations and the whole overlapping tool batch have separate timing fields.

Within `utils.telemetry`, `run_state` owns timing and task-local lookup,
`execution` owns agent/tool operation scopes and the SDK `RunProbe`,
`live_trace` owns inspector retention/redaction, and `model_trace` projects
visible model inputs and responses. `sub_call` holds nested-call persistence IDs.
The private `_redaction` component owns credential matching and per-span text
buffers; LiveTrace owns applying retention limits and closing those buffers.
`ModelTrace` owns prepared-request inspection as well as response inspection;
provider-option interpretation is separate from the visible-input projection.
Model request scopes bind their inspector task-locally. Prepared-request capture
checks the current span identity, so nested tool scopes and discarded model spans
cannot overwrite a parent request. Nested scope exit restores the prior inspector.
Invocation tracing tracks terminal events through iterator cleanup. Closing
after DONE preserves that terminal inspector status; cleanup failures still
mark the execution as failed. Subclass forwarding shares an invocation only
within the same task; a child task calling the same agent gets its own identity.

`LiveTrace` is an optional, caller-owned inspector. Operational logs use the
installed shared `observability` package and contain identities/outcomes, not
prompts or results. Inspector inputs/results omit private reasoning/signatures
and redact configured secrets, including values split across text chunks.
An unfinished credential prefix is redacted when a span exits. Configured secret
values of any nonzero length are covered. Inspector span names and payload keys
are redacted as well.
Shared-prefix credentials wait for enough streamed input to select the longest
complete match. Replacement markers are never matched again as credential values.
Sensitive field names are case-insensitive and ignore separators, covering API
keys, access/refresh/bearer tokens and private keys as well as passwords, cookies,
authorization, secrets and credentials.

Full capture remains the default. For bounded retention, construct
`LiveTrace(max_spans=..., max_capture_chars=..., max_text_chars=...)`. Payloads
exceeding the serialized-character budget become `[truncated]`; streamed text
is capped per span and events per span follow `max_spans` when configured.
`truncated` reports any omission. These are retention limits, not a strict
allocation budget for constructing SDK inputs or serializing a payload.

## Model runtime

`ModelRuntime` owns lazily constructed route bundles; `ManagedModel` holds route
admission and an HTTP/2 stream slot until response closure. Opening failures
release admission before backoff. Capacity is shared by provider, endpoint and
Vertex project/region. URLs must use HTTP(S) and have no embedded credentials,
query or fragment. Numeric settings require finite numbers and integer counts;
provider-specific environment capacity settings take precedence over global ones.

`ProviderRequestRetryPolicy` decides transient HTTP status/read-timeout recovery
before exposing a response. Connection, proxy, write, pool and cancellation
failures are ineligible. Retry-After is finite, bounded guidance; over-budget
waits return the failure. `ModelRequestRetryPolicy` only prepares stream recovery
inputs, preserving history and visible text. The agent owns applying those plans
and never repeats tool execution as a transport retry. `InterruptedResponse`
captures history membership as a tuple and constructs fresh continuation lists;
SDK message identity is preserved, including completed tool calls and results.

Connection circuits observe actual TCP/proxy/target-TLS setup. A single recovering
probe owns the current generation; stale outcomes cannot reopen or close another
probe's circuit. Existing pooled connections remain usable during outages.
Each request's trace dispatcher composes existing callbacks around synchronous
connection observation: started callbacks run before circuit admission, and
outcome callbacks see the updated circuit. Callback errors and cancellation
propagate; only circuit admission failures are remembered as circuit rejection.
Transport read timeouts measure each response's header/body inactivity, including
HTTP/2 multiplexing and environment proxies, rather than total stream duration.
The first target-header receive event starts the header deadline; repeated trace
events cannot extend it. Prior trace callbacks, including an explicit None
value, are restored on success, failure and cancellation.

Stop model operations before closing the runtime. Shutdown is idempotent and
waiter cancellation cannot cancel owned cleanup. Every client close completes
before shutdown reports grouped failures. Partial-construction cleanup failures
are retained for shutdown. No provider credentials are needed by the tests.

The private default registry serializes acquisition, endpoint updates and
retirement. Runtime subclasses have separate defaults and configurations. A
shutdown task owns retirement even when a waiter cancels; completion detaches
the instance on success or failure. Explicitly closing the default instance
blocks acquisition until cleanup finishes, then allows a fresh instance. Endpoint
updates rejected by an opened provider leave all prior defaults intact.

## Completion and embedding clients

`LLMClient` shares ModelRuntime transport behavior for string model names.
Its optional `max_concurrency` must be a positive integer and covers both the
model call and awaited storage. Structured output uses SDK validation/correction
history; raw output uses the direct API. Closing a partially consumed stream
releases its response and capacity. Streams do not persist history.

`VectorEmbedder` validates inputs before I/O, truncates each to 7,000 characters,
and sends sequential batches of at most eight inputs and 60,000 characters.
These are approximate token budgets, with provider truncation also enabled.
Each response must contain one nonempty vector per input; invalid responses stop
before another batch is sent. Cancellation and provider failures propagate.
Injected async clients remain caller-owned; lazily created clients need aclose.

## Persistence projection

`RunRecords` commits rows and tool links only after a complete message converts.
A conversion failure logs `agent.persistence_record_failed` and skips that
message. Subsequent messages still convert. Tool returns link to the most recent
preceding successfully recorded response; reused IDs never inherit old durations.
Usage repeats on response rows, so read call-level values once per `call_id`.

Message selection uses the invocation boundary and includes emitted unrecorded
output even when an invalid slice falls back to SDK new messages. Payloads are
snapshotted, binary values become metadata summaries, and cyclic references
become `[circular]`. Word counting ignores cyclic edges. Extremely deep acyclic
payloads can still exceed Python's recursion limit during storage projection and
cause that message to be skipped. Writers own storage retries and durability.

Invocation dispatch contains record-assembly failures and awaits storage by
default. Awaited storage failures propagate and emit
`agent.persistence_write_failed`. Optional background writes use a dispatcher
owned by BaseAgent, with
background_write_concurrency=4 active writes and background_write_queue_size=128
queued batches by default. Ordinary ChatWriter adapters work unchanged. Admission
is immediate: full queues raise PersistenceQueueFull and emit
agent.persistence_queue_full; closed dispatchers reject submissions. Admission
errors can raise after a stream's terminal event, as awaited writer errors do.

Accepted writes inherit each submitting execution's context. Write failures emit
the same agent.persistence_write_failed event and do not fail the already-returned
caller. Adapter-cancelled writes emit agent.persistence_write_cancelled and do
not stop workers. No storage retry is added. drain_persistence waits for accepted
work without closing admission; aclose stops admission, drains and stops workers.
Use the agent and these lifecycle methods on the same event loop. Cancelling a
close waiter leaves shutdown running. Stop invocations before closing; model
runtime, toolset and storage-adapter lifetimes remain caller-owned. Writes remain
best effort under process loss, without a durability guarantee.

`InMemoryChatWriter` takes a deep snapshot on each completed write. Later edits
to the submitted builder, event payloads or metadata cannot change stored batches.
Its public `batches` remain editable for inspection. `iter_events()` reads stored
event objects in batch order without creating a combined list; `all_events`
materializes that same view. Memory storage remains unbounded and process-local.

## Console output and constants

`StreamPrinter(output=..., color=...)` supports an injected text stream and
TTY-aware color. StreamEvent constructors project typed payload records
shallowly, retaining SDK/object references and explicit null fields. The mutable
`data` dictionary accepts caller extensions, including arbitrary status keys.

`models.catalog.ModelId` and `ReasoningEffort` are enumerable string enums of
curated model identifiers and effort labels. For example, `ModelId.GPT_6_LUNA`
can be supplied where a model-name string is expected. The model shortlist and
its September 2026 review sources are in [model catalog](model-catalog.md).
Applications choose their own defaults.
Use enum members or explicit strings; membership does not establish provider
availability.
