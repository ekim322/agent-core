# Agent Core

Private, reusable Python infrastructure for building asynchronous agents.
Import from `agent_core`; keep your application agents in your own package.

## Package scope

This package provides reusable asynchronous agent infrastructure. Applications
define their own agents, prompts, tools, authorization and storage adapters.
Implementations should keep responsibilities explicit, propagate cancellation,
and preserve clear ownership of retries and resource cleanup. See
[design principles](docs/design-principles.md) and
[capability contracts](docs/capability-contracts.md) before changing those
boundaries.

## Install and verify

Python 3.12 or newer is required. From this repository directory:

```sh
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e './vendor/observability' -e '.[dev,asgi,sql]'
python -m pytest -q
python -m hatchling build
```

Consumers install both local checkouts with
`python -m pip install -e ./agent-core/vendor/observability -e ./agent-core`
or declare equivalent local-path dependencies in their package manager. The
observability submodule pins the shared implementation; install that checkout
alongside agent-core rather than resolving the same-named package from an index.
The repository's uv source configuration selects the submodule automatically.
Use `[asgi]` for the optional ASGI middleware and `[sql]` for SQL filtering.
Persistence is awaited by default. `BaseAgent` callers may opt into
`persist_in_background=True` for best-effort writes that can be lost at shutdown.
Each BaseAgent owns bounded background storage: four active writes and 128
queued batches by default. Configure `background_write_concurrency` and
`background_write_queue_size` at construction. A full queue raises
`PersistenceQueueFull`; background write failures emit
`agent.persistence_write_failed` with the originating execution context. Stop
invocations, then await `agent.aclose()` to drain accepted writes and stop its
workers. `agent.drain_persistence()` waits without closing admission. Ordinary
ChatWriter adapters work with both storage modes; adapters remain caller-owned.
`LLMClient.run` awaits its writer and propagates storage failures.

The package does not load environment files or configure telemetry on import.

## Extend BaseAgent

```python
import asyncio
from agent_core import BaseAgent
from pydantic_ai.models.test import TestModel

class GreetingAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model=model, system_prompt="Answer clearly and briefly.")

async def main():
    agent = GreetingAgent(TestModel(custom_output_text="Hello"))
    result = await agent.run("Say hello", persist=False)
    print(result.output)

asyncio.run(main())
```

Built-in model providers are OpenAI, Anthropic, and Google. Pass provider model
names or caller-supplied model objects in real applications. Register tools
at construction or on `agent.agent`. Supply typed per-run dependencies through
`deps_type` and `deps`; no application services are imported by this package.
For request envelopes, override `_prepare_run` to return `PreparedRequest`.
Both `run(request=...)` and `stream(request=...)` use that preparation.
`stream()` yields `StreamEvent` records; close partially consumed streams with
`contextlib.aclosing`. Override `_handle_node`, `_map_request_event`, or
`_map_tool_event` only when you need to customize those behaviors.

Pydantic AI is pinned to 2.54.0. Pass `toolsets` at construction for reusable SDK
tool collections, or per call to replace them. `retries` accepts an integer or
`{"tools": 2, "output": 1}` for separate validation retry budgets;
`model_request_retries` controls interrupted-response recovery.
Tools can raise Pydantic AI's `ModelRetry` to request a correction within the
tool retry budget; other tool exceptions return recoverable errors to the model.
Pass `UsageLimits` from `pydantic_ai.usage` as `usage_limits` to `run` or `stream`
to cap SDK-accounted requests, tokens, tool calls or cost across recovery and
finalization. Without explicit usage limits, `max_hops` owns the model-turn cap
and triggers an answer with tools disabled. An explicit usage limit raises on
exhaustion and may prevent that final answer. Provider transport retries are
outside SDK request accounting.

## Ownership and lifecycle

- `agent/base.py`: public invocation API, preparation, subclass hooks and cleanup.
  `request.py` defines caller-prepared inputs.
- `streaming.py`: shared event contracts and console rendering for agents and
  direct model clients.
- `agent/_execution/`: private invocation state, model/tool loop, node streaming,
  event mapping, tool overrides and persistence dispatch. `recovery.py` plans
  replay/continuation of an interrupted response. These modules do not import
  BaseAgent. Execution prompts live alongside the loop and recovery policy.
- `models/`: `completion.py` provides single-pass completions; `embeddings.py`
  provides vectors; `config.py` translates model options; `catalog.py` holds
  enumerable `ModelId` and `ReasoningEffort` strings. See the curated
  [model shortlist](docs/model-catalog.md) and its review sources. `runtime/` owns
  connections, capacity, transport
  and pre-response retry policy, independently of agent execution. Its private
  `_default.py` registry owns process-default configuration and retirement.
- `tools/`: `catalog.py` selects/discloses registered tools; `visibility.py`
  composes preparation policies; `validation.py` checks argument formats;
  `outcomes.py` identifies recoverable tool errors.
- `sql/table.py`: bounded SQL filtering over caller-supplied in-memory records.
  `sql/executor.py` owns dedicated workers, bounded admission and queue deadlines.
- `persistence/`: records and the `ChatWriter` storage interface; applications
  implement the adapter. `InMemoryChatWriter` snapshots submitted batches for
  tests and notebooks; `iter_events()` inspects them without flattening a list.
- `utils/telemetry/`: execution state, nested calls and live traces. The private
  `_redaction.py` owns credential matching and independent text-stream buffers.
- `vendor/observability/`: shared logging/OpenTelemetry startup, context,
  operation timing, query CLI and optional ASGI integration.

Use top-level imports such as `from agent_core import BaseAgent, ModelRuntime`
for the main public interfaces. Capability-specific imports follow the paths
above, for example `from agent_core.tools.catalog import LoadTool`. Logger and
meter namespaces use stable identities so saved queries and instrumentation
scopes continue to match.

String model names use the process-default `ModelRuntime`. After all work stops,
await `ModelRuntime.close_default()`. Each runtime subclass owns a separate
default and endpoint configuration. After an explicit default-instance close
finishes, acquisition creates a fresh instance; acquisition during shutdown
raises. To own an isolated runtime, use
`async with ModelRuntime() as runtime`, build a model with
`ModelRuntime.make_model(name, runtime=runtime)`, and pass that object to the agent.
Provider credentials and runtime settings come from environment configuration;
see `models/runtime/settings.py` and `providers.py` for supported settings.
Constructing provider models may require credentials before a request is sent.

See [capability contracts](docs/capability-contracts.md) for tool catalog/gating,
SQL filtering, trace retention, retry eligibility and runtime cleanup behavior.

SQL tables share a dedicated executor with four workers and 32 queued jobs by
default; excess work raises `SQLQueryCapacityExceeded`. The query deadline
includes parsing, queue wait and execution. Call
`await SQLQueryExecutor.close_default()` from `agent_core.sql` at shutdown, or
own a custom executor:

```python
from agent_core.sql import SQLQueryExecutor, VirtualTable

async with SQLQueryExecutor(max_workers=4, max_queued=16) as executor:
    table = VirtualTable("rows", [{"value": 1}], executor=executor)
    rows = await table.run_sql_query("select * from rows", timeout_seconds=2)
```

Import `Observability` and `ObservabilitySettings` from `observability`.
Start `Observability(ObservabilitySettings(service_name="your-service"))`
explicitly and close it at application shutdown. Only one observability runtime
may be active per process. Configure OTLP export through validated settings.
Applications supply their service identity; the shared default is `application`.
Agent instrumentation uses that same runtime and task-local context. Use
`python -m observability --help` for structured-log investigation. Filter agent
records with `--field run_id=VALUE` or `--field tool_call_id=VALUE`. Field filters
match exact string values. Generic completion logs and instrumentation scopes
use the shared `observability.operations` name.

Storage emits `operation.completed` records for `persistence.write` in both
BaseAgent storage modes and in `LLMClient.run`. These include duration, outcome,
available session/message IDs and `persistence_mode` (`awaited` or `background`).
Background records also include `queue_wait_ms`, measured before writing begins.
`persistence.enqueue` reports queue admission separately: `ok` means accepted,
and `rejected` means the queue was full. To inspect actual storage completion in
saved JSON logs:

```sh
python -m observability logs --file requests.jsonl \
  --operation persistence.write --field message_id=VALUE --pretty
```

See [storage telemetry contracts](docs/capability-contracts.md#storage-telemetry)
for outcome and correlation details. The application must retain stdout or
configure a telemetry backend to retrieve these records after execution.

The agent-core wheel declares the shared package dependency; it does not contain
the submodule's source. Install the shared package from the pinned checkout or
its corresponding wheel as well.

## Agent development skills

`.agents/skills/` contains reusable development skills with their references and
templates. These files are repository tooling, not part of the Python wheel.
The root `AGENTS.md` provides this package's navigation and boundaries.
