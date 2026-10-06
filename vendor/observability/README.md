# Observability

Connect logs from every file to one request ID. Measure the steps that matter.
Query saved records to find errors and slow work. Give the same evidence to a
coding agent. This follows [OpenAI's harness-engineering approach](https://openai.com/index/harness-engineering/):
make application behavior visible to people and agents.

## Install

Use Python 3.12 or later. Clone this repository or add it as a submodule.
Install it in your application's Python environment:

```sh
python -m pip install -e /path/to/observability
```

For FastAPI or Starlette request recording, install the optional dependency:

```sh
python -m pip install -e '/path/to/observability[asgi]'
```

Then add the [request middleware](docs/usage.md#http-applications).
This option records incoming HTTP requests. Other Python work can use the base install.

## Two tools with different jobs

`bind_observability_context(request_id="req-123")` connects records.
It adds the request ID to logs written inside the block, including logs from
functions in other files. Those functions do not need to receive the ID or add it
to each logger call.

`observe_operation("report.generate")` measures one step:

1. On entry, it starts a span. A span records the selected step in a trace.
2. Your code runs. Logs inside that step include its trace and span IDs.
3. On exit, it writes an `operation.completed` log with duration and outcome.
   It also records count and duration metrics.
4. If an exception leaves the block, it records the error and passes the exception
   back to your code.

Nested operations share a trace. Each operation has its own span.
Wrapping a whole request measures the whole request. Wrap individual steps when
you also need their timings. The library does not automatically time every function.
Your logger calls supply facts such as row counts, provider names, and report IDs.

## Shared workflow: follow one request through three files

A request enters `app.py`. It calls `reports.py`, which calls `storage.py`.
Report generation then fails. `app.py` catches the error and records the result.

The report and storage code is the same for a background job and an HTTP request.
Choose one of the two short entry points below. Each starts a request context; the
logs in all three files then receive the same request ID.

The functions below simulate a database, a provider, and report work.
The observability calls use this library's real API. Save `storage.py` and
`reports.py` in your application, then add one of the entry points below.

### `storage.py`: record facts about the data

```python
import logging

logger = logging.getLogger(__name__)


def load_rows():
    rows = [{"amount": 40}, {"amount": 60}]  # Simulated database result.
    logger.info(
        "Report rows loaded",
        extra={"event_name": "report.rows_loaded", "row_count": len(rows)},
    )
    return rows


def save_report(report):
    report_id = "report-789"  # Simulated storage result.
    logger.info(
        "Report saved",
        extra={"event_name": "report.saved", "report_id": report_id},
    )
    return report_id
```

Neither function receives a request ID. Both use ordinary Python logging.

### `reports.py`: measure the steps

```python
import logging
from observability import observe_operation
from storage import load_rows, save_report

logger = logging.getLogger(__name__)


def create_report():
    with observe_operation("report.load"):
        rows = load_rows()

    with observe_operation("report.generate"):
        logger.info(
            "Report provider call started",
            extra={"event_name": "report.provider_started", "provider": "demo"},
        )
        # Simulated provider failure. Replace with your report-generation call.
        raise TimeoutError("simulated provider timeout")

    with observe_operation("report.save"):
        return save_report(rows)
```

The load operation includes the call into `storage.py`.
Its log receives the active load span ID.
The generation error stops execution before the save step.

### Option 1: run it as a job or script

```python
import logging
from observability import (
    Observability, ObservabilitySettings,
    bind_observability_context, observe_operation,
)
from reports import create_report

logger = logging.getLogger(__name__)
with open("requests.jsonl", "w", encoding="utf-8") as logs:
    runtime = Observability(ObservabilitySettings(service_name="reports"), stream=logs)
    runtime.start()
    try:
        with bind_observability_context(request_id="req-123"):
            logger.info("Request received", extra={"event_name": "request.received"})
            try:
                with observe_operation("request.process"):
                    create_report()
                status = "ok"
            except TimeoutError:
                status = "retry_required"
                logger.warning(
                    "Report retry required",
                    extra={"event_name": "report.retry_required"},
                )
            logger.info(
                "Request finished",
                extra={"event_name": "request.finished", "status": status},
            )
    finally:
        runtime.close()
```

`request.process` measures the report workflow across both called files.
The request context stays active during error handling. This entry point saves
logs to `requests.jsonl`. The example records a retry requirement but does not
perform a retry. Running it again replaces the file.

### Query the job

```sh
python -m observability logs --file requests.jsonl --request-id req-123 --pretty
```

The command returns eight records. Below is sample output with selected fields.
IDs are shortened for readability. Actual trace IDs have 32 hex characters;
span IDs have 16. Durations vary on each run.

All records have `request_id: req-123`. The timed workflow has one trace.
The outer operation and its two executed steps each have a different span:

```text
request req-123
│
├─ request.received                     app.py       no active span
│
├─ request.process                     trace aaaa / span 1111
│  ├─ report.load                      trace aaaa / span 2222
│  │  ├─ report.rows_loaded            storage.py    row_count=2
│  │  └─ operation.completed                        ok, 0.3 ms
│  │
│  ├─ report.generate                  trace aaaa / span 3333
│  │  ├─ report.provider_started       reports.py    provider=demo
│  │  └─ operation.completed                        error, TimeoutError, 0.2 ms
│  │
│  └─ operation.completed                           error, TimeoutError, 0.8 ms
│
├─ report.retry_required               app.py       no active span
└─ request.finished                    app.py       no active span
```

This diagram shows the nesting created by the code. The JSON log query returns
records in source order. It does not build this tree or include parent-span IDs.
Exported traces retain the parent-child relationships.

Here are those same eight records. Other fields, including timestamps, service
information, and error code locations, are omitted here.

```json
[
  {
    "event_name": "request.received",
    "logger": "__main__",
    "request_id": "req-123",
    "message": "Request received"
  },
  {
    "event_name": "report.rows_loaded",
    "logger": "storage",
    "request_id": "req-123",
    "trace_id": "aaaa",
    "span_id": "2222",
    "message": "Report rows loaded",
    "row_count": 2
  },
  {
    "event_name": "operation.completed",
    "request_id": "req-123",
    "trace_id": "aaaa",
    "span_id": "2222",
    "operation": "report.load",
    "outcome": "ok",
    "duration_ms": 0.3
  },
  {
    "event_name": "report.provider_started",
    "logger": "reports",
    "request_id": "req-123",
    "trace_id": "aaaa",
    "span_id": "3333",
    "message": "Report provider call started",
    "provider": "demo"
  },
  {
    "event_name": "operation.completed",
    "request_id": "req-123",
    "trace_id": "aaaa",
    "span_id": "3333",
    "operation": "report.generate",
    "outcome": "error",
    "error_type": "TimeoutError",
    "duration_ms": 0.2
  },
  {
    "event_name": "operation.completed",
    "request_id": "req-123",
    "trace_id": "aaaa",
    "span_id": "1111",
    "operation": "request.process",
    "outcome": "error",
    "error_type": "TimeoutError",
    "duration_ms": 0.8
  },
  {
    "event_name": "report.retry_required",
    "logger": "__main__",
    "request_id": "req-123",
    "severity": "WARNING",
    "message": "Report retry required"
  },
  {
    "event_name": "request.finished",
    "logger": "__main__",
    "request_id": "req-123",
    "message": "Request finished",
    "status": "retry_required"
  }
]
```

Compare the IDs:

- `request_id` connects all eight records, across all three files.
- `trace_id` connects the records inside the timed workflow.
- `span_id: 2222` connects the storage log to the load completion.
- `span_id: 3333` connects the provider log to the generation failure.
- `span_id: 1111` identifies the outer workflow completion.

The entry and error-handling records keep the request ID after or before the spans.
Use the request ID to retrieve the complete sequence.

You can now see the work before the failure and the handling after it.
There is no save record because the failed run did not reach that step.
The records locate the failure; they do not establish why the provider timed out.

To see only failed operation records:

```sh
python -m observability logs --file requests.jsonl --request-id req-123 --outcome error --pretty
```

To see operation duration summaries:

```sh
python -m observability summary --file requests.jsonl --event-name operation.completed --pretty
```

The outer duration includes the inner steps. Do not add those durations together.

The request context follows ordinary calls and `await` in the same process.
New async tasks normally inherit it. Separate processes, remote services, and
queued jobs require explicit ID propagation. See [usage](docs/usage.md#runtime-and-correlation).

### Option 2: use it in a FastAPI web app

Use the same `storage.py` and `reports.py`. Middleware creates the request ID,
binds it to the request, and measures the full HTTP response. Your handler does
not call `bind_observability_context()` or wrap the whole request in
`observe_operation()`.

Install the optional integration and the web server:

```sh
python -m pip install -e '/path/to/observability[asgi]' fastapi uvicorn
```

Save this as `app.py` in the same directory as `reports.py` and `storage.py`:

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from observability import Observability
from observability.integrations.asgi import HTTPObservabilityMiddleware
from reports import create_report

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    runtime = Observability()
    runtime.start()
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(HTTPObservabilityMiddleware)


@app.post("/reports")
async def report_endpoint():
    logger.info("Request received", extra={"event_name": "request.received"})
    try:
        report_id = create_report()
        response = {"report_id": report_id}
    except TimeoutError:
        logger.warning(
            "Report retry required", extra={"event_name": "report.retry_required"}
        )
        response = JSONResponse(status_code=503, content={"status": "retry_required"})
    logger.info("Request finished", extra={"event_name": "request.finished"})
    return response
```

Start the server and save its logs:

```sh
uvicorn app:app > requests.jsonl 2>&1
```

In another terminal, send a request:

```sh
curl -i -X POST http://127.0.0.1:8000/reports
```

The response includes a generated ID:

```http
HTTP/1.1 503 Service Unavailable
x-request-id: 7e090e95-ff33-46ba-b58c-ef904b0bcb87

{"status":"retry_required"}
```

Copy the actual `x-request-id` value from your response and query all logs for
that request:

```sh
python -m observability logs --file requests.jsonl \
  --request-id 7e090e95-ff33-46ba-b58c-ef904b0bcb87 --pretty
```

The HTTP request produces eight records and has this span structure:

```text
HTTP POST /reports                 request ID + HTTP span
├─ request.received                app.py; HTTP span
├─ report.load                     child span
│  ├─ report.rows_loaded           storage.py; load span
│  └─ operation.completed          ok + load duration
├─ report.generate                 child span
│  ├─ report.provider_started      reports.py; generation span
│  └─ operation.completed          error + generation duration
├─ report.retry_required           app.py; HTTP span
├─ request.finished                app.py; HTTP span
└─ operation.completed             http.request; status 503 + total duration
```

The request ID connects all eight records. The trace ID connects logs inside the
HTTP span and its child operations. Each timed step has its own span ID.
The final `http.request` record contains the status code and full response duration.
The timeout is caught by the handler, so the response is 503 and middleware reports
an error outcome. The generation completion keeps the error type and code locations.

Filter by the request ID to see work from every file in one result. Filter by
`--outcome error` to see failed operation completions. The job example uses a
request ID that your code supplies; the HTTP middleware generates one and returns
it in the response header.

The example uses synchronous simulated report functions. Use async calls or worker
threads for blocking work in a real async application. Middleware creates a fresh
request ID and trace; it does not accept inbound request IDs or trace parents.
Separate services and queued jobs need an explicit ID propagation policy.
See [HTTP usage](docs/usage.md#http-applications).

## Run locally or send data to a backend

### Local development

With no OTLP endpoint set, the library writes JSON logs to stdout. Spans and
metrics are measured in the running process but are not saved. Redirect stdout
to a file to keep and query logs:

```sh
python app.py
python -m observability logs --file requests.jsonl --request-id req-123 --pretty
```

This works without a collector or storage service. Your application or container
platform must keep the file if you need it after the process exits.

### Deployed application

To keep and query logs, traces, and metrics after an application exits, send them
to an OTLP collector that writes to your chosen telemetry backend. Deploy the
collector and backend separately. The app must be able to reach the collector.
For example, an app and collector on the same machine may use
`http://localhost:4318`. In Docker or Kubernetes, use the collector's service name
and port, such as `http://otel-collector:4318`; `localhost` inside a container
means that container. This library sends to `/v1/logs`, `/v1/traces`, and
`/v1/metrics` under the collector's **HTTP base URL**.

The [`.env.example`](.env.example) file lists every setting with safe local defaults.
Copy it into your application as `.env` and edit it for your environment. For local
development, load it into the process before starting the app, for example:

```sh
set -a
. ./.env
set +a
python app.py
```

With a blank endpoint, the app writes logs to stdout and does not export to a
collector. Set the endpoint to your collector's HTTP base URL to enable export.
For an authenticated collector, set its required headers and store any token as
a secret in your deployment platform. The endpoint must be an HTTP(S) base URL
without credentials, query, or fragment. The library reads process environment
variables; it does not load `.env` files itself. Use your shell, container, or
deployment platform to load these values. Do not commit real secrets in `.env`.
See [all settings](docs/usage.md#configuration).

The library exports data; it does not host the collector, backend, dashboards, or
query service. Once records are stored there, use that backend's query tools to
inspect traces and metrics. The `observability` CLI reads saved JSON logs; it does
not query OTLP backends.

Keep credentials and user or model payloads out of log messages. Use fixed messages
and safe named fields. Put request and run IDs in fields, not operation names.

## Further reading

- [Usage](docs/usage.md): lifecycle, HTTP requests, settings, and query limits.
- [Architecture](docs/architecture.md): code locations and known gaps.
- [Development](docs/development.md): tests, builds, and contributor setup.
- [Contributor map](AGENTS.md): instructions for changes.
- [Execution plans](docs/exec-plans/README.md): records of substantial work.
