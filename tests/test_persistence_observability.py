"""Saved Python logs identify admission, writer outcomes and queue latency."""

import asyncio
from datetime import datetime, timezone
import io
import json
import logging

import pytest
from observability import (
    Observability,
    ObservabilitySettings,
    bind_observability_context,
    observe_operation,
)
from observability.cli import main
from observability.correlation import current_observability_context
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from agent_core import LLMClient, PersistenceQueueFull
from agent_core.persistence import RunRecords, background
from agent_core.persistence.background import BackgroundPersistence, write_records


@pytest.fixture
def telemetry():
    output = io.StringIO()
    spans = InMemorySpanExporter()
    metrics = InMemoryMetricReader()
    runtime = Observability(
        ObservabilitySettings(service_name="storage-test", environment="test"),
        stream=output,
        span_processors=[SimpleSpanProcessor(spans)],
        metric_readers=[metrics],
    )
    runtime.start()
    try:
        yield output, spans, metrics
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["awaited", "background", "completion"])
@pytest.mark.parametrize("outcome", ["ok", "error", "cancelled"])
def test_writer_outcomes_retain_ids_and_are_queryable(
    telemetry, mode, outcome, tmp_path, capsys
):
    output, exporter, metrics = telemetry
    calls = []
    since = datetime.now(timezone.utc).isoformat()

    class Writer:
        async def write(self, batch):
            calls.append(current_observability_context())
            logging.getLogger("storage.adapter").info(
                "Storage adapter entered",
                extra={"event_name": "storage.adapter.entered"},
            )
            if outcome == "error":
                raise LookupError("private-storage-error")
            if outcome == "cancelled":
                raise asyncio.CancelledError

    async def check():
        writer = Writer()
        owner = BackgroundPersistence(writer)
        try:
            with (
                bind_observability_context(
                    request_id="request", run_id="run", agent_execution_id="execution",
                    queue_wait_ms=999,
                ),
                observe_operation("caller") as caller,
            ):
                batch = RunRecords(session_id="session", message_id="message")
                if mode == "background":
                    await owner.submit(batch)
                    # Successful admission precedes any writer execution.
                    assert calls == []
                    assert not any(
                        s.name == "persistence.write" for s in exporter.get_finished_spans()
                    )
                else:
                    if mode == "completion":
                        client = LLMClient(
                            TestModel(custom_output_text="private-answer"), writer=writer
                        )
                        pending = client.run(
                            [ModelRequest([UserPromptPart("private-prompt")])],
                            session_id="session",
                            message_id="message",
                        )
                    else:
                        pending = write_records(writer, batch)
                    if outcome == "ok":
                        await pending
                    else:
                        error = LookupError if outcome == "error" else asyncio.CancelledError
                        with pytest.raises(error):
                            await pending
                assert current_observability_context()["queue_wait_ms"] == 999
                caller_id = caller.span.get_span_context().span_id
            assert current_observability_context() == {}
            await owner.drain()
            assert current_observability_context() == {}
            return caller_id
        finally:
            await owner.aclose()

    caller_id = asyncio.run(check())
    until = datetime.now(timezone.utc).isoformat()
    assert len(calls) == 1
    assert calls[0]["run_id"] == "run"
    assert calls[0]["session_id"] == "session"
    assert calls[0]["message_id"] == "message"
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    completion, = [
        row for row in records if row.get("operation") == "persistence.write"
    ]
    assert completion["outcome"] == outcome
    assert completion["duration_ms"] >= 0
    assert completion["persistence_mode"] == (
        "background" if mode == "background" else "awaited"
    )
    assert completion["agent_execution_id"] == "execution"
    assert completion["request_id"] == "request"
    write_span, = [
        s for s in exporter.get_finished_spans() if s.name == "persistence.write"
    ]
    assert write_span.parent.span_id == caller_id
    assert write_span.attributes["outcome"] == outcome
    assert completion["trace_id"] == f"{write_span.context.trace_id:032x}"
    assert completion["span_id"] == f"{write_span.context.span_id:016x}"
    adapter_log, = [r for r in records if r["event_name"] == "storage.adapter.entered"]
    for field in ("run_id", "session_id", "message_id", "trace_id", "span_id"):
        assert adapter_log[field] == completion[field]
    if mode == "background":
        enqueue, = [
            row for row in records if row.get("operation") == "persistence.enqueue"
        ]
        assert enqueue["outcome"] == "ok"
        assert completion["queue_wait_ms"] >= 0
        assert write_span.attributes["queue_wait_ms"] == completion["queue_wait_ms"]
        assert enqueue["queue_wait_ms"] is None
    else:
        assert completion["queue_wait_ms"] is None
        assert "queue_wait_ms" not in write_span.attributes
    if outcome == "error":
        assert completion["error_type"] == "LookupError"
        failures = [r for r in records if r["event_name"] == "agent.persistence_write_failed"]
        assert len(failures) == 1
    elif outcome == "cancelled":
        cancellations = [
            r for r in records if r["event_name"] == "agent.persistence_write_cancelled"
        ]
        assert len(cancellations) == 1
    assert "private-" not in output.getvalue()
    for resource in metrics.get_metrics_data().resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                assert all(
                    set(point.attributes) == {"operation", "outcome"}
                    for point in metric.data.data_points
                )

    path = tmp_path / "actual.jsonl"
    path.write_text(output.getvalue(), encoding="utf-8")
    assert main(
        [
            "logs", "--file", str(path), "--service-name", "storage-test",
            "--environment", "test", "--since", since, "--until", until,
            "--request-id", "request", "--field", "run_id=run",
            "--field", "message_id=message", "--field", "session_id=session",
            "--operation", "persistence.write", "--outcome", outcome,
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["matched_records"] == 1 and not result["truncated"]
    assert result["records"] == [completion]


def test_background_admission_rejection_and_queue_wait_are_distinct(telemetry, monkeypatch):
    output, exporter, _ = telemetry
    clock = [10.0]
    monkeypatch.setattr(background, "perf_counter", lambda: clock[0])

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        stored = []

        class Writer:
            async def write(self, batch):
                entered.set()
                await release.wait()
                stored.append(batch.message_id)

        owner = BackgroundPersistence(Writer(), max_concurrency=1, max_queued=1)
        try:
            with bind_observability_context(run_id="first"):
                await owner.submit(RunRecords(message_id="first"))
            await entered.wait()
            clock[0] = 11.0
            with bind_observability_context(run_id="second"):
                await owner.submit(RunRecords(message_id="second"))
            with bind_observability_context(run_id="rejected"):
                with pytest.raises(PersistenceQueueFull) as caught:
                    await owner.submit(RunRecords(message_id="rejected"))
                assert isinstance(caught.value.__cause__, asyncio.QueueFull)
            assert current_observability_context() == {}
            clock[0] = 13.0
        finally:
            release.set()
            await owner.aclose()
        assert stored == ["first", "second"]

    asyncio.run(check())
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    writes = [r for r in records if r.get("operation") == "persistence.write"]
    assert [(r["run_id"], r["queue_wait_ms"]) for r in writes] == [
        ("first", 0), ("second", 2000)
    ]
    assert all(r["outcome"] == "ok" for r in writes)
    admission = [r for r in records if r.get("operation") == "persistence.enqueue"]
    assert [(r["run_id"], r["outcome"]) for r in admission] == [
        ("first", "ok"), ("second", "ok"), ("rejected", "rejected")
    ]
    write_spans = [s for s in exporter.get_finished_spans() if s.name == "persistence.write"]
    assert all(s.parent is None for s in write_spans)
    assert write_spans[0].context.trace_id != write_spans[1].context.trace_id
