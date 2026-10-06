"""The application-owned shared runtime captures agent instrumentation."""

import asyncio
import io
import json

import observability
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai.models.test import TestModel

from agent_core import BaseAgent
from observability import bind_observability_context
from observability.correlation import current_observability_context


def test_shared_application_runtime_captures_agent_execution():
    output, spans = io.StringIO(), InMemorySpanExporter()
    runtime = observability.Observability(
        observability.ObservabilitySettings(service_name="consumer"), stream=output,
        span_processors=[SimpleSpanProcessor(spans)],
    )
    runtime.start()
    try:
        with bind_observability_context(request_id="request-1"):
            result = asyncio.run(BaseAgent(model=TestModel(custom_output_text="Done")).run(
                "Start", persist=False,
            ))
        assert result.output == "Done"
    finally:
        runtime.close()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    completion = next(row for row in records if row.get("operation") == "agent.run")
    assert completion["service_name"] == "consumer"
    assert completion["request_id"] == "request-1"
    assert any(span.context.trace_id == int(completion["trace_id"], 16)
               for span in spans.get_finished_spans())
    assert current_observability_context() == {}
