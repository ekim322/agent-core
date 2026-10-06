"""Retry, circuit and resource ownership independent of real providers."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from agent_core.agent._execution.recovery import InterruptedResponse, ModelRequestRetryPolicy
from agent_core.models.runtime.retry import ProviderRequestRetryPolicy
from agent_core.models.runtime.circuit import _ConnectionCircuitScope, _ConnectionCircuitTrace, _OriginCircuit
from agent_core.models.runtime.errors import ModelProviderUnavailable, ModelTransportUnavailable
from agent_core.models.runtime.providers import ModelProviderFactory
from agent_core.models.runtime.routes import ModelRoute
from agent_core.models.runtime.runtime import ModelRuntime
from agent_core.models.runtime.settings import ModelProviderSettings, ModelRuntimeSettings
from agent_core.models.runtime.transport import ModelTransportFactory


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "1"])
def test_configuration_rejects_invalid_budget_types_and_values(value):
    with pytest.raises(ValueError):
        ModelRuntimeSettings(read_timeout_seconds=value)
    with pytest.raises(ValueError):
        ProviderRequestRetryPolicy(max_delay_seconds=value)
    with pytest.raises(ValueError):
        ModelProviderSettings(max_in_flight=1, queue_timeout_seconds=value)


def test_capacity_counts_cannot_be_boolean_or_fractional(monkeypatch):
    with pytest.raises(ValueError):
        ModelRuntimeSettings(status_max_retries=1.5)
    with pytest.raises(ValueError):
        ModelProviderSettings(max_in_flight=True)
    monkeypatch.setenv("LLM_MODEL_READ_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ValueError, match="LLM_MODEL_READ_TIMEOUT_SECONDS"):
        ModelRuntimeSettings.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.test",
        "https://user:password@example.test",
        "https://example.test?q=1",
    ],
)
def test_route_endpoints_are_http_and_credential_free(url):
    with pytest.raises(ValueError):
        ModelRoute("openai", base_url=url)


def test_retry_headers_ignore_nonfinite_values_and_respect_wait_budget():
    cause = RuntimeError()
    cause.response = SimpleNamespace(
        headers={"retry-after-ms": "nan", "retry-after": "4"}
    )
    error = ModelHTTPError(429, "test")
    error.__cause__ = cause
    policy = ProviderRequestRetryPolicy(max_retry_after_seconds=5)
    assert policy.retry_after_seconds(error) == 4
    assert policy.can_retry(error, 0)
    assert 4 <= policy.delay_seconds(error, 1) <= 5
    cause.response.headers["retry-after"] = "10"
    assert not policy.can_retry(error, 0)
    with pytest.raises(ValueError):
        policy.delay_seconds(error, 1)
    cause.response.headers["retry-after"] = "inf"
    assert policy.retry_after_seconds(error) is None
    assert 0 <= policy.delay_seconds(error, 100000) <= 8


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectTimeout("setup"),
        httpx2.WriteTimeout("upload"),
        httpx.PoolTimeout("pool"),
        asyncio.CancelledError(),
    ],
)
def test_recovery_policies_do_not_own_cancellation_or_setup_write_pool_failures(error):
    error.__cause__ = httpx2.ReadTimeout("misleading cause")
    assert not ProviderRequestRetryPolicy().can_retry(error, 0)
    assert not ModelRequestRetryPolicy().is_retryable_error(error)


def test_continuation_preserves_completed_tool_history_without_duplicate_text():
    request = ModelRequest(
        parts=[UserPromptPart("question")], run_id="run", conversation_id="chat"
    )
    partial = ModelResponse(
        parts=[TextPart("partial")],
        run_id="run",
        conversation_id="chat",
        state="interrupted",
    )
    history = [request, partial]
    original_inputs = {"user_prompt": "question", "deps": object()}
    error = InterruptedResponse(httpx.ReadTimeout("read"), tuple(history), "partial")
    plan = ModelRequestRetryPolicy(base_delay_seconds=0).plan_retry(
        error, 1, original_inputs
    )
    assert len(history) == 2 and len(plan.iter_kwargs["message_history"]) == 3
    assert plan.iter_kwargs["message_history"][1] is partial
    assert plan.iter_kwargs["message_history"][-1].run_id == "run"
    assert original_inputs["user_prompt"] == "question"
    assert plan.prefix_delta == "partial"


def test_circuit_recovery_has_one_probe_and_ignores_stale_success(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "agent_core.models.runtime.circuit.time.monotonic", lambda: clock[0]
    )
    circuit = _OriginCircuit(failure_threshold=1, cooldown_seconds=5)
    old, failed = circuit.acquire(), circuit.acquire()
    circuit.record_failure(failed)
    circuit.record_success(old)
    assert circuit.is_open
    with pytest.raises(ModelTransportUnavailable):
        circuit.acquire()
    clock[0] = 15
    probe = circuit.acquire()
    with pytest.raises(ModelTransportUnavailable):
        circuit.acquire()
    circuit.abort(probe)
    replacement = circuit.acquire()
    circuit.record_success(probe)
    assert circuit.is_open, "an aborted probe cannot resolve its replacement"
    circuit.record_success(replacement)
    assert not circuit.is_open


def test_connection_scope_never_replaces_cancellation_with_remembered_failure():
    scope = _ConnectionCircuitScope(
        _OriginCircuit(failure_threshold=1, cooldown_seconds=0)
    )
    scope.transport_unavailable = ModelTransportUnavailable(
        "blocked", retry_after_seconds=0
    )
    with pytest.raises(asyncio.CancelledError):
        with scope.bind():
            raise asyncio.CancelledError()


def test_tls_probe_ignores_proxy_tls_and_waits_for_target_tls():
    circuit = _OriginCircuit(failure_threshold=1, cooldown_seconds=0)
    circuit.record_failure(circuit.acquire())
    observed = _ConnectionCircuitTrace(
        circuit=circuit,
        target_scheme="https",
        target_host="target.test",
    )

    async def check():
        observed.observe("connection.connect_tcp.started", {})
        observed.observe("connection.connect_tcp.complete", {})
        observed.observe(
            "connection.start_tls.started", {"server_hostname": b"proxy.test"}
        )
        observed.observe("connection.start_tls.complete", {})
        assert circuit.is_open
        observed.observe("proxy.start_tls.started", {"server_hostname": b"target.test"})
        observed.observe("proxy.start_tls.complete", {})
        assert not circuit.is_open

    asyncio.run(check())


def _fake_runtime(monkeypatch, model):
    monkeypatch.setattr(
        ModelTransportFactory,
        "build_http_client",
        staticmethod(lambda *args: AsyncMock()),
    )
    monkeypatch.setattr(
        ModelProviderFactory, "build_provider", staticmethod(lambda *args: object())
    )
    monkeypatch.setattr(
        ModelProviderFactory, "build_base_model", staticmethod(lambda *args: model)
    )
    capacity = ModelProviderSettings(
        1,
        max_queued=1,
        max_connections=1,
        max_keepalive_connections=1,
        http2_streams_per_connection=1,
        http2_only=True,
    )
    return ModelRuntime(
        settings=ModelRuntimeSettings(openai=capacity, retry_base_delay_seconds=0)
    )


def test_open_stream_read_failure_is_not_retried_and_slot_is_returned(monkeypatch):
    class Model(TestModel):
        @asynccontextmanager
        async def request_stream(self, *args):
            calls.append("opened")
            try:
                yield object()
            finally:
                calls.append("closed")

    calls = []

    async def check():
        async with _fake_runtime(monkeypatch, Model()) as runtime:
            model = ModelRuntime.make_model("openai:test", runtime=runtime)
            with pytest.raises(httpx2.ReadTimeout):
                async with model.request_stream([], None, ModelRequestParameters()):
                    raise httpx2.ReadTimeout("after exposure")
            assert model._limiter.running_count == 0
            assert model._shard_slots.qsize() == 1

    asyncio.run(check())
    assert calls == ["opened", "closed"]


def test_cancelled_admission_waiter_never_consumes_another_call_slot(monkeypatch):
    entered, release = None, None

    class Model(TestModel):
        async def request(self, *args):
            entered.set()
            await release.wait()
            return ModelResponse(parts=[TextPart("ok")])

    async def check():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        async with _fake_runtime(monkeypatch, Model()) as runtime:
            model = ModelRuntime.make_model("openai:test", runtime=runtime)
            first = asyncio.create_task(
                model.request([], None, ModelRequestParameters())
            )
            await entered.wait()
            second = asyncio.create_task(
                model.request([], None, ModelRequestParameters())
            )
            await asyncio.sleep(0)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert model._limiter.running_count == 1
            release.set()
            await first
            assert model._limiter.running_count == 0 and model._shard_slots.qsize() == 1

    asyncio.run(check())


def test_shutdown_waiter_cancellation_does_not_abandon_closes_or_hide_sibling_errors(
    monkeypatch,
):
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        closed = []

        async def close_ok():
            entered.set()
            await release.wait()
            closed.append("ok")

        async def close_failed():
            closed.append("failed")
            raise ValueError("close failure")

        clients = [
            SimpleNamespace(aclose=close_failed),
            SimpleNamespace(aclose=close_ok),
        ]
        monkeypatch.setattr(
            ModelTransportFactory,
            "build_http_client",
            staticmethod(lambda *args: clients.pop(0)),
        )
        monkeypatch.setattr(
            ModelProviderFactory, "build_provider", staticmethod(lambda *args: object())
        )
        runtime = ModelRuntime(settings=ModelRuntimeSettings())
        runtime._resources_for(ModelRoute("openai"))
        waiter = asyncio.create_task(runtime.aclose())
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert "ok" not in closed
        release.set()
        with pytest.raises(ExceptionGroup, match="shutdown"):
            await runtime.aclose()
        assert closed == ["failed", "ok"]
        with pytest.raises(RuntimeError, match="closed"):
            runtime._resources_for(ModelRoute("openai"))

    asyncio.run(check())


@pytest.mark.parametrize("visible_text", ["", "partial"])
def test_recovery_inputs_are_independent_and_preserve_tool_history(visible_text):
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    tool_call = ModelResponse(parts=[ToolCallPart("lookup", {}, "call-1")])
    tool_result = ModelRequest(
        parts=[ToolReturnPart("lookup", "result", "call-1")],
        run_id="run", conversation_id="chat",
    )
    history = [tool_call, tool_result]
    snapshot = InterruptedResponse(httpx.ReadTimeout("read"), tuple(history), visible_text)
    history.clear()
    policy = ModelRequestRetryPolicy(base_delay_seconds=0)
    first = policy.plan_retry(snapshot, 1, {"deps": "request-deps"})
    second = policy.plan_retry(snapshot, 2, {})
    messages = first.iter_kwargs["message_history"]
    assert messages is not second.iter_kwargs["message_history"]
    assert messages[:2] == [tool_call, tool_result]
    assert messages[0] is tool_call and messages[1] is tool_result
    assert len(messages) == (4 if visible_text else 2)
    if visible_text:
        assert messages[2].parts[0].content == visible_text
        assert messages[2].state == "interrupted"
        assert messages[3].run_id == "run"
        assert messages[3].conversation_id == "chat"
    messages.clear()
    assert len(snapshot.messages) == 2
    assert len(second.iter_kwargs["message_history"]) == (4 if visible_text else 2)
