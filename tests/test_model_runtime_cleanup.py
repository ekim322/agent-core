"""Route construction failures and environment pool-shape precedence."""

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel

from agent_core.models.runtime.retry import ProviderRequestRetryPolicy
from agent_core.models.runtime import managed as managed_model

from agent_core.models.runtime.providers import ModelProviderFactory
from agent_core.models.runtime.routes import ModelRoute
from agent_core.models.runtime.runtime import ModelRuntime
from agent_core.models.runtime.settings import ModelRuntimeSettings
from agent_core.models.runtime.transport import ModelTransportFactory


@pytest.mark.parametrize("failure_stage", ["client", "provider"])
@pytest.mark.parametrize("failure_type", [ValueError, asyncio.CancelledError])
def test_partial_route_construction_closes_clients_without_publishing(
    monkeypatch, failure_stage, failure_type,
):
    async def check():
        clients = []
        provider_calls = 0
        failure = failure_type("construction interrupted")

        def build_client(*args):
            if failure_stage == "client" and clients:
                raise failure
            client = AsyncMock()
            clients.append(client)
            return client

        def build_provider(*args):
            nonlocal provider_calls
            provider_calls += 1
            if failure_stage == "provider" and provider_calls == 2:
                raise failure
            return object()

        monkeypatch.setattr(ModelTransportFactory, "build_http_client", staticmethod(build_client))
        monkeypatch.setattr(ModelProviderFactory, "build_provider", staticmethod(build_provider))
        runtime = ModelRuntime(settings=ModelRuntimeSettings())
        route = ModelRoute("openai", base_url="https://provider.test/v1")
        try:
            with pytest.raises(failure_type) as caught:
                runtime._resources_for(route)
            assert caught.value is failure
            assert not runtime._resources
        finally:
            await runtime.aclose()
        assert len(clients) == (1 if failure_stage == "client" else 2)
        for client in clients:
            client.aclose.assert_awaited_once()

    asyncio.run(check())


@pytest.mark.parametrize(
    "provider,global_connections,specific_connections,global_keepalive,specific_keepalive,expected",
    [
        ("google", None, None, None, None, (32, 32)),
        ("google", 20, None, 8, None, (32, 32)),
        ("google", 40, 48, 8, None, (48, 48)),
        ("google", 40, 48, 8, 16, (48, 16)),
        ("openai", 4, None, 3, None, (4, 3)),
        ("openai", 4, 6, None, None, (6, 6)),
        ("openai", 4, 6, 3, None, (6, 3)),
        ("openai", 4, 6, 3, 4, (6, 4)),
    ],
)
def test_connection_and_keepalive_environment_precedence(
    monkeypatch, provider, global_connections, specific_connections,
    global_keepalive, specific_keepalive, expected,
):
    for name in tuple(os.environ):
        if name.startswith("LLM_MODEL_"):
            monkeypatch.delenv(name)
    overrides = {
        "LLM_MODEL_MAX_CONNECTIONS": global_connections,
        f"LLM_MODEL_{provider.upper()}_MAX_CONNECTIONS": specific_connections,
        "LLM_MODEL_MAX_KEEPALIVE_CONNECTIONS": global_keepalive,
        f"LLM_MODEL_{provider.upper()}_MAX_KEEPALIVE_CONNECTIONS": specific_keepalive,
    }
    for name, value in overrides.items():
        if value is not None:
            monkeypatch.setenv(name, str(value))
    settings = ModelRuntimeSettings.from_env().for_provider(provider)
    assert (settings.max_connections, settings.max_keepalive_connections) == expected


@pytest.mark.parametrize("operation_name", ["request", "request_stream"])
def test_retry_reporting_matches_for_requests_and_streams(monkeypatch, caplog, operation_name):
    attempts = 0
    retry_counter = Mock()
    meter = SimpleNamespace(
        create_counter=lambda *args, **kwargs: retry_counter,
        create_up_down_counter=lambda *args, **kwargs: Mock(),
    )
    monkeypatch.setattr(managed_model, "get_meter", lambda *args: meter)
    monkeypatch.setattr(ProviderRequestRetryPolicy, "delay_seconds", lambda *args: 0)

    def check_attempt():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx2.ReadTimeout("provider silence")

    async def respond(messages, info):
        check_attempt()
        return ModelResponse(parts=[TextPart("recovered")])

    async def stream(messages, info):
        check_attempt()
        yield "recovered"

    monkeypatch.setattr(ModelTransportFactory, "build_http_client", staticmethod(lambda *args: AsyncMock()))
    monkeypatch.setattr(ModelProviderFactory, "build_provider", staticmethod(lambda *args: object()))
    monkeypatch.setattr(ModelProviderFactory, "build_base_model", staticmethod(
        lambda *args: FunctionModel(respond, stream_function=stream)))

    async def check():
        async with ModelRuntime(settings=ModelRuntimeSettings()) as runtime:
            model = ModelRuntime.make_model("google:test", runtime=runtime, base_url="https://provider.test/v1")
            if operation_name == "request":
                await model.request([], None, ModelRequestParameters())
            else:
                async with model.request_stream([], None, ModelRequestParameters()) as response:
                    async for _ in response:
                        pass
            assert model._limiter.running_count == 0

    with caplog.at_level(logging.WARNING, logger=managed_model.__name__):
        asyncio.run(check())
    assert attempts == 2
    retry_counter.add.assert_called_once_with(1, {"provider": "google", "operation": operation_name})
    records = [record for record in caplog.records if getattr(record, "event_name", None) == "model.retry"]
    assert len(records) == 1
    record = records[0]
    assert (record.retry_attempt, record.retry_limit, record.delay_seconds) == (1, 2, 0)
    assert record.status_code is None
    assert record.error_type == "ReadTimeout"


def test_default_runtime_subclasses_own_separate_instances_and_endpoints():
    class First(ModelRuntime):
        pass

    class Second(ModelRuntime):
        pass

    async def check():
        First.configure_default_endpoints({"openai": "https://first.test/v1"})
        Second.configure_default_endpoints({"openai": "https://second.test/v1"})
        first, second = First.default(), Second.default()
        try:
            assert First.default() is first
            assert Second.default() is second
            assert isinstance(first, First) and isinstance(second, Second)
            assert first._resolve_base_url("openai", None) == "https://first.test/v1"
            assert second._resolve_base_url("openai", None) == "https://second.test/v1"
        finally:
            await First.close_default()
            await Second.close_default()

    asyncio.run(check())


@pytest.mark.parametrize("close_fails", [False, True])
def test_default_retirement_is_shared_and_survives_waiter_cancellation(close_fails):
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        close_calls = 0

        class Runtime(ModelRuntime):
            async def aclose(self):
                nonlocal close_calls
                close_calls += 1
                entered.set()
                await release.wait()
                await super().aclose()
                if close_fails and close_calls == 1:
                    raise ValueError("shutdown failed")

        original = Runtime.default()
        first = asyncio.create_task(Runtime.close_default())
        await entered.wait()
        second = asyncio.create_task(Runtime.close_default())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(RuntimeError, match="closing"):
            Runtime.default()
        with pytest.raises(RuntimeError, match="closing"):
            Runtime.configure_default_endpoints({"openai": "https://changed.test"})
        release.set()
        if close_fails:
            with pytest.raises(ValueError, match="shutdown failed"):
                await second
        else:
            await second
        assert close_calls == 1
        replacement = Runtime.default()
        assert replacement is not original
        await Runtime.close_default()
        assert close_calls == 2

    asyncio.run(check())


def test_explicit_default_close_blocks_replacement_until_cleanup_finishes(monkeypatch):
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        async def close_client():
            entered.set()
            await release.wait()

        monkeypatch.setattr(ModelTransportFactory, "build_http_client", staticmethod(
            lambda *args: SimpleNamespace(aclose=close_client)))
        monkeypatch.setattr(ModelProviderFactory, "build_provider", staticmethod(
            lambda *args: object()))

        class Runtime(ModelRuntime):
            pass

        original = Runtime.default()
        original._resources_for(ModelRoute("openai"))
        closing = asyncio.create_task(original.aclose())
        await entered.wait()
        with pytest.raises(RuntimeError, match="closing"):
            Runtime.default()
        with pytest.raises(RuntimeError, match="closing"):
            Runtime.configure_default_endpoints({"openai": "https://changed.test"})
        release.set()
        await closing
        replacement = Runtime.default()
        assert replacement is not original
        await Runtime.close_default()

    asyncio.run(check())


def test_default_endpoint_update_is_atomic_when_an_open_provider_rejects_it(monkeypatch):
    monkeypatch.setattr(ModelTransportFactory, "build_http_client", staticmethod(
        lambda *args: AsyncMock()))
    monkeypatch.setattr(ModelProviderFactory, "build_provider", staticmethod(
        lambda *args: object()))

    class Runtime(ModelRuntime):
        pass

    async def check():
        Runtime.configure_default_endpoints({"openai": "https://original.test/v1"})
        original = Runtime.default()
        original._resources_for(ModelRoute("openai", base_url="https://original.test/v1"))
        try:
            with pytest.raises(RuntimeError, match="already opened"):
                Runtime.configure_default_endpoints({
                    "openai": "https://changed.test/v1",
                    "google": "https://google.test/v1",
                })
            assert "google" not in original._base_urls
        finally:
            await Runtime.close_default()
        replacement = Runtime.default()
        try:
            assert replacement._resolve_base_url("openai", None) == "https://original.test/v1"
            assert "google" not in replacement._base_urls
        finally:
            await Runtime.close_default()

    asyncio.run(check())
