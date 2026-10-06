import asyncio

import httpx
import httpx2
import pytest

from agent_core.agent._execution.recovery import ModelRequestRetryPolicy
from agent_core.models.runtime.runtime import ModelRuntime


@pytest.mark.parametrize('module', [httpx, httpx2])
@pytest.mark.parametrize('error', ['ReadError', 'ReadTimeout', 'RemoteProtocolError'])
def test_stream_recovery_handles_both_http_client_families(module, error):
    failure = getattr(module, error)('interrupted stream')
    wrapped = RuntimeError('provider wrapper')
    wrapped.__cause__ = failure
    policy = ModelRequestRetryPolicy()
    assert policy.is_retryable_error(failure)
    assert policy.is_retryable_error(wrapped)
    assert not policy.is_retryable_error(module.ConnectTimeout('connection'))


def test_locked_anthropic_sdk_accepts_managed_transport(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    async def run():
        try:
            model = ModelRuntime.make_model('claude-sonnet-4-6')
            assert model is not None
        finally:
            await ModelRuntime.close_default()
    asyncio.run(run())


def test_response_guards_preserve_environment_proxy_routing(monkeypatch):
    from agent_core.models.runtime.circuit import _OriginCircuit
    from agent_core.models.runtime.settings import ModelRuntimeSettings
    from agent_core.models.runtime.transport import ModelTransportFactory, _ResponseInactivityTransport

    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.test:3128')
    monkeypatch.setenv('NO_PROXY', 'direct.test')

    async def check():
        settings = ModelRuntimeSettings()
        for limits in (settings.openai, settings.google):
            async with ModelTransportFactory.build_http_client(settings, limits,
                    _OriginCircuit(failure_threshold=5, cooldown_seconds=15)) as client:
                direct = client._transport_for_url(httpx2.URL('https://direct.test'))
                proxied = client._transport_for_url(httpx2.URL('https://provider.test'))
                assert direct is client._transport
                assert proxied is not direct
                assert isinstance(direct, _ResponseInactivityTransport)
                assert isinstance(proxied, _ResponseInactivityTransport)
                assert client.timeout.read == 30
    asyncio.run(check())
