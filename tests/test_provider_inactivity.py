"""Provider silence retries and active research without wall-clock deadlines."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import httpx2
import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from agent_core.agent.base import BaseAgent
from agent_core.models.runtime.retry import ProviderRequestRetryPolicy
from agent_core.models.runtime.errors import ModelProviderUnavailable
from agent_core.models.runtime.providers import ModelProviderFactory
from agent_core.models.runtime.runtime import ModelRuntime
from agent_core.models.runtime.settings import ModelRuntimeSettings
from agent_core.models.runtime.transport import ModelTransportFactory
from agent_core.models.runtime.circuit import _OriginCircuit


@pytest.mark.parametrize('module', [httpx, httpx2])
def test_pre_response_retry_policy_handles_wrapped_read_timeout_only(module):
    policy = ProviderRequestRetryPolicy()
    for error, retry in [(module.ReadTimeout, True), (module.ConnectTimeout, False),
                         (module.WriteTimeout, False), (module.PoolTimeout, False),
                         (module.ReadError, False)]:
        wrapped = RuntimeError('provider wrapper')
        wrapped.__cause__ = error('silence')
        assert policy.can_retry(wrapped, 0) is retry
        assert not policy.can_retry(wrapped, 2)


@pytest.mark.parametrize('stage', ['before_open', 'after_open'])
@pytest.mark.parametrize('recover', [True, False])
@pytest.mark.parametrize('max_hops', [0, 50], ids=['finalizer', 'ordinary'])
def test_silent_stream_retries_are_bounded_and_release_capacity(monkeypatch, stage, recover, max_hops):
    async def check():
        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            if recover and attempts == 2:
                yield 'Recovered'
                return
            if stage == 'after_open':
                yield 'Partial answer. '
            raise httpx2.ReadTimeout('provider stopped responding')

        monkeypatch.setattr(ModelProviderFactory, 'build_provider', staticmethod(lambda *args: object()))
        monkeypatch.setattr(ModelProviderFactory, 'build_base_model', staticmethod(
            lambda *args: FunctionModel(stream_function=stream)))
        # Keep the actual retry loops; eliminate their jittered backoff in this test.
        monkeypatch.setattr(ProviderRequestRetryPolicy, 'delay_seconds', lambda *args: 0)
        from agent_core.agent._execution.recovery import ModelRequestRetryPolicy
        monkeypatch.setattr(ModelRequestRetryPolicy, '_delay_seconds', lambda *args: 0)
        async with ModelRuntime(settings=ModelRuntimeSettings()) as runtime:
            model = ModelRuntime.make_model('google:test', runtime=runtime, base_url='https://provider.test/v1')
            agent = BaseAgent(model)
            if recover:
                result = await agent.run('Answer', max_hops=max_hops, persist=False)
                assert result.output == ('Partial answer. ' if stage == 'after_open' else '') + 'Recovered'
                assert attempts == 2
            else:
                with pytest.raises(ModelProviderUnavailable if stage == 'before_open' else httpx2.ReadTimeout):
                    await agent.run('Answer', max_hops=max_hops, persist=False)
                assert attempts == 3
            assert model._limiter.running_count == 0
            assert model._limiter.waiting_count == 0
    asyncio.run(check())


def test_transport_read_timeout_is_inactivity_not_total_duration():
    async def check():
        finished = asyncio.Event()
        idle_seconds = 0.15

        async def serve(reader, writer):
            try:
                await reader.readuntil(b'\r\n\r\n')
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n')
                # More than two read-timeout windows of active traffic, then silence.
                for _ in range(12):
                    writer.write(b'x')
                    await writer.drain()
                    await asyncio.sleep(0.03)
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                finished.set()

        server = await asyncio.start_server(serve, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        settings = ModelRuntimeSettings(read_timeout_seconds=idle_seconds)
        client = ModelTransportFactory.build_http_client(settings, settings.google,
            _OriginCircuit(failure_threshold=5, cooldown_seconds=15))
        try:
            async with server, client:
                started = asyncio.get_running_loop().time()
                data = b''
                with pytest.raises(httpx2.ReadTimeout):
                    async with client.stream('GET', f'http://127.0.0.1:{port}') as response:
                        async for chunk in response.aiter_bytes():
                            data += chunk
                assert data == b'x' * 12
                assert asyncio.get_running_loop().time() - started > idle_seconds * 2
                await asyncio.wait_for(finished.wait(), 2)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(check())




@pytest.mark.parametrize('silent_phase', ['headers', 'body'])
def test_http2_other_response_traffic_does_not_hide_silence(silent_phase):
    from h2.config import H2Configuration
    from h2.connection import H2Connection
    from h2.events import RequestReceived

    async def check():
        connections = 0
        active_tasks = set()
        handlers = set()

        async def serve(reader, writer):
            nonlocal connections
            connections += 1
            handlers.add(asyncio.current_task())
            connection = H2Connection(config=H2Configuration(client_side=False, header_encoding='utf-8'))
            connection.initiate_connection()
            writer.write(connection.data_to_send())

            async def active_response(stream_id):
                for _ in range(12):
                    connection.send_data(stream_id, b'x')
                    writer.write(connection.data_to_send())
                    await writer.drain()
                    await asyncio.sleep(0.03)
                connection.end_stream(stream_id)
                writer.write(connection.data_to_send())
                await writer.drain()

            try:
                while data := await reader.read(65536):
                    for event in connection.receive_data(data):
                        if not isinstance(event, RequestReceived):
                            continue
                        active = dict(event.headers)[':path'] == '/active'
                        if active or silent_phase == 'body':
                            connection.send_headers(event.stream_id, [(':status', '200')])
                        if active:
                            task = asyncio.create_task(active_response(event.stream_id))
                            active_tasks.add(task)
                    writer.write(connection.data_to_send())
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(serve, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        settings = ModelRuntimeSettings(read_timeout_seconds=0.15)
        client = ModelTransportFactory.build_http_client(settings, settings.openai,
            _OriginCircuit(failure_threshold=5, cooldown_seconds=15))
        active = None
        try:
            async with server, client:
                active = asyncio.create_task(client.get(f'http://127.0.0.1:{port}/active'))
                with pytest.raises(httpx2.ReadTimeout):
                    await client.get(f'http://127.0.0.1:{port}/silent')
                assert not active.done()
                response = await active
                assert response.http_version == 'HTTP/2'
                assert response.content == b'x' * 12
                assert connections == 1
        finally:
            if active is not None:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            for task in active_tasks | handlers:
                task.cancel()
            await asyncio.gather(*active_tasks, *handlers, return_exceptions=True)
            server.close()
            await server.wait_closed()
    asyncio.run(check())


def test_response_header_timer_preserves_proxy_setup_and_cancellation():
    from agent_core.models.runtime.transport import _ResponseInactivityTransport

    async def check():
        # CONNECT header tracing must not start the model-response deadline.
        async def proxy_setup(request):
            await request.extensions['trace']('http11.receive_response_headers.started',
                {'request': SimpleNamespace(method=b'CONNECT')})
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return httpx2.Response(200, content=b'ok')

        request = httpx2.Request('POST', 'https://provider.test', extensions={'timeout': {'read': 0}})
        transport = _ResponseInactivityTransport(httpx2.MockTransport(proxy_setup))
        response = await transport.handle_async_request(request)
        await response.aclose()
        await transport.aclose()

        entered, closed = asyncio.Event(), asyncio.Event()

        async def blocked_headers(request):
            try:
                await request.extensions['trace']('http2.receive_response_headers.started',
                    {'request': SimpleNamespace(method=b'POST')})
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        request = httpx2.Request('POST', 'https://provider.test', extensions={'timeout': {'read': 30}})
        transport = _ResponseInactivityTransport(httpx2.MockTransport(blocked_headers))
        task = asyncio.create_task(transport.handle_async_request(request))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert 'trace' not in request.extensions
        await transport.aclose()
    asyncio.run(check())


@pytest.mark.parametrize('failure_type', [TimeoutError, ValueError, asyncio.CancelledError])
@pytest.mark.parametrize('prior_trace', ['absent', 'none', 'callback'])
def test_header_deadline_preserves_foreign_failures_and_prior_trace(failure_type, prior_trace):
    from agent_core.models.runtime.transport import _ResponseInactivityTransport

    async def check():
        observed = []

        async def existing_trace(event, details):
            observed.append(event)

        extensions = {'timeout': {'read': 30}}
        if prior_trace != 'absent':
            extensions['trace'] = existing_trace if prior_trace == 'callback' else None
        request = httpx2.Request('GET', 'https://provider.test', extensions=extensions)
        error = failure_type('foreign failure')

        async def fail(request):
            await request.extensions['trace']('http2.receive_response_headers.started', {})
            raise error

        async with _ResponseInactivityTransport(httpx2.MockTransport(fail)) as transport:
            with pytest.raises(failure_type) as caught:
                await transport.handle_async_request(request)
        assert caught.value is error
        if prior_trace == 'absent':
            assert 'trace' not in request.extensions
        else:
            assert request.extensions['trace'] is (existing_trace if prior_trace == 'callback' else None)
        assert observed == (['http2.receive_response_headers.started'] if prior_trace == 'callback' else [])

    asyncio.run(check())


def test_repeated_header_trace_cannot_extend_response_deadline(monkeypatch):
    from agent_core.models.runtime.transport import _ResponseInactivityTransport

    async def check():
        loop = asyncio.get_running_loop()

        async def repeated_headers(request):
            trace = request.extensions['trace']
            await trace('http2.receive_response_headers.started', {})
            # Simulate a duplicate event at a later clock reading without
            # sleeping. The original zero deadline must remain in force.
            with monkeypatch.context() as patch:
                patch.setattr(asyncio, 'get_running_loop',
                    lambda: SimpleNamespace(time=lambda: loop.time() + 60))
                await trace('http2.receive_response_headers.started', {})
            await asyncio.Event().wait()

        request = httpx2.Request('GET', 'https://provider.test', extensions={'timeout': {'read': 0}})
        async with _ResponseInactivityTransport(httpx2.MockTransport(repeated_headers)) as transport:
            task = asyncio.create_task(transport.handle_async_request(request))
            watchdog = loop.call_later(1, task.cancel)
            try:
                with pytest.raises(httpx2.ReadTimeout):
                    await task
                assert 'trace' not in request.extensions
            finally:
                watchdog.cancel()

    asyncio.run(check())


@pytest.mark.parametrize('failure_type', [TimeoutError, ValueError, asyncio.CancelledError])
def test_body_deadline_preserves_foreign_failures(failure_type):
    from agent_core.models.runtime.transport import _ResponseInactivityTransport

    async def check():
        error = failure_type('foreign body failure')

        class FailedBody(httpx2.AsyncByteStream):
            async def __aiter__(self):
                raise error
                yield b''

        async def response(request):
            return httpx2.Response(200, stream=FailedBody())

        request = httpx2.Request('GET', 'https://provider.test', extensions={'timeout': {'read': 30}})
        async with _ResponseInactivityTransport(httpx2.MockTransport(response)) as transport:
            result = await transport.handle_async_request(request)
            try:
                with pytest.raises(failure_type) as caught:
                    await anext(result.stream.__aiter__())
                assert caught.value is error
            finally:
                await result.aclose()

    asyncio.run(check())
