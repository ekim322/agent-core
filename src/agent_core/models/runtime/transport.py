"""Per-response inactivity deadlines, independent of multiplexed socket traffic."""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal

import httpx2 as httpx

from agent_core.models.runtime.circuit import (
    _ConnectionCircuitRequestHook,
    _OriginCircuit,
)
from agent_core.models.runtime.settings import (
    ModelProviderSettings,
    ModelRuntimeSettings,
)


class _ResponseReadDeadline:
    """Own header admission to the read timer and fresh deadlines for body reads."""

    def __init__(self, request: httpx.Request):
        self.request = request
        self.seconds = request.extensions.get("timeout", {}).get("read")

    @asynccontextmanager
    async def waiting(self, phase: Literal["headers", "body"]):
        deferred = phase == "headers"
        timer = asyncio.timeout(None if deferred else self.seconds)
        try:
            async with timer:
                if deferred:
                    with _HeaderReadTrace(self.request, timer, self.seconds).installed():
                        yield
                else:
                    yield
        except TimeoutError as error:
            if not timer.expired():
                raise
            raise httpx.ReadTimeout(
                f"Provider response {phase} became inactive", request=self.request
            ) from error


class _HeaderReadTrace:
    """Start the response deadline once HTTPcore begins receiving target headers."""

    _START_EVENTS = frozenset({
        "http11.receive_response_headers.started",
        "http2.receive_response_headers.started",
    })

    def __init__(
        self, request: httpx.Request, timer: asyncio.Timeout, seconds: float | None
    ):
        self._extensions = request.extensions
        self._timer = timer
        self._seconds = seconds
        self._previous = self._extensions.get("trace")
        self._was_present = "trace" in self._extensions
        self._started = False

    async def __call__(self, event: str, details: dict[str, Any]) -> None:
        if event in self._START_EVENTS and not self._started:
            target_request = details.get("request")
            if getattr(target_request, "method", None) != b"CONNECT":
                self._started = True
                if self._seconds is not None:
                    expires = asyncio.get_running_loop().time() + self._seconds
                    self._timer.reschedule(expires)
        if self._previous is not None:
            await self._previous(event, details)

    @contextmanager
    def installed(self):
        self._extensions["trace"] = self
        try:
            yield
        finally:
            if self._was_present:
                self._extensions["trace"] = self._previous
            else:
                self._extensions.pop("trace", None)


class _ResponseInactivityStream(httpx.AsyncByteStream):
    def __init__(self, source: httpx.AsyncByteStream, deadline: _ResponseReadDeadline):
        self._source = source
        self._deadline = deadline

    async def __aiter__(self):
        iterator = self._source.__aiter__()
        while True:
            async with self._deadline.waiting("body"):
                chunk = await anext(iterator, None)
            if chunk is None:
                return
            yield chunk

    async def aclose(self) -> None:
        await self._source.aclose()


class _ResponseInactivityTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport):
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        budget = _ResponseReadDeadline(request)
        async with budget.waiting("headers"):
            response = await self._transport.handle_async_request(request)
        response.stream = _ResponseInactivityStream(response.stream, budget)
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()


class _ResponseInactivityClient(httpx.AsyncClient):
    """Wrap both direct and proxy transports using the pinned HTTPX2 hooks."""

    def _init_transport(self, *args: Any, **kwargs: Any):
        transport = super()._init_transport(*args, **kwargs)
        return _ResponseInactivityTransport(transport)

    def _init_proxy_transport(self, *args: Any, **kwargs: Any):
        transport = super()._init_proxy_transport(*args, **kwargs)
        return _ResponseInactivityTransport(transport)


class _ComponentTimeoutAsyncClient(_ResponseInactivityClient):
    """Restore component timeouts when the Google SDK echoes our read timeout."""

    def _restore_component_timeouts(self, supplied: dict) -> dict:
        value = supplied.get("timeout")
        if type(value) in {int, float} and self.timeout.read is not None:
            if math.isclose(value, self.timeout.read, rel_tol=0, abs_tol=0.001):
                return {**supplied, "timeout": self.timeout}
        return supplied

    def build_request(self, *args, **kwargs):
        return super().build_request(*args, **self._restore_component_timeouts(kwargs))

    async def request(self, *args, **kwargs):
        return await super().request(*args, **self._restore_component_timeouts(kwargs))


class ModelTransportFactory:
    @staticmethod
    def build_http_client(
        runtime_settings: ModelRuntimeSettings,
        provider_settings: ModelProviderSettings,
        circuit: _OriginCircuit,
    ) -> httpx.AsyncClient:
        pool = (
            (1, 1)
            if provider_settings.http2_only
            else (
                provider_settings.max_connections,
                provider_settings.max_keepalive_connections,
            )
        )
        limits = httpx.Limits(
            max_connections=pool[0],
            max_keepalive_connections=pool[1],
            keepalive_expiry=runtime_settings.keepalive_expiry_seconds,
        )
        timeout = httpx.Timeout(
            **{
                component: getattr(runtime_settings, component + "_timeout_seconds")
                for component in ("connect", "read", "write", "pool")
            }
        )
        client_class = (
            _ResponseInactivityClient
            if provider_settings.http2_only
            else _ComponentTimeoutAsyncClient
        )
        return client_class(
            http1=not provider_settings.http2_only,
            http2=True,
            timeout=timeout,
            limits=limits,
            event_hooks={"request": [_ConnectionCircuitRequestHook(circuit)]},
        )
