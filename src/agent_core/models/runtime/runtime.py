"""Lazy route construction and explicit ownership of provider resources."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx2 as httpx
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.models import Model
from pydantic_ai.providers import Provider

from agent_core.models.runtime._default import DefaultRuntime
from agent_core.models.runtime.retry import ProviderRequestRetryPolicy
from agent_core.models.runtime.circuit import _OriginCircuit
from agent_core.models.runtime.managed import ManagedModel
from agent_core.models.runtime.providers import ModelProviderFactory
from agent_core.models.runtime.routes import (
    ConnectionProvider,
    ModelRoute,
    ModelRouting,
)
from agent_core.models.runtime.settings import (
    ModelProviderSettings,
    ModelRuntimeSettings,
)
from agent_core.models.runtime.transport import ModelTransportFactory

_LOG = logging.getLogger("agent_core.utils.model_runtime.runtime")


@dataclass(frozen=True)
class _RouteResources:
    http_clients: tuple[httpx.AsyncClient, ...]
    providers: tuple[Provider[Any], ...]
    limiter: ConcurrencyLimiter
    circuit: _OriginCircuit
    settings: ModelProviderSettings
    shard_slots: asyncio.Queue[int] | None


async def _close_clients(clients) -> None:
    """Wait for all closes before raising any error; no sibling is abandoned."""
    results = await asyncio.gather(
        *(client.aclose() for client in clients), return_exceptions=True
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise BaseExceptionGroup("Model client cleanup failed", failures)


class ModelRuntime:
    """Share provider connections and request capacity across models on a route.

    ``make_model`` constructs route resources on first use and reuses them for
    the same provider, endpoint and Vertex project/region. Each returned model
    shares that route's admission limiter, connection circuit and HTTP clients.
    The runtime owns those clients; model objects do not close them.

    Keep an instance within one asynchronous application lifetime. Construction
    is synchronous; models and cleanup must be used on the application's loop.
    Stop all model work before closing. Shutdown waits for every client and
    retained partial-construction cleanup task, then reports grouped failures.
    Repeated close calls await the same shielded cleanup task, so cancellation
    of a waiter does not abandon shutdown.
    """

    _defaults: ClassVar[DefaultRuntime] = DefaultRuntime()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._defaults = DefaultRuntime()

    def __init__(
        self,
        settings: ModelRuntimeSettings | None = None,
        *,
        base_urls: Mapping[ConnectionProvider, str | None] | None = None,
    ) -> None:
        self.settings = (
            settings if settings is not None else ModelRuntimeSettings.from_env()
        )
        endpoints = type(self)._defaults.endpoints()
        for provider in ("openai", "anthropic"):
            value = os.environ.get(provider.upper() + "_BASE_URL")
            if value:
                endpoints.setdefault(provider, value)
        endpoints.update(base_urls or {})
        self._base_urls = self._normalize_base_urls(endpoints)
        self._resources: dict[ModelRoute, _RouteResources] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._retry_policy = ProviderRequestRetryPolicy(
            self.settings.status_max_retries,
            self.settings.retry_base_delay_seconds,
            self.settings.retry_max_delay_seconds,
            self.settings.retry_after_max_seconds,
        )

    @staticmethod
    def _normalize_base_urls(base_urls) -> dict[ConnectionProvider, str]:
        normalized = {}
        for provider, value in base_urls.items():
            if provider not in ModelRouting.VALID_PROVIDERS:
                raise ValueError(f"Unsupported provider base URL: {provider!r}")
            endpoint = ModelRouting.normalize_base_url(value)
            if endpoint is not None:
                normalized[provider] = endpoint
        return normalized

    @classmethod
    def configure_default_endpoints(
        cls, base_urls: Mapping[ConnectionProvider, str | None]
    ) -> None:
        """Set process-default endpoints before affected provider routes open.

        Changing an endpoint for an already opened provider raises RuntimeError.
        Omitted or blank endpoint entries leave prior defaults intact.
        """
        endpoints = cls._normalize_base_urls(base_urls)
        cls._defaults.configure(endpoints)

    def _update_default_endpoints(
        self, endpoints: Mapping[ConnectionProvider, str]
    ) -> None:
        with self._lock:
            if self._closed and self._close_task is not None:
                if not self._close_task.done():
                    raise RuntimeError(
                        "Default runtime is closing; endpoint changes are unavailable"
                    )
            changed_providers = {
                provider
                for provider, endpoint in endpoints.items()
                if self._base_urls.get(provider) != endpoint
            }
            opened_providers = {route.provider for route in self._resources}
            if changed_providers & opened_providers:
                raise RuntimeError(
                    "Default ModelRuntime already opened routes with different base URLs"
                )
            self._base_urls.update(endpoints)

    @staticmethod
    def infer_provider(model_name: str) -> ConnectionProvider:
        return ModelRouting.infer_provider(model_name)

    @classmethod
    def make_model(
        cls,
        name: str,
        *,
        base_url: str | None = None,
        vertex_project: str | None = None,
        vertex_region: str | None = None,
        runtime: ModelRuntime | None = None,
    ) -> Model:
        """Build a model sharing resources owned by the supplied or default runtime.

        Provider names can be explicit (provider:model) or inferred from a
        recognized model name. Provider construction may require credentials.
        Close the owning runtime only after every model operation has stopped.
        """
        owner = cls.default() if runtime is None else runtime
        provider = ModelRouting.infer_provider(name)
        route = ModelRoute(
            provider,
            vertex_project,
            vertex_region,
            owner._resolve_base_url(provider, base_url),
        )
        model_name = ModelRouting.unprefixed_model_name(name)
        resources = owner._resources_for(route)
        implementations = tuple(
            ModelProviderFactory.build_base_model(model_name, route, adapter)
            for adapter in resources.providers
        )
        return ManagedModel(
            implementations,
            route=route,
            limiter=resources.limiter,
            retry_policy=owner._retry_policy,
            circuit=resources.circuit,
            provider_settings=resources.settings,
            shard_slots=resources.shard_slots,
        )

    def _resolve_base_url(
        self, provider: ConnectionProvider, override: str | None
    ) -> str | None:
        if override is not None:
            return ModelRouting.normalize_base_url(override)
        if provider in self._base_urls:
            return self._base_urls[provider]
        if provider == "openai":
            return ModelRouting.normalize_base_url(
                ModelProviderFactory.openai_base_url_from_env()
            )
        return None

    def _resources_for(self, route: ModelRoute) -> _RouteResources:
        with self._lock:
            if self._closed:
                raise RuntimeError("ModelRuntime is closed")
            if route not in self._resources:
                bundle = self._construct_route(route)
                self._resources[route] = bundle
                _LOG.info(
                    "Provider route initialized",
                    extra={
                        "event_name": "model.capacity_ready",
                        "provider": route.provider,
                        "http2_only": bundle.settings.http2_only,
                        "shard_count": bundle.settings.shard_count,
                        "streams_per_shard": bundle.settings.http2_streams_per_connection,
                        "max_in_flight": bundle.settings.max_in_flight,
                        "max_queued": bundle.settings.max_queued,
                        "queue_timeout_seconds": bundle.settings.queue_timeout_seconds,
                        "connect_timeout_seconds": self.settings.connect_timeout_seconds,
                        "pool_timeout_seconds": self.settings.pool_timeout_seconds,
                    },
                )
            return self._resources[route]

    def _construct_route(self, route: ModelRoute) -> _RouteResources:
        capacity = self.settings.for_provider(route.provider)
        circuit = _OriginCircuit(
            failure_threshold=self.settings.circuit_failure_threshold,
            cooldown_seconds=self.settings.circuit_cooldown_seconds,
        )
        clients = []
        adapters = []
        try:
            for _ in range(capacity.shard_count):
                client = ModelTransportFactory.build_http_client(
                    self.settings, capacity, circuit
                )
                clients.append(client)
                adapters.append(ModelProviderFactory.build_provider(route, client))
            return _RouteResources(
                tuple(clients),
                tuple(adapters),
                ConcurrencyLimiter(
                    max_running=capacity.max_in_flight,
                    max_queued=capacity.max_queued,
                    name="llm-model:" + route.label,
                ),
                circuit,
                capacity,
                self._build_shard_slots(capacity),
            )
        except BaseException:
            self._close_partial_route_clients(clients)
            raise

    def _close_partial_route_clients(self, clients) -> None:
        """Close clients from failed construction; retain async failures for shutdown."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(_close_clients(clients))
            except BaseException as error:
                _LOG.error(
                    "Partial route cleanup failed",
                    extra={
                        "event_name": "model.cleanup_failed",
                        "error_type": type(error).__name__,
                    },
                )
        else:
            # Keep terminal tasks until shutdown so cleanup failures are observed.
            self._cleanup_tasks.add(
                loop.create_task(_close_clients(clients), name="model_partial_cleanup")
            )

    @staticmethod
    def _build_shard_slots(
        settings: ModelProviderSettings,
    ) -> asyncio.Queue[int] | None:
        if not settings.http2_only:
            return None
        slots = asyncio.Queue(maxsize=settings.max_in_flight)
        streams = settings.http2_streams_per_connection
        capacities = [
            min(streams, settings.max_in_flight - index * streams)
            for index in range(settings.shard_count)
        ]
        for position in range(streams):
            for index, capacity in enumerate(capacities):
                if position < capacity:
                    slots.put_nowait(index)
        return slots

    async def _shutdown(self, clients, partial_tasks) -> None:
        results = await asyncio.gather(
            _close_clients(clients), *partial_tasks, return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("Model runtime shutdown failed", errors)

    async def aclose(self) -> None:
        """Prevent new routes and await owned cleanup, shielded from waiter cancellation."""
        with self._lock:
            if self._close_task is None:
                self._closed = True
                clients = [
                    client
                    for route in self._resources.values()
                    for client in route.http_clients
                ]
                tasks = tuple(self._cleanup_tasks)
                self._resources.clear()
                self._cleanup_tasks.clear()
                self._close_task = asyncio.create_task(
                    self._shutdown(clients, tasks), name="model_runtime_shutdown"
                )
            task = self._close_task
        await asyncio.shield(task)

    async def __aenter__(self) -> ModelRuntime:
        if self._closed:
            raise RuntimeError("ModelRuntime is closed")
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _can_reuse_as_default(self) -> bool:
        with self._lock:
            if not self._closed:
                return True
            if self._close_task is not None and not self._close_task.done():
                raise RuntimeError(
                    "Default runtime is closing; a replacement cannot be acquired yet"
                )
            return False

    @classmethod
    def default(cls) -> ModelRuntime:
        """Obtain this runtime class's default, creating a fresh one after closure."""
        return cls._defaults.acquire(cls)

    @classmethod
    async def close_default(cls) -> None:
        """Await retirement of this class's default, even if another waiter cancels."""
        await cls._defaults.close()
