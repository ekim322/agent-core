"""A model facade whose route owns admission, shards and response retries."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from observability import bind_observability_context, get_meter, observe_operation
from pydantic_ai import RunContext
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.exceptions import ConcurrencyLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from agent_core.models.runtime.retry import ProviderRequestRetryPolicy
from agent_core.models.runtime.circuit import (
    TransportErrors,
    _ConnectionCircuitScope,
    _OriginCircuit,
)
from agent_core.models.runtime.errors import (
    ModelProviderUnavailable,
    ModelRouteSaturated,
)
from agent_core.models.runtime.routes import ModelRoute
from agent_core.models.runtime.settings import ModelProviderSettings

_LOG = logging.getLogger("agent_core.utils.model_runtime.managed_model")
T = TypeVar("T")


@dataclass(frozen=True)
class _DispatchPermit:
    model: Model
    shard_index: int | None


class ManagedModel(WrapperModel):
    """Hold route capacity for the complete operation, including stream closure.

    Only opening failures are retried. Exceptions after yielding a stream belong
    to the agent's history-aware recovery policy. Backoff releases admission.
    """

    def __init__(
        self,
        wrapped_models: tuple[Model, ...],
        *,
        route: ModelRoute,
        limiter: ConcurrencyLimiter,
        retry_policy: ProviderRequestRetryPolicy,
        circuit: _OriginCircuit,
        provider_settings: ModelProviderSettings,
        shard_slots: asyncio.Queue[int] | None,
    ) -> None:
        if not wrapped_models or len(wrapped_models) != provider_settings.shard_count:
            raise ValueError("wrapped_models must match the configured shard count")
        super().__init__(wrapped_models[0])
        self._wrapped_models = tuple(wrapped_models)
        self._route = route
        self._limiter = limiter
        self._retry_policy = retry_policy
        self._circuit = circuit
        self._provider_settings = provider_settings
        self._shard_slots = shard_slots

    @property
    def route(self) -> ModelRoute:
        return self._route

    @property
    def base_url(self) -> str | None:
        return self.wrapped.base_url

    def _route_saturated_error(
        self, operation: str, reason: str
    ) -> ModelRouteSaturated:
        labels = dict(provider=self.route.provider, operation=operation, reason=reason)
        get_meter("agent_core.utils.model_runtime.managed_model").create_counter(
            "model.saturation.count", unit="{event}"
        ).add(1, labels)
        _LOG.warning(
            "Provider admission unavailable",
            extra={
                "event_name": "model.saturated",
                "provider": self.route.provider,
                "model_name": self.model_name,
                "model_operation": operation,
                "reason": reason,
                "running_count": self._limiter.running_count,
                "waiting_count": self._limiter.waiting_count,
                "max_in_flight": self._limiter.max_running,
            },
        )
        return ModelRouteSaturated(
            f"Model route {self.route.label} is saturated ({reason})"
        )

    @asynccontextmanager
    async def _admit_operation(self, operation: str):
        """Wait for route admission, then reserve a shard until the caller exits."""
        capacity = get_meter(
            "agent_core.utils.model_runtime.managed_model"
        ).create_up_down_counter("model.capacity", unit="{request}")
        labels = dict(provider=self.route.provider, operation=operation)
        waiting = dict(labels, state="waiting")
        active = dict(labels, state="active")
        capacity.add(1, waiting)
        try:
            with observe_operation("model.queue", success_log_level=logging.DEBUG):
                async with asyncio.timeout(
                    self._provider_settings.queue_timeout_seconds
                ):
                    await self._limiter.acquire(
                        f"model:{self.route.label}:{self.model_name}:{operation}"
                    )
        except ConcurrencyLimitExceeded as error:
            raise self._route_saturated_error(operation, "queue_full") from error
        except TimeoutError as error:
            raise self._route_saturated_error(operation, "queue_timeout") from error
        finally:
            capacity.add(-1, waiting)
        shard = None
        capacity.add(1, active)
        try:
            if self._shard_slots is not None:
                try:
                    shard = self._shard_slots.get_nowait()
                except asyncio.QueueEmpty as error:
                    raise RuntimeError(
                        "route capacity has no corresponding HTTP/2 slot"
                    ) from error
            model = self._wrapped_models[0 if shard is None else shard]
            yield _DispatchPermit(model, shard)
        finally:
            if shard is not None:
                self._shard_slots.put_nowait(shard)
            self._limiter.release()
            capacity.add(-1, active)

    def _provider_unavailable_error(
        self, error: Exception
    ) -> ModelProviderUnavailable | None:
        policy = self._retry_policy
        if not (policy.is_read_timeout(error) or policy.is_retryable_status(error)):
            return None
        status = policy.status_code(error)
        retry_after = policy.retry_after_seconds(error)
        return ModelProviderUnavailable(
            "Model provider exhausted its pre-response retry budget",
            status_code=status,
            retry_after_seconds=(
                policy.max_delay_seconds if retry_after is None else retry_after
            ),
        )

    async def _wait_for_retry(
        self, error: Exception, attempt: int, operation: str
    ) -> None:
        delay = self._retry_policy.delay_seconds(error, attempt)
        get_meter("agent_core.utils.model_runtime.managed_model").create_counter(
            "model.retry.count", unit="{retry}"
        ).add(
            1,
            dict(provider=self.route.provider, operation=operation),
        )
        _LOG.warning(
            "Provider response recovery scheduled",
            extra={
                "event_name": "model.retry",
                "retry_attempt": attempt,
                "retry_limit": self._retry_policy.max_retries,
                "delay_seconds": delay,
                "status_code": self._retry_policy.status_code(error),
                "error_type": type(error).__name__,
            },
        )
        await asyncio.sleep(delay)

    @asynccontextmanager
    async def _dispatch_with_opening_retries(
        self, operation: str, dispatch: Callable[[AsyncExitStack, Model], Awaitable[T]]
    ):
        """Expose a response with capacity held; retry only failures before exposure.

        Each failed opening releases the shard, limiter and circuit scope
        before backoff. Consumer errors and stream-close failures propagate
        after exposure so this layer cannot replay an already observed answer.
        """
        with (
            bind_observability_context(
                provider=self.route.provider, model_name=self.model_name
            ),
            observe_operation("model." + operation),
        ):
            retries_used = 0
            while True:
                response_exposed = False
                try:
                    async with AsyncExitStack() as lifetime:
                        permit = await lifetime.enter_async_context(
                            self._admit_operation(operation)
                        )
                        lifetime.enter_context(
                            _ConnectionCircuitScope(self._circuit).bind()
                        )
                        response = await dispatch(lifetime, permit.model)
                        response_exposed = True
                        yield response
                    return
                except Exception as error:
                    if response_exposed:
                        raise
                    if TransportErrors.is_pool_side_error(error):
                        raise self._route_saturated_error(
                            operation, "connection_pool_timeout"
                        ) from error
                    if not self._retry_policy.can_retry(error, retries_used):
                        unavailable = self._provider_unavailable_error(error)
                        if unavailable is not None:
                            raise unavailable from error
                        raise
                    retries_used += 1
                    await self._wait_for_retry(error, retries_used, operation)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        async def dispatch(lifetime, model):
            return await model.request(
                messages, model_settings, model_request_parameters
            )

        async with self._dispatch_with_opening_retries("request", dispatch) as response:
            return response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        async def dispatch(lifetime, model):
            return await model.count_tokens(
                messages, model_settings, model_request_parameters
            )

        async with self._dispatch_with_opening_retries(
            "count_tokens", dispatch
        ) as usage:
            return usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ):
        async def dispatch(lifetime, model):
            return await lifetime.enter_async_context(
                model.request_stream(
                    messages,
                    model_settings,
                    model_request_parameters,
                    run_context,
                )
            )

        async with self._dispatch_with_opening_retries(
            "request_stream", dispatch
        ) as response:
            yield response
