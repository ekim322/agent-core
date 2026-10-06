"""Circuit permits belong to actual connection setup, never pooled requests."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import httpcore
import httpcore2
import httpx
import httpx2

from agent_core._validation import count, exception_chain, seconds
from agent_core.models.runtime.errors import ModelTransportUnavailable

_LOG = logging.getLogger("agent_core.utils.model_runtime.circuit")
_CONNECT = tuple(
    getattr(module, name)
    for module in (httpcore, httpcore2, httpx, httpx2)
    for name in ("ConnectError", "ConnectTimeout", "ProxyError")
)
_POOL = tuple(module.PoolTimeout for module in (httpcore, httpcore2, httpx, httpx2))


class TransportErrors:
    @staticmethod
    def is_connect_side_error(exc: BaseException) -> bool:
        return any(isinstance(cause, _CONNECT) for cause in exception_chain(exc))

    @staticmethod
    def is_pool_side_error(exc: BaseException) -> bool:
        return any(isinstance(cause, _POOL) for cause in exception_chain(exc))


@dataclass(frozen=True)
class _CircuitPermit:
    generation: int
    probe: bool = False


class _OriginCircuit:
    """Generation guards keep stale connection outcomes from resetting outages."""

    def __init__(self, *, failure_threshold: int, cooldown_seconds: float) -> None:
        self._threshold = count(failure_threshold, "failure_threshold")
        self._cooldown = seconds(cooldown_seconds, "cooldown_seconds")
        self._failures = 0
        self._blocked_until: float | None = None
        self._probe: _CircuitPermit | None = None
        self._generation = 0

    @property
    def is_open(self) -> bool:
        return self._blocked_until is not None

    def reject_if_open(self) -> None:
        if not self.is_open:
            return
        remaining = max(0.0, self._blocked_until - time.monotonic())
        if remaining or self._probe is not None:
            raise ModelTransportUnavailable(
                (
                    "Model connection circuit is awaiting recovery"
                    if remaining
                    else "Model connection recovery probe is busy"
                ),
                retry_after_seconds=remaining or 1.0,
            )

    def acquire(self) -> _CircuitPermit:
        self.reject_if_open()
        permit = _CircuitPermit(self._generation, self.is_open)
        if permit.probe:
            self._probe = permit
        return permit

    def abort(self, permit: _CircuitPermit) -> None:
        if self._probe is permit:
            self._probe = None

    def _owns_state(self, permit: _CircuitPermit) -> bool:
        if permit.generation != self._generation:
            return False
        return not self.is_open or self._probe is permit

    def record_success(self, permit: _CircuitPermit) -> None:
        if not self._owns_state(permit):
            return
        recovering = self.is_open
        self._failures = 0
        self._blocked_until = None
        self._probe = None
        if recovering:
            self._generation += 1
            _LOG.info(
                "Connection recovery succeeded",
                extra={"event_name": "model.circuit_closed"},
            )

    def record_failure(self, permit: _CircuitPermit) -> None:
        if not self._threshold or not self._owns_state(permit):
            return
        self._failures += 1
        if self.is_open or self._failures >= self._threshold:
            self._generation += 1
            self._blocked_until = time.monotonic() + self._cooldown
            self._probe = None
            _LOG.warning(
                "Connection attempts paused",
                extra={
                    "event_name": "model.circuit_opened",
                    "failure_count": self._failures,
                    "failure_threshold": self._threshold,
                    "cooldown_seconds": self._cooldown,
                },
            )


_TraceCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


def _hostname(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    return str(value or "").rstrip(".").lower()


class _ConnectionCircuitTrace:
    """Follow TCP, proxy negotiation and target TLS until one setup resolves."""

    def __init__(
        self,
        *,
        circuit: _OriginCircuit,
        target_scheme: str,
        target_host: object,
    ) -> None:
        self._circuit = circuit
        self._secure = target_scheme.lower() in {"https", "wss"}
        self._host = _hostname(target_host)
        self._permit: _CircuitPermit | None = None
        self._resolved = False
        self._target_tls: set[str] = set()

    @property
    def has_pending_attempt(self) -> bool:
        return self._permit is not None and not self._resolved

    def observe(self, name: str, info: dict[str, Any]) -> None:
        operation, _, phase = name.rpartition(".")
        if operation.endswith(("connect_tcp", "connect_unix_socket")):
            if phase == "started" and self._permit is None and not self._resolved:
                self._permit = self._circuit.acquire()
            elif phase == "failed":
                self._settle_connection_attempt(info.get("exception"), failed=True)
            elif (
                phase == "complete"
                and not self._secure
                and not operation.startswith("socks.")
            ):
                self._settle_connection_attempt(None)
        elif operation.endswith("setup_socks5_connection"):
            if phase == "failed":
                self._settle_connection_attempt(info.get("exception"), failed=True)
            elif phase == "complete" and not self._secure:
                self._settle_connection_attempt(None)
        elif operation.endswith("start_tls"):
            if (
                phase == "started"
                and _hostname(info.get("server_hostname")) == self._host
            ):
                self._target_tls.add(operation)
            elif phase == "failed":
                self._target_tls.discard(operation)
                self._settle_connection_attempt(info.get("exception"), failed=True)
            elif phase == "complete" and operation in self._target_tls:
                self._target_tls.remove(operation)
                self._settle_connection_attempt(None)

    def _settle_connection_attempt(
        self, error: object, *, failed: bool = False, abort: bool = False
    ) -> None:
        if not self.has_pending_attempt:
            return
        permit = self._permit
        self._resolved = True
        if abort or isinstance(error, (asyncio.CancelledError, GeneratorExit)):
            self._circuit.abort(permit)
        elif failed:
            self._circuit.record_failure(permit)
        else:
            self._circuit.record_success(permit)

    def finish(self, exc: BaseException | None) -> None:
        failed = exc is not None and TransportErrors.is_connect_side_error(exc)
        self._settle_connection_attempt(exc, failed=failed, abort=not failed)


class _ConnectionTraceDispatch:
    """Compose HTTP trace notifications around synchronous circuit decisions.

    Existing started callbacks run before admission; outcome callbacks run after
    the circuit is updated. Callback errors propagate without being mistaken for
    circuit rejection. The operation scope receives only admission failures.
    """

    def __init__(
        self,
        observer: _ConnectionCircuitTrace,
        previous: _TraceCallback | None,
        report_rejection: Callable[[ModelTransportUnavailable], None],
    ) -> None:
        self._observer = observer
        self._report_rejection = report_rejection
        self._before_admission = self._after_outcome = (self._apply_circuit_event,)
        if previous is not None:
            self._before_admission = (previous, self._apply_circuit_event)
            self._after_outcome = (self._apply_circuit_event, previous)

    async def __call__(self, name: str, details: dict[str, Any]) -> None:
        notifications = self._after_outcome
        if name.endswith(".started"):
            notifications = self._before_admission
        for notify in notifications:
            await notify(name, details)

    async def _apply_circuit_event(self, name: str, details: dict[str, Any]) -> None:
        try:
            self._observer.observe(name, details)
        except ModelTransportUnavailable as denial:
            self._report_rejection(denial)
            raise


_ACTIVE: ContextVar[_ConnectionCircuitScope | None] = ContextVar(
    "model_connection_scope", default=None
)
_MARKER = "model_runtime_connection_circuit"


class _ConnectionCircuitScope:
    """Associate HTTP hooks with one operation and settle incomplete setups."""

    def __init__(self, circuit: _OriginCircuit) -> None:
        self.circuit = circuit
        self._traces: list[_ConnectionCircuitTrace] = []
        self.transport_unavailable: ModelTransportUnavailable | None = None

    def _remember(self, error: ModelTransportUnavailable) -> None:
        self.transport_unavailable = error

    def attach(self, request: httpx2.Request) -> None:
        if request.extensions.get(_MARKER):
            return
        trace = _ConnectionCircuitTrace(
            circuit=self.circuit,
            target_scheme=request.url.scheme,
            target_host=request.extensions.get("sni_hostname") or request.url.raw_host,
        )
        dispatcher = _ConnectionTraceDispatch(
            trace, request.extensions.get("trace"), self._remember
        )
        request.extensions.update(trace=dispatcher, **{_MARKER: True})
        self._traces.append(trace)

    def finish(self, exc: BaseException | None) -> None:
        pending = [trace for trace in self._traces if trace.has_pending_attempt]
        for index, trace in enumerate(pending):
            trace.finish(exc if index == len(pending) - 1 else None)

    @contextmanager
    def bind(self):
        token = _ACTIVE.set(self)
        try:
            yield
        except BaseException as error:
            self.finish(error)
            if (
                not isinstance(error, (asyncio.CancelledError, GeneratorExit))
                and self.transport_unavailable is not None
            ):
                if self.transport_unavailable is not error:
                    raise self.transport_unavailable from error
            raise
        else:
            self.finish(None)
            if self.transport_unavailable is not None:
                raise self.transport_unavailable
        finally:
            _ACTIVE.reset(token)


class _ConnectionCircuitRequestHook:
    def __init__(self, circuit: _OriginCircuit) -> None:
        self._circuit = circuit

    async def __call__(self, request: httpx2.Request) -> None:
        active = _ACTIVE.get()
        if active is not None and active.circuit is self._circuit:
            active.attach(request)
