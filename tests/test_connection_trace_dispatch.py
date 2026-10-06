"""HTTP trace composition preserves admission, outcome and cancellation order."""

import asyncio

import httpx2
import pytest

from agent_core.models.runtime.circuit import _ConnectionCircuitScope, _OriginCircuit
from agent_core.models.runtime.errors import ModelTransportUnavailable


def test_existing_callback_surrounds_admission_and_observes_recovery():
    circuit = _OriginCircuit(failure_threshold=1, cooldown_seconds=0)
    circuit.record_failure(circuit.acquire())
    notifications = []

    class PreviousCallback:
        def __bool__(self):
            return False

        async def __call__(self, name, details):
            notifications.append((name, circuit.is_open))

    async def check():
        scope = _ConnectionCircuitScope(circuit)
        request = httpx2.Request("GET", "http://target.test")
        request.extensions["trace"] = PreviousCallback()
        with scope.bind():
            scope.attach(request)
            callback = request.extensions["trace"]
            await callback("connection.connect_tcp.started", {})
            await callback("connection.connect_tcp.complete", {})
        assert notifications == [
            ("connection.connect_tcp.started", True),
            ("connection.connect_tcp.complete", False),
        ]
        assert scope.transport_unavailable is None

    asyncio.run(check())


@pytest.mark.parametrize("cancel_on", ["started", "complete"])
def test_callback_cancellation_propagates_and_releases_probe(cancel_on):
    circuit = _OriginCircuit(failure_threshold=1, cooldown_seconds=0)
    circuit.record_failure(circuit.acquire())

    async def previous(name, details):
        if name.endswith(cancel_on):
            raise asyncio.CancelledError()

    async def check():
        scope = _ConnectionCircuitScope(circuit)
        request = httpx2.Request("GET", "https://target.test")
        request.extensions["trace"] = previous
        with pytest.raises(asyncio.CancelledError), scope.bind():
            scope.attach(request)
            callback = request.extensions["trace"]
            await callback("connection.connect_tcp.started", {})
            await callback("connection.connect_tcp.complete", {})
        # Incomplete TLS setup releases the recovery permit without counting a
        # callback cancellation as another connection failure.
        assert circuit.is_open
        replacement = circuit.acquire()
        circuit.abort(replacement)
        assert scope.transport_unavailable is None

    asyncio.run(check())


@pytest.mark.parametrize("callback_rejects", [False, True])
def test_only_circuit_rejections_are_remembered_by_operation(callback_rejects):
    circuit = _OriginCircuit(failure_threshold=1, cooldown_seconds=60)
    circuit.record_failure(circuit.acquire())
    callback_error = ModelTransportUnavailable("callback failed", retry_after_seconds=0)
    notifications = []

    async def previous(name, details):
        notifications.append(name)
        if callback_rejects:
            raise callback_error

    async def check():
        scope = _ConnectionCircuitScope(circuit)
        request = httpx2.Request("GET", "http://target.test")
        request.extensions["trace"] = previous
        with pytest.raises(ModelTransportUnavailable) as failure, scope.bind():
            scope.attach(request)
            await request.extensions["trace"]("connection.connect_tcp.started", {})
        assert notifications == ["connection.connect_tcp.started"]
        if callback_rejects:
            assert failure.value is callback_error
            assert scope.transport_unavailable is None
        else:
            assert scope.transport_unavailable is failure.value

    asyncio.run(check())
