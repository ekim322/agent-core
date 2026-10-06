"""Serialize process-default configuration, acquisition and retirement."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from threading import RLock
from typing import TYPE_CHECKING

from agent_core.models.runtime.routes import ConnectionProvider

if TYPE_CHECKING:
    from agent_core.models.runtime.runtime import ModelRuntime


class DefaultRuntime:
    """Own a runtime generation until its shutdown has finished.

    The reentrant lock permits runtime construction to read an endpoint snapshot
    while acquisition still holds the lifecycle lock. Resource locks are always
    acquired after this lock, so configuration cannot race with publication.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._endpoints: dict[ConnectionProvider, str] = {}
        self._owner: ModelRuntime | None = None
        self._closing: asyncio.Task[None] | None = None

    def endpoints(self) -> dict[ConnectionProvider, str]:
        with self._lock:
            return self._endpoints.copy()

    def acquire(self, create: Callable[[], ModelRuntime]) -> ModelRuntime:
        with self._lock:
            self._require_available()
            owner = self._owner
            if owner is not None and owner._can_reuse_as_default():
                return owner
            replacement = create()
            self._owner = replacement
            return replacement

    def configure(self, endpoints: Mapping[ConnectionProvider, str]) -> None:
        with self._lock:
            self._require_available()
            owner = self._owner
            if owner is not None:
                owner._update_default_endpoints(endpoints)
            self._endpoints.update(endpoints)

    def _require_available(self) -> None:
        if self._closing is not None:
            raise RuntimeError("Default runtime is closing; wait for shutdown to finish")

    async def close(self) -> None:
        with self._lock:
            if self._owner is None:
                return
            if self._closing is None:
                self._closing = asyncio.create_task(
                    self._retire(self._owner), name="default_model_runtime_shutdown"
                )
                self._closing.add_done_callback(self._observe_close)
            pending = self._closing
        await asyncio.shield(pending)

    async def _retire(self, owner: ModelRuntime) -> None:
        try:
            await owner.aclose()
        finally:
            with self._lock:
                self._owner = None
                self._closing = None

    @staticmethod
    def _observe_close(task: asyncio.Task[None]) -> None:
        # A cancelled waiter may leave nobody awaiting the shutdown outcome.
        if not task.cancelled():
            task.exception()
