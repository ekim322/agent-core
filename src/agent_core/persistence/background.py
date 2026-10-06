"""Bound best-effort storage work and drain it under a caller-owned lifecycle."""

from __future__ import annotations

import asyncio
import logging
from contextvars import Context, copy_context
from dataclasses import dataclass
from time import perf_counter

from observability import bind_observability_context, observe_operation

from agent_core._validation import count
from agent_core.persistence.records import RunRecords
from agent_core.persistence.writer import ChatWriter

_LOG = logging.getLogger("agent_core._execution.storage")


class PersistenceQueueFull(RuntimeError):
    """Background storage cannot accept another batch within its queue budget."""


async def write_records(
    writer: ChatWriter, records: RunRecords, *, queued_at: float | None = None
) -> None:
    """Measure actual storage completion and propagate failure without retrying.

    Background workers supply a monotonic queue timestamp so queue wait is
    reported separately from writer duration. Safe batch identifiers connect
    writer logs and the completion record to the originating execution.
    """
    fields: dict[str, object] = {
        key: value
        for key, value in {
            "agent_name": records.agent_name,
            "message_id": records.message_id,
            "session_id": records.session_id,
        }.items()
        if value is not None
    }
    fields["persistence_mode"] = "awaited" if queued_at is None else "background"
    fields["queue_wait_ms"] = None
    if queued_at is not None:
        fields["queue_wait_ms"] = max(0, round((perf_counter() - queued_at) * 1000, 3))
    with bind_observability_context(**fields), observe_operation("persistence.write"):
        try:
            await writer.write(records)
        except asyncio.CancelledError:
            _LOG.info(
                "Storage write was cancelled",
                extra={
                    "event_name": "agent.persistence_write_cancelled",
                    "agent_name": records.agent_name,
                    "message_id": records.message_id,
                    "session_id": records.session_id,
                },
            )
            raise
        except Exception:
            _LOG.exception(
                "Invocation records could not be stored",
                extra={
                    "event_name": "agent.persistence_write_failed",
                    "agent_name": records.agent_name,
                    "message_id": records.message_id,
                    "session_id": records.session_id,
                },
            )
            raise


@dataclass(frozen=True)
class _Submission:
    records: RunRecords
    context: Context
    queued_at: float


class BackgroundPersistence:
    """Run bounded best-effort storage under one agent's lifecycle.

    Submission returns after queue admission; a full queue raises
    PersistenceQueueFull. Fixed workers preserve the submitting execution's
    context and log write failures without stopping subsequent queued work.
    drain waits for accepted work; aclose also rejects new submissions and
    stops workers. Cancelling a close waiter leaves shutdown running.
    The supplied storage adapter stays caller-owned. Use one event loop and
    do not mutate queued records. Process loss can still lose accepted work.
    """

    def __init__(
        self, writer: ChatWriter, *, max_concurrency: int = 4, max_queued: int = 128
    ) -> None:
        self._writer = writer
        self._capacity = count(max_concurrency, "max_concurrency", minimum=1)
        self._queue: asyncio.Queue[_Submission] = asyncio.Queue(
            maxsize=count(max_queued, "max_queued", minimum=1)
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False
        self._closing: asyncio.Task[None] | None = None

    def _require_loop(self) -> None:
        current = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = current
        elif self._loop is not current:
            raise RuntimeError("BackgroundPersistence must use its owning event loop")

    async def submit(self, records: RunRecords) -> None:
        """Accept a batch immediately or reject it; never accumulate admission waiters."""
        self._require_loop()
        if self._closed:
            raise RuntimeError("BackgroundPersistence is closed")
        # Capture the submitting span before starting the admission span.
        submission_context = copy_context()
        queue_error: asyncio.QueueFull | None = None
        fields: dict[str, object] = {
            key: value
            for key, value in {
                "agent_name": records.agent_name,
                "message_id": records.message_id,
                "session_id": records.session_id,
            }.items()
            if value is not None
        }
        with (
            bind_observability_context(
                **fields, persistence_mode="background", queue_wait_ms=None
            ),
            observe_operation("persistence.enqueue") as observed,
        ):
            try:
                self._queue.put_nowait(
                    _Submission(records, submission_context, perf_counter())
                )
            except asyncio.QueueFull as error:
                queue_error = error
                observed.set_outcome("rejected")
                _LOG.warning(
                    "Background storage queue is full",
                    extra={"event_name": "agent.persistence_queue_full"},
                )
            else:
                if not self._workers:
                    # Idle workers must not retain the first invocation's payload/context.
                    self._workers = [
                        asyncio.create_task(
                            self._consume(),
                            name="agent_persistence_worker",
                            context=Context(),
                        )
                        for _ in range(self._capacity)
                    ]
        if queue_error is not None:
            raise PersistenceQueueFull(
                "Background storage queue is full"
            ) from queue_error

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                operation = asyncio.create_task(
                    write_records(self._writer, item.records, queued_at=item.queued_at),
                    name="agent_persistence_write",
                    context=item.context,
                )
                try:
                    await operation
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    # The adapter cancelled this write, not the queue worker.
                except Exception:
                    # write_records emitted the originating failure. Keep consuming.
                    pass
            finally:
                self._queue.task_done()
                del item, operation

    async def drain(self) -> None:
        """Wait until all currently accepted writes have settled, including failures."""
        self._require_loop()
        await self._queue.join()

    async def _shutdown(self) -> None:
        await self._queue.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def aclose(self) -> None:
        """Stop admission and complete accepted writes despite waiter cancellation."""
        self._require_loop()
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.create_task(
                self._shutdown(), name="agent_persistence_shutdown", context=Context()
            )
        await asyncio.shield(self._closing)

    async def __aenter__(self) -> BackgroundPersistence:
        self._require_loop()
        if self._closed:
            raise RuntimeError("BackgroundPersistence is closed")
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
