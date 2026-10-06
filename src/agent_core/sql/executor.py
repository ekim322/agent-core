"""Own dedicated bounded SQL workers and submission-to-completion deadlines."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextvars import copy_context
from threading import Condition, Lock, Thread
from typing import ClassVar, TypeVar

from observability import observe_operation

from agent_core._validation import count

T = TypeVar("T")
_LOG = logging.getLogger(__name__)


class SQLQueryError(ValueError):
    """Invalid, disallowed, failed or deadline-exhausted SQL execution."""


class SQLQueryCapacityExceeded(SQLQueryError):
    """The dedicated SQL executor has no remaining running or queued capacity."""


class _QueryControl:
    """Deliver interruption to the worker-owned connection under a shared lock."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._connection = None
        self.stopped = False

    def attach(self, connection) -> None:
        with self._lock:
            if self.stopped:
                raise SQLQueryError("sql_query was interrupted before execution")
            self._connection = connection

    def interrupt(self) -> None:
        with self._lock:
            self.stopped = True
            if self._connection is not None:
                self._connection.interrupt()

    def detach(self) -> None:
        with self._lock:
            self._connection = None


class SQLQueryExecutor:
    """Run SQL on a dedicated pool with a finite accepted-work budget.

    Pass this executor to VirtualTable to own capacity and shutdown explicitly.
    The default permits four running and 32 queued jobs. Excess submissions raise
    SQLQueryCapacityExceeded immediately. Deadlines include executor queue wait;
    expired queued work is cancelled before it can open a database connection.
    Running cancellation interrupts the connection and joins worker cleanup.

    aclose rejects new work and waits for accepted jobs and pool shutdown.
    Cancelling a close waiter leaves cleanup running. Stop query callers before
    closing. Tables without an injected executor share default(); applications
    must call close_default at shutdown. Query submission may use different event
    loops; concurrent close waiters must use the same loop.
    """

    _default: ClassVar[SQLQueryExecutor | None] = None
    _default_lock: ClassVar[Lock] = Lock()

    def __init__(self, *, max_workers: int = 4, max_queued: int = 32) -> None:
        workers = count(max_workers, "max_workers", minimum=1)
        self._budget = workers + count(max_queued, "max_queued")
        self._lock = Condition()
        self._pending: set[Future] = set()
        self._queued: dict[Future, Callable] = {}
        self._closed = False
        self._closing: asyncio.Task[None] | None = None
        self._workers = [
            Thread(target=self._consume, name=f"agent_sql_{index}", daemon=True)
            for index in range(workers)
        ]
        started = []
        try:
            for worker in self._workers:
                worker.start()
                started.append(worker)
        except BaseException:
            with self._lock:
                self._closed = True
                self._lock.notify_all()
            for worker in started:
                worker.join()
            raise

    def _submit(self, operation: Callable[[], T]) -> Future[T]:
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLQueryExecutor is closed")
            if len(self._pending) >= self._budget:
                _LOG.warning(
                    "SQL execution capacity exhausted",
                    extra={"event_name": "agent.sql_capacity_exceeded"},
                )
                raise SQLQueryCapacityExceeded("SQL execution capacity exhausted")
            future: Future[T] = Future()
            context = copy_context()
            self._queued[future] = lambda: context.run(operation)
            self._pending.add(future)
            self._lock.notify()
        # A completed future invokes callbacks synchronously; register outside lock.
        future.add_done_callback(self._release)
        return future

    def _release(self, future: Future) -> None:
        with self._lock:
            # Remove cancelled queued work, including its captured table/context.
            # ThreadPoolExecutor leaves cancelled work items queued until a worker
            # consumes them, allowing cancellation storms to retain unbounded data.
            self._queued.pop(future, None)
            self._pending.discard(future)

    def _consume(self) -> None:
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._queued or self._closed)
                if not self._queued:
                    return
                future = next(iter(self._queued))
                operation = self._queued.pop(future)
                ready = future.set_running_or_notify_cancel()
            if ready:
                try:
                    result = operation()
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
                    del result
            del operation, future

    async def run(
        self, operation: Callable[[_QueryControl], T], *, deadline: float
    ) -> T:
        """Dispatch within an absolute monotonic deadline, then settle cleanup."""
        if deadline <= time.monotonic():
            raise SQLQueryError(
                "sql_query exceeded its execution limit before dispatch"
            )
        with observe_operation("sql.query") as observed:
            control = _QueryControl()
            submitted_at = time.monotonic()
            dispatched_at = None

            def execute():
                nonlocal dispatched_at
                dispatched_at = time.monotonic()
                return operation(control)

            future = self._submit(execute)
            worker = asyncio.wrap_future(future)
            budget = asyncio.timeout(max(0.0, deadline - time.monotonic()))
            try:
                async with budget:
                    return await asyncio.shield(worker)
            except (TimeoutError, asyncio.CancelledError) as error:
                if isinstance(error, TimeoutError) and not budget.expired():
                    raise
                control.interrupt()
                future.cancel()
                cancelled_during_cleanup = await self._settle(worker, control)
                if cancelled_during_cleanup and not isinstance(
                    error, asyncio.CancelledError
                ):
                    raise asyncio.CancelledError from error
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise SQLQueryError("sql_query exceeded its execution limit") from error
            finally:
                admitted_at = (
                    time.monotonic() if dispatched_at is None else dispatched_at
                )
                observed.set_attribute(
                    "queue_wait_ms", max(0, round(1000 * (admitted_at - submitted_at)))
                )

    @staticmethod
    async def _settle(worker: asyncio.Future, control: _QueryControl) -> bool:
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if not worker.cancelled():
                    cancelled = True
                control.interrupt()
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        return cancelled

    def _join_workers(self, completion: Future[None]) -> None:
        try:
            for worker in self._workers:
                worker.join()
        except BaseException as error:
            completion.set_exception(error)
        else:
            completion.set_result(None)

    async def _shutdown(self, pending: tuple[Future, ...]) -> None:
        await asyncio.gather(
            *(asyncio.wrap_future(job) for job in pending), return_exceptions=True
        )
        completion: Future[None] = Future()
        Thread(
            target=self._join_workers, args=(completion,), name="agent_sql_shutdown"
        ).start()
        await asyncio.wrap_future(completion)

    async def aclose(self) -> None:
        """Drain accepted jobs and join pool threads; caller cancellation is isolated."""
        with self._lock:
            if self._closing is None:
                self._closed = True
                self._lock.notify_all()
                self._closing = asyncio.create_task(
                    self._shutdown(tuple(self._pending)), name="agent_sql_shutdown"
                )
            closing = self._closing
        await asyncio.shield(closing)

    async def __aenter__(self) -> SQLQueryExecutor:
        if self._closed:
            raise RuntimeError("SQLQueryExecutor is closed")
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    @classmethod
    def default(cls) -> SQLQueryExecutor:
        with cls._default_lock:
            owner = cls._default
            if owner is not None:
                with owner._lock:
                    if owner._closed:
                        if owner._closing is not None and not owner._closing.done():
                            raise RuntimeError("Default SQL executor is closing")
                        owner = None
            if owner is None:
                owner = cls()
                cls._default = owner
            return owner

    @classmethod
    async def close_default(cls) -> None:
        with cls._default_lock:
            owner = cls._default
        if owner is not None:
            await owner.aclose()
