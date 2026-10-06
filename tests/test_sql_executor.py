"""SQL admission, queue deadlines and cleanup use a dedicated bounded executor."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from agent_core.sql import (
    SQLQueryCapacityExceeded,
    SQLQueryError,
    SQLQueryExecutor,
    VirtualTable,
)


def test_sql_is_independent_of_an_occupied_default_executor():
    async def check():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        entered, release = Event(), Event()

        def occupy():
            entered.set()
            release.wait(3)

        blocker = loop.run_in_executor(None, occupy)
        while not entered.is_set():
            await asyncio.sleep(0)
        try:
            async with SQLQueryExecutor(max_workers=1) as executor:
                table = VirtualTable("rows", [{"a": 1}], executor=executor)
                assert await asyncio.wait_for(
                    table.run_sql_query("select * from rows"), 2
                ) == [{"a": 1}]
        finally:
            release.set()
            await blocker

    asyncio.run(check())


def test_queued_deadline_prevents_execution_and_releases_capacity():
    async def check():
        entered, release = Event(), Event()
        invoked = []
        executor = SQLQueryExecutor(max_workers=1, max_queued=1)

        def block(control):
            entered.set()
            assert release.wait(3)
            return "first"

        first = asyncio.create_task(executor.run(block, deadline=time.monotonic() + 2))
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            with pytest.raises(SQLQueryError, match="execution limit"):
                await executor.run(
                    lambda control: invoked.append("expired"),
                    deadline=time.monotonic() + 0.02,
                )
            assert invoked == [] and not first.done()
            queued = asyncio.create_task(
                executor.run(lambda control: "next", deadline=time.monotonic() + 2)
            )
            await asyncio.sleep(0)
            with pytest.raises(SQLQueryCapacityExceeded):
                await executor.run(lambda control: None, deadline=time.monotonic() + 2)
            release.set()
            assert await first == "first"
            assert await queued == "next"
        finally:
            release.set()
            await first
            await executor.aclose()

    asyncio.run(check())


def test_queued_cancellation_does_not_run_query_and_close_drains_active_work():
    async def check():
        entered, release = Event(), Event()
        invoked = []
        executor = SQLQueryExecutor(max_workers=1, max_queued=1)

        def block(control):
            entered.set()
            assert release.wait(3)

        now = time.monotonic
        first = asyncio.create_task(executor.run(block, deadline=now() + 2))
        assert await asyncio.to_thread(entered.wait, 2)
        queued = asyncio.create_task(
            executor.run(
                lambda control: invoked.append("cancelled"), deadline=now() + 2
            )
        )
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        closing = asyncio.create_task(executor.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(RuntimeError, match="closed"):
            await executor.run(lambda control: None, deadline=now() + 2)
        release.set()
        await first
        await executor.aclose()
        assert invoked == []

    asyncio.run(check())


def test_shared_default_can_be_closed_and_recreated():
    async def check():
        first = SQLQueryExecutor.default()
        assert await VirtualTable("rows", [{"a": 1}]).run_sql_query(
            "select * from rows"
        ) == [{"a": 1}]
        await SQLQueryExecutor.close_default()
        assert SQLQueryExecutor.default() is not first
        await SQLQueryExecutor.close_default()

    asyncio.run(check())


def test_cancelled_queued_jobs_release_inputs_while_workers_are_busy():
    import gc
    import weakref

    async def check():
        entered, release = Event(), Event()
        executor = SQLQueryExecutor(max_workers=1, max_queued=1)

        def block(control):
            entered.set()
            assert release.wait(3)

        class Payload:
            pass

        first = asyncio.create_task(executor.run(block, deadline=time.monotonic() + 2))
        assert await asyncio.to_thread(entered.wait, 2)
        references = []
        try:
            for _ in range(20):
                payload = Payload()
                references.append(weakref.ref(payload))
                queued = asyncio.create_task(
                    executor.run(
                        lambda control, value=payload: value,
                        deadline=time.monotonic() + 2,
                    )
                )
                await asyncio.sleep(0)
                queued.cancel()
                try:
                    await queued
                except asyncio.CancelledError:
                    pass
                del queued, payload
            await asyncio.sleep(0)  # Release the last completed task callback.
            gc.collect()
            assert all(reference() is None for reference in references)
        finally:
            release.set()
            await first
            await executor.aclose()

    asyncio.run(check())


def test_partial_worker_startup_failure_closes_started_threads(monkeypatch):
    from threading import Thread

    created = []

    class FailingThread(Thread):
        def start(self):
            created.append(self)
            if len(created) == 2:
                raise RuntimeError("worker startup failed")
            super().start()

    monkeypatch.setattr("agent_core.sql.executor.Thread", FailingThread)
    with pytest.raises(RuntimeError, match="worker startup failed"):
        SQLQueryExecutor(max_workers=2)
    assert not created[0].is_alive()
