"""Bounded background storage preserves context, failures and shutdown ownership."""

import asyncio

import pytest
from observability import bind_observability_context
from observability.correlation import current_observability_context
from pydantic_ai.models.test import TestModel

from agent_core import BaseAgent, PersistenceQueueFull
from agent_core.persistence.background import BackgroundPersistence
from agent_core.persistence.records import RunRecords


def test_agent_keeps_plain_writer_background_mode_and_rejects_overload():
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        stored = []

        class Writer:
            async def write(self, batch):
                entered.set()
                await release.wait()
                stored.append(batch.message_id)

        agent = BaseAgent(
            TestModel(custom_output_text="answer"),
            writer=Writer(),
            background_write_concurrency=1,
            background_write_queue_size=1,
        )
        try:
            await agent.run("first", message_id="first", persist_in_background=True)
            await entered.wait()
            await agent.run("second", message_id="second", persist_in_background=True)
            with pytest.raises(PersistenceQueueFull):
                await agent.run("third", persist_in_background=True)
        finally:
            release.set()
            await agent.aclose()
        assert stored == ["first", "second"]
        with pytest.raises(RuntimeError, match="closed"):
            await agent.run("later", persist_in_background=True)

    asyncio.run(check())


def test_worker_limit_and_per_submission_context_survive_a_failure(caplog):
    async def check():
        full, release = asyncio.Event(), asyncio.Event()
        active = peak = 0
        contexts = []

        class Writer:
            async def write(self, records):
                nonlocal active, peak
                active += 1
                peak = max(active, peak)
                if active == 2:
                    full.set()
                try:
                    await release.wait()
                    contexts.append(current_observability_context()["run_id"])
                    if records.message_id == "fail":
                        raise LookupError("failed write")
                finally:
                    active -= 1

        owner = BackgroundPersistence(Writer(), max_concurrency=2, max_queued=4)
        for name in ("fail", "second", "third", "fourth"):
            with bind_observability_context(run_id=name):
                await owner.submit(RunRecords(message_id=name))
        await full.wait()
        assert peak == 2
        release.set()
        await owner.drain()
        assert peak == 2 and active == 0
        assert contexts == ["fail", "second", "third", "fourth"]
        assert current_observability_context() == {}
        await owner.aclose()

    asyncio.run(check())
    assert (
        len(
            [
                r
                for r in caplog.records
                if getattr(r, "event_name", None) == "agent.persistence_write_failed"
            ]
        )
        == 1
    )


def test_cancelled_close_waiter_does_not_cancel_accepted_storage():
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        stored = []

        class Writer:
            async def write(self, records):
                entered.set()
                await release.wait()
                stored.append(records.message_id)

        owner = BackgroundPersistence(Writer(), max_concurrency=1)
        await owner.submit(RunRecords(message_id="accepted"))
        await entered.wait()
        closing = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(RuntimeError, match="closed"):
            await owner.submit(RunRecords())
        release.set()
        await owner.aclose()
        await owner.aclose()
        assert stored == ["accepted"]

    asyncio.run(check())


def test_cancelled_adapter_write_does_not_stop_the_queue_worker():
    async def check():
        stored = []

        class Writer:
            async def write(self, records):
                if records.message_id == "cancelled":
                    raise asyncio.CancelledError
                stored.append(records.message_id)

        owner = BackgroundPersistence(Writer(), max_concurrency=1)
        await owner.submit(RunRecords(message_id="cancelled"))
        await owner.submit(RunRecords(message_id="next"))
        await asyncio.wait_for(owner.aclose(), 2)
        assert stored == ["next"]

    asyncio.run(check())
