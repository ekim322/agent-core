"""Storage policy contains projection failures and observes detached writes."""

import asyncio
from types import SimpleNamespace

import pytest
from observability import bind_observability_context
from observability.correlation import current_observability_context
from pydantic_ai.messages import ModelRequest, UserPromptPart

from agent_core.agent._execution.persistence import save_messages
from agent_core.persistence import InMemoryChatWriter
from agent_core.persistence.background import BackgroundPersistence
from agent_core.utils.telemetry.run_state import RunState


async def persist(writer, **overrides):
    values = dict(
        result=SimpleNamespace(all_messages=lambda: [ModelRequest([UserPromptPart("Hello")])]),
        agent_run=None,
        session_id="chat",
        message_id="message",
        user_id="user",
        metadata={"agent_id": "untrusted", "custom": "value"},
        state=RunState(),
        run_error=None,
        total_latency_ms=73,
    )
    values.update(overrides)
    await save_messages(writer, "Example", "owned-id", **values)


def test_storage_batch_preserves_identity_and_failure_takes_precedence():
    writer = InMemoryChatWriter()
    asyncio.run(persist(writer, run_error=ValueError(""), max_hops_finalized=True))
    batch, = writer.batches
    assert batch.metadata == {"agent_id": "owned-id", "custom": "value"}
    assert batch.session_id == "chat" and batch.message_id == "message"
    assert [row.kind for row in batch.events] == ["user_msg", "run_failed"]
    assert batch.events[-1].content == {"detail": "ValueError"}
    assert sum(row.total_latency_ms == 73 for row in batch.events) == 1


def test_projection_failure_skips_writer_and_reports_stage(caplog):
    def broken_history():
        raise ValueError("invalid history")

    writer = InMemoryChatWriter()
    asyncio.run(persist(writer, result=SimpleNamespace(all_messages=broken_history)))
    assert writer.batches == []
    assert [r.event_name for r in caplog.records] == ["agent.persistence_build_failed"]


def test_awaited_storage_propagates_failure_without_retry(caplog):
    class BrokenWriter:
        calls = 0

        async def write(self, batch):
            self.calls += 1
            raise LookupError("storage unavailable")

    writer = BrokenWriter()
    with pytest.raises(LookupError, match="storage unavailable"):
        asyncio.run(persist(writer))
    assert writer.calls == 1
    assert [r.event_name for r in caplog.records] == [
        "agent.persistence_write_failed",
        "operation.completed",
    ]
    assert caplog.records[-1].operation == "persistence.write"
    assert caplog.records[-1].outcome == "error"


def test_awaited_storage_cancellation_reaches_writer():
    async def check():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        class Writer:
            async def write(self, batch):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        task = asyncio.create_task(persist(Writer()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    asyncio.run(check())


def test_background_storage_outlives_dispatch_and_reports_correlated_failure(caplog):
    async def check():
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        contexts, unhandled = [], []
        asyncio.get_running_loop().set_exception_handler(lambda loop, info: unhandled.append(info))

        class Writer:
            async def write(self, batch):
                contexts.append(current_observability_context())
                entered.set()
                try:
                    await release.wait()
                    raise LookupError("storage unavailable")
                finally:
                    finished.set()

        owner = BackgroundPersistence(Writer())
        with bind_observability_context(run_id="origin", agent_execution_id="execution"):
            await persist(Writer(), persist_in_background=True, background_persistence=owner)
        await entered.wait()
        assert not finished.is_set()
        assert current_observability_context() == {}
        release.set()
        await finished.wait()
        await asyncio.sleep(0)  # Let the completed task's exception observer run.
        assert unhandled == []
        context, = contexts
        assert context["run_id"] == "origin"
        assert context["agent_execution_id"] == "execution"
        assert context["agent_name"] == "Example"
        assert context["message_id"] == "message" and context["session_id"] == "chat"
        assert context["persistence_mode"] == "background"
        assert context["queue_wait_ms"] >= 0
        await owner.aclose()

    asyncio.run(check())
    failure, = [r for r in caplog.records if r.event_name == "agent.persistence_write_failed"]
    assert failure.message_id == "message" and failure.session_id == "chat"
