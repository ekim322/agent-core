"""Memory storage behaves like a write boundary for mutable record builders."""

import asyncio

from agent_core.persistence import EventRecord, InMemoryChatWriter, RunRecords


def test_each_write_keeps_an_independent_batch_and_nested_payload_snapshot():
    async def check():
        writer = InMemoryChatWriter()
        records = RunRecords(
            message_id="first",
            metadata={"labels": ["initial"]},
            events=[EventRecord(kind="text", content={"items": ["first"]})],
        )
        await writer.write(records)
        records.message_id = "second"
        records.metadata["labels"].append("later")
        records.events[0].content["items"].append("second")
        records.events.append(EventRecord(kind="text", content="new row"))
        await writer.write(records)
        records.events.clear()

        first, second = writer.batches
        assert first.message_id == "first" and second.message_id == "second"
        assert first.metadata == {"labels": ["initial"]}
        assert second.metadata == {"labels": ["initial", "later"]}
        assert first.events[0].content == {"items": ["first"]}
        assert second.events[0].content == {"items": ["first", "second"]}
        assert [row.content for row in writer.all_events] == [
            {"items": ["first"]}, {"items": ["first", "second"]}, "new row"
        ]
        assert list(writer.iter_events()) == writer.all_events

    asyncio.run(check())


def test_writer_instances_and_materialized_event_lists_are_independent():
    async def check():
        first, second = InMemoryChatWriter(), InMemoryChatWriter()
        await first.write(RunRecords(events=[EventRecord(kind="text", content="one")]))
        view = first.all_events
        view.clear()
        assert len(first.all_events) == 1
        assert second.batches == [] and second.all_events == []

    asyncio.run(check())
