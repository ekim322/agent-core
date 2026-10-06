from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field

from agent_core.persistence.records import EventRecord, RunRecords


class ChatWriter(ABC):
    """Store a batch of agent or completion records through an application adapter.

    Implement write and inject the adapter into BaseAgent or LLMClient. The
    adapter owns transactions, storage retries and durable commit semantics.
    Awaited writes propagate failures to the caller; BaseAgent's optional
    background mode schedules the write without waiting for confirmation.
    """

    @abstractmethod
    async def write(self, records: RunRecords) -> None: ...


@dataclass(eq=False, repr=False)
class InMemoryChatWriter(ChatWriter):
    """Record independent batch snapshots for local inspection and tests.

    A completed write keeps the values submitted at that moment, even if the
    caller later extends or edits its record builder. The retained batches and
    their event objects remain directly editable for local inspection.
    """

    batches: list[RunRecords] = field(default_factory=list, init=False)

    async def write(self, records: RunRecords) -> None:
        snapshot = deepcopy(records)
        self.batches.append(snapshot)

    def iter_events(self) -> Iterator[EventRecord]:
        """Read events in batch order without allocating a combined event list."""
        for batch in self.batches:
            yield from batch.events

    @property
    def all_events(self) -> list[EventRecord]:
        return list(self.iter_events())
