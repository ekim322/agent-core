from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_core.models.embeddings import VectorEmbedder


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_content(
        self, *, model: str, contents: list[str], config: object
    ) -> object:
        self.calls.append(contents)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[float(len(text))]) for text in contents]
        )


def test_embed_preserves_order_and_batches_requests() -> None:
    models = FakeModels()
    embedder = VectorEmbedder(client=SimpleNamespace(models=models))

    result = asyncio.run(embedder.embed([str(index) for index in range(9)]))

    assert result == [[1.0]] * 9
    assert [len(call) for call in models.calls] == [8, 1]


def test_embed_returns_one_vector_for_one_string() -> None:
    models = FakeModels()
    embedder = VectorEmbedder(client=SimpleNamespace(models=models))

    result = asyncio.run(embedder.embed("hello"))

    assert result == [5.0]


def test_embed_rejects_non_string_batch_values() -> None:
    embedder = VectorEmbedder(client=SimpleNamespace(models=FakeModels()))

    with pytest.raises(TypeError, match="input at index 1 must be text"):
        asyncio.run(embedder.embed(["valid", 3]))  # type: ignore[list-item]


def test_close_only_releases_owned_clients(monkeypatch):
    from unittest.mock import AsyncMock
    from agent_core.models import embeddings as embedding_model

    async def check():
        client = SimpleNamespace(models=FakeModels(), aclose=AsyncMock())
        monkeypatch.setattr(
            embedding_model.genai, "Client", lambda: SimpleNamespace(aio=client)
        )
        owned = VectorEmbedder()
        await owned.embed("text")
        await owned.aclose()
        await owned.aclose()
        client.aclose.assert_awaited_once()
        client.aclose.reset_mock()
        await VectorEmbedder(client=client).aclose()
        client.aclose.assert_not_awaited()

    asyncio.run(check())


@pytest.mark.parametrize("returned_count", [7, 9])
def test_invalid_batch_count_stops_before_subsequent_requests(returned_count):
    class WrongCount(FakeModels):
        async def embed_content(self, **kwargs):
            self.calls.append(kwargs["contents"])
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0])] * returned_count
            )

    models = WrongCount()
    with pytest.raises(RuntimeError, match=f"Expected 8 embedding vectors; provider returned {returned_count}"):
        asyncio.run(
            VectorEmbedder(client=SimpleNamespace(models=models)).embed(["text"] * 9)
        )
    assert len(models.calls) == 1


@pytest.mark.parametrize("values", [None, []])
def test_missing_vector_values_are_rejected(values):
    class MissingVector(FakeModels):
        async def embed_content(self, **kwargs):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])

    with pytest.raises(RuntimeError, match="no vector values"):
        asyncio.run(
            VectorEmbedder(client=SimpleNamespace(models=MissingVector())).embed("text")
        )


def test_empty_inputs_avoid_io_and_truncation_preserves_order():
    models = FakeModels()
    embedder = VectorEmbedder(client=SimpleNamespace(models=models))
    assert asyncio.run(embedder.embed([])) == []
    assert models.calls == []
    vectors = asyncio.run(embedder.embed(["a" * 8_000, "", "b"]))
    assert vectors == [[7000.0], [0.0], [1.0]]
    assert models.calls == [["a" * 7_000, "", "b"]]


def test_embedding_cancellation_propagates_without_issuing_next_batch():
    async def check():
        entered = asyncio.Event()

        class BlockingModels(FakeModels):
            async def embed_content(self, **kwargs):
                self.calls.append(kwargs["contents"])
                entered.set()
                await asyncio.Event().wait()

        models = BlockingModels()
        task = asyncio.create_task(
            VectorEmbedder(client=SimpleNamespace(models=models)).embed(["text"] * 9)
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(models.calls) == 1

    asyncio.run(check())
