"""Embed ordered text inputs with bounded, sequential Google GenAI requests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types
from observability import bind_observability_context, observe_operation

DEFAULT_VECTOR_EMBEDDING_MODEL = "text-embedding-005"
_MAX_CHARS_PER_INPUT = 7_000
_MAX_CHARS_PER_REQUEST = 60_000
_MAX_INPUTS_PER_REQUEST = 8


class VectorEmbedder:
    """Return one vector for a string, or ordered vectors for a list of strings.

    Each input is truncated to 7,000 characters. Requests contain at most eight
    inputs and 60,000 characters; these are token-budget heuristics, and the
    provider also receives auto_truncate=True. Applications needing every part
    of a long document should split it before embedding.

    A supplied async SDK client stays caller-owned. Otherwise the client is
    created on first use; call aclose when work has stopped. Provider failures
    and cancellation propagate without retries in this component.
    """

    def __init__(
        self, model: str = DEFAULT_VECTOR_EMBEDDING_MODEL, *, client: Any | None = None
    ) -> None:
        self.model = model
        self._provided_client = client
        self._created_client = None

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Validate all inputs, then embed and validate one bounded batch at a time."""
        single = isinstance(text, str)
        inputs = [text] if single else list(text)
        for position, item in enumerate(inputs):
            if not isinstance(item, str):
                raise TypeError(f"Embedding input at index {position} must be text")
        vectors = []
        for batch in self._request_batches(inputs):
            vectors.extend(await self._request_vectors(batch))
        if single:
            return vectors[0]
        return vectors

    @staticmethod
    def _request_batches(inputs: list[str]) -> Iterator[list[str]]:
        """Construct only the next request; never materialize a second batch list."""
        offset = 0
        while offset < len(inputs):
            batch = []
            remaining = _MAX_CHARS_PER_REQUEST
            end = min(len(inputs), offset + _MAX_INPUTS_PER_REQUEST)
            while offset < end:
                value = inputs[offset][:_MAX_CHARS_PER_INPUT]
                if len(value) > remaining:
                    break
                batch.append(value)
                remaining -= len(value)
                offset += 1
            yield batch

    async def _request_vectors(self, batch: list[str]) -> list[list[float]]:
        """Include response validation in the request's observed outcome."""
        with (
            bind_observability_context(
                provider="google", model_name=self.model, input_count=len(batch)
            ),
            observe_operation("embedding.request"),
        ):
            client = self._provided_client
            if client is None:
                if self._created_client is None:
                    self._created_client = genai.Client().aio
                client = self._created_client
            response = await client.models.embed_content(
                contents=batch,
                config=types.EmbedContentConfig(auto_truncate=True),
                model=self.model,
            )
            returned = response.embeddings or []
            if len(returned) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} embedding vectors; provider returned {len(returned)}"
                )
            vectors = []
            for position, embedding in enumerate(returned):
                if embedding.values is None or not embedding.values:
                    raise RuntimeError(
                        f"embedding at index {position} has no vector values"
                    )
                vectors.append(list(embedding.values))
            return vectors

    async def aclose(self) -> None:
        """Release the lazily created client; leave injected clients untouched."""
        owned = self._created_client
        if owned is not None:
            await owned.aclose()
            self._created_client = None
