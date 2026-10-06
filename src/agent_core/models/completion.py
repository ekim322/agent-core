"""Await raw or validated completions, with optional storage and local capacity."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai import Agent, direct
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings

from agent_core._validation import count
from agent_core.models.config import ModelConfig
from agent_core.models.runtime import ModelRuntime
from agent_core.persistence import ChatWriter, RunRecords
from agent_core.persistence.background import write_records
from agent_core.streaming import StreamEvent
from agent_core.utils.telemetry.sub_call import SubCallTrace

logger = logging.getLogger("agent_core.llm_client")


class LLMClient:
    """Request an answer directly, or let the SDK validate structured output.

    Raw calls use the SDK direct API. Structured calls use a tool-free SDK agent
    and may request corrections when output validation fails. Both paths can
    store their complete history through a caller-supplied writer.

    A local concurrency permit covers the complete call, including awaited
    storage, or the lifetime of an open stream. ModelRuntime separately owns
    shared provider connections and transport retries. Custom models retain
    the transport behavior supplied by their owner.
    """

    def __init__(
        self,
        model: Model | str,
        *,
        config: ModelConfig | None = None,
        max_concurrency: int | None = None,
        writer: ChatWriter | None = None,
    ):
        if max_concurrency is not None:
            count(max_concurrency, "max_concurrency", minimum=1)
        self._writer = writer
        self._config = config
        self._semaphore = None
        if max_concurrency is not None:
            self._semaphore = asyncio.Semaphore(max_concurrency)
        self.model, self._model_name = self._model_identity(model)

    @staticmethod
    def _model_identity(selection: Model | str) -> tuple[Model, str]:
        if isinstance(selection, str):
            return ModelRuntime.make_model(selection), selection
        return selection, getattr(selection, "model_name", "")

    def _select_call(
        self,
        model: Model | str | None,
        config: ModelConfig | None,
    ) -> tuple[Model, ModelSettings | None]:
        selected = (self.model, self._model_name)
        if model is not None:
            selected = self._model_identity(model)
        active_config = self._config
        if config is not None:
            active_config = config
        settings = None
        if active_config is not None:
            settings = active_config.to_settings(selected[1])
        return selected[0], settings

    @asynccontextmanager
    async def _call_capacity(self):
        """Release the local permit on completion, storage failure or cancellation."""
        if self._semaphore is None:
            yield
        else:
            async with self._semaphore:
                yield

    async def run(
        self,
        messages: list[ModelMessage],
        *,
        model: Model | str | None = None,
        config: ModelConfig | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        user_id: str | None = None,
        trace_metadata: SubCallTrace | None = None,
        persist: bool = True,
        output_type: OutputSpec[Any] | None = None,
    ) -> ModelResponse | Any:
        """Return a raw response or validated answer after optional awaited storage.

        output_type owns SDK output parameters, so it cannot accompany explicit
        model_request_parameters. Structured correction messages are included
        in storage. Conversion failures log and skip storage; writer failures
        and cancellation propagate. Identifiers tag stored rows; SubCallTrace
        supplies nested-call metadata. Model/config overrides apply to this call.
        """
        if model_request_parameters is not None and output_type is not None:
            raise ValueError(
                "model_request_parameters and output_type are mutually exclusive"
            )
        selected, settings = self._select_call(model, config)
        async with self._call_capacity():
            answer, history = await self._answer_history(
                messages,
                selected,
                settings,
                output_type,
                model_request_parameters,
            )
            if persist and self._writer is not None:
                await self._store_history(
                    history, session_id, message_id, user_id, trace_metadata
                )
            return answer

    @staticmethod
    async def _answer_history(
        messages: list[ModelMessage],
        model: Model,
        settings: ModelSettings | None,
        output_type: OutputSpec[Any] | None,
        parameters: ModelRequestParameters | None,
    ) -> tuple[Any, list[ModelMessage]]:
        """Keep the answer paired with the exact history that produced it."""
        if output_type is None:
            response = await direct.model_request(
                model=model,
                messages=messages,
                model_request_parameters=parameters,
                model_settings=settings,
            )
            history = list(messages)
            history.append(response)
            return response, history
        validator = Agent(model=model, output_type=output_type)
        validated = await validator.run(
            model_settings=settings, message_history=messages
        )
        return validated.output, list(validated.all_messages())

    async def _store_history(
        self,
        history: list[ModelMessage],
        session_id: str | None,
        message_id: str | None,
        user_id: str | None,
        trace_metadata: SubCallTrace | None,
    ) -> None:
        try:
            metadata = None
            if trace_metadata is not None:
                metadata = trace_metadata.to_metadata()
            batch = RunRecords(
                message_id=message_id,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
            )
            batch.add_run(history)
        except Exception:
            logger.exception(
                "Completion history could not be projected; storage was skipped",
                extra={"event_name": "agent.persistence_build_failed"},
            )
            return
        if not batch.is_empty():
            await write_records(self._writer, batch)

    async def stream(
        self,
        messages: list[ModelMessage],
        *,
        model: Model | str | None = None,
        config: ModelConfig | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield visible text/reasoning updates while holding call capacity.

        This operation does not store history or emit a terminal event. Identifier
        arguments remain unused. Close partially consumed iterators with aclosing
        so the SDK response and local permit are released promptly.
        """
        selected, settings = self._select_call(model, config)
        async with self._call_capacity():
            async with direct.model_request_stream(
                model=selected,
                messages=messages,
                model_request_parameters=model_request_parameters,
                model_settings=settings,
            ) as response:
                async for update in response:
                    visible = self._visible_update(update)
                    if visible is not None:
                        yield visible

    @staticmethod
    def _visible_update(event: ModelResponseStreamEvent) -> StreamEvent | None:
        """Map either an initial part or an incremental delta to visible output."""
        if isinstance(event, PartStartEvent):
            part = event.part
            content = getattr(part, "content", None)
        elif isinstance(event, PartDeltaEvent):
            part = event.delta
            content = getattr(part, "content_delta", None)
        else:
            return None
        if not content:
            return None
        if isinstance(part, (ThinkingPart, ThinkingPartDelta)):
            return StreamEvent.reasoning_delta(content)
        if isinstance(part, (TextPart, TextPartDelta)):
            return StreamEvent.text_delta(content)
        return None
