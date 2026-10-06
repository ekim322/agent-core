"""Run or stream an agent with request preparation and optional message storage."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing, nullcontext
from dataclasses import fields
from time import perf_counter
from typing import Any, Generic

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRetries, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponseStreamEvent,
    HandleResponseEvent,
)
from pydantic_ai.models import Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.run import AgentRun
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from agent_core.agent._execution.event_mapping import (
    map_request_event,
    map_tool_event,
    success_event,
    failure_event,
)
from agent_core.agent._execution.node_stream import stream_node
from agent_core.agent._execution.loop import PhaseExecutor
from agent_core.agent.request import Deps, Req, UserPrompt, PreparedRequest
from agent_core.agent._execution.invocation import Invocation
from agent_core.agent._execution.persistence import save_messages
from agent_core.agent._execution.tool_overrides import recoverable_tool, tool_overrides
from agent_core.streaming import EventType, StreamEvent
from agent_core.models.config import ModelConfig
from agent_core.persistence.background import BackgroundPersistence
from agent_core._validation import count
from agent_core.persistence import ChatWriter
from agent_core.agent._execution.prompts import MAX_HOPS_FINALIZE_PROMPT
from agent_core._retry import DEFAULT_MODEL_REQUEST_RETRY_ATTEMPTS
from agent_core.agent._execution.recovery import ModelRequestRetryPolicy
from agent_core.models.runtime import ModelRuntime
from agent_core.streaming import StreamPrinter
from agent_core.utils.telemetry.execution import RunProbe, traced_agent_stream
from agent_core.utils.telemetry.run_state import RunState

logger = logging.getLogger("agent_core.base_agent")
DEFAULT_MAX_HOPS = 50


class BaseAgent(Generic[Deps, Req]):
    """Answer prompts with model calls and tools, returning or streaming the result.

    Construct with a model and optional tools, dependencies and model config.
    ``run`` returns a Pydantic AI result; ``stream`` yields normalized model and
    tool events followed by a terminal event. Both use the same execution path.
    Override ``_prepare_run`` to accept application request objects, or the node
    and event hooks to customize streaming. ``agent`` exposes the underlying
    Pydantic AI agent for registration and advanced configuration.

    ModelRuntime owns provider connections, concurrency limits and retries before
    a response opens. Agent execution can recover interrupted responses and
    request a final answer with tools disabled after ``max_hops`` model turns.
    Cancellation propagates; close partially consumed streams with
    ``contextlib.aclosing``.

    A supplied writer receives completed or interrupted messages after execution
    resources close. Writer failures propagate when writes are awaited. Callers
    can select bounded best-effort background writes. Stop invocations and await
    aclose to drain them before shutdown; process loss can still lose queued work.
    """

    def __init__(
        self,
        model: Model | str,
        *,
        config: ModelConfig | None = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[Deps] | None = None,
        tools: Sequence[Tool[Deps]] = (),
        toolsets: Sequence[AbstractToolset[Deps]] = (),
        capabilities: Sequence[AbstractCapability] = (),
        max_hops: int = DEFAULT_MAX_HOPS,
        max_hops_finalize_prompt: str = MAX_HOPS_FINALIZE_PROMPT,
        max_concurrency: int | None = None,
        tool_timeout: float | None = None,
        retries: int | AgentRetries | None = None,
        model_request_retries: int = DEFAULT_MODEL_REQUEST_RETRY_ATTEMPTS,
        agent_id: str | None = None,
        agent_name: str | None = None,
        instrument: bool | None = None,
        writer: ChatWriter | None = None,
        background_write_concurrency: int = 4,
        background_write_queue_size: int = 128,
    ):
        """Configure the model, tools and execution policy shared by calls.

        ``retries`` controls Pydantic AI validation retries: pass an integer for
        both tools and output, or ``AgentRetries`` to budget them separately.
        ``model_request_retries`` controls interrupted-response recovery. Provider
        transport retry settings belong to ModelRuntime.

        ``max_concurrency`` limits simultaneous SDK runs on this agent in addition
        to any shared model-route limit. ``tool_timeout`` is the default timeout
        for tools without their own timeout. Toolsets keep their SDK lifecycle
        and failure behavior; the recoverable-error wrapper applies to ``tools``.
        Background storage permits background_write_concurrency active writes
        and background_write_queue_size queued batches per agent. A full queue
        raises PersistenceQueueFull; call aclose to drain and stop its workers.
        """
        self._agent_id = agent_id or agent_name or type(self).__name__
        self._agent_name = agent_name or self._agent_id
        self._model_name = (
            model if isinstance(model, str) else getattr(model, "model_name", "")
        )
        self._config = config
        count(background_write_concurrency, "background_write_concurrency", minimum=1)
        count(background_write_queue_size, "background_write_queue_size", minimum=1)
        self._writer = writer
        self._background_persistence = (
            BackgroundPersistence(
                writer,
                max_concurrency=background_write_concurrency,
                max_queued=background_write_queue_size,
            )
            if writer is not None
            else None
        )
        self._max_hops = max_hops
        self._max_hops_finalize_prompt = max_hops_finalize_prompt
        self._model_request_retry_policy = ModelRequestRetryPolicy(
            max_attempts=model_request_retries
        )
        self._tools = tuple(
            recoverable_tool(tool, type(self).__name__) for tool in tools
        )
        agent_options: dict[str, Any] = {
            "model": (
                ModelRuntime.make_model(model) if isinstance(model, str) else model
            ),
            # Instructions remain attached when a caller supplies dialogue history.
            "instructions": system_prompt,
            "name": self.agent_name,
            "tools": self._tools,
            "toolsets": tuple(toolsets),
            "capabilities": [*capabilities, RunProbe()],
        }
        optional_agent_options = {
            "deps_type": deps_type,
            "max_concurrency": max_concurrency,
            "tool_timeout": tool_timeout,
            "retries": retries,
            "instrument": instrument,
        }
        agent_options.update(
            {
                name: value
                for name, value in optional_agent_options.items()
                if value is not None
            }
        )
        if config is not None:
            agent_options["model_settings"] = config.to_settings(self._model_name)
        self._agent: Agent[Deps, str] = Agent(**agent_options)

    async def drain_persistence(self) -> None:
        """Wait for this agent's accepted background writes, including failed writes."""
        if self._background_persistence is not None:
            await self._background_persistence.drain()

    async def aclose(self) -> None:
        """Drain and stop background storage after invocations have stopped.

        Cancelling a close waiter leaves shutdown running. The supplied writer,
        toolsets and model runtime remain owned by their respective callers.
        Subsequent background submissions raise; ordinary awaited writes keep
        their existing caller-owned lifecycle.
        """
        if self._background_persistence is not None:
            await self._background_persistence.aclose()

    @property
    def agent(self) -> Agent[Deps, str]:
        """Expose the Pydantic AI agent for tool registration and configuration."""
        return self._agent

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    async def run(
        self,
        user_msg: UserPrompt | None = None,
        *,
        request: Req | None = None,
        message_history: list[ModelMessage] | None = None,
        instructions: str | Sequence[str] | None = None,
        deps: Deps | None = None,
        model: Model | str | None = None,
        config: ModelConfig | None = None,
        output_type: OutputSpec[Any] | None = None,
        tools: Sequence[Tool[Deps]] | None = None,
        extra_tools: Sequence[Tool[Deps]] | None = None,
        toolsets: Sequence[AbstractToolset[Deps]] | None = None,
        max_hops: int | None = None,
        usage_limits: UsageLimits | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
        persist_in_background: bool = False,
        print_outputs: bool = False,
    ) -> AgentRunResult[Any]:
        """Return the completed result, raising execution or awaited writer failures.

        ``request`` is translated by ``_prepare_run``; explicit non-None values
        override those prepared inputs, including empty values. ``tools`` replaces
        the tools registered at construction, whereas ``extra_tools`` adds tools.
        Supply only one of these tool arguments. ``output_type`` requests
        validated structured output. Select ``persist=False`` to skip storage,
        or ``persist_in_background=True`` to schedule a best-effort write that
        raises PersistenceQueueFull when the bounded queue cannot accept a batch.
        Await aclose at shutdown to complete accepted background writes.

        ``usage_limits`` bounds SDK-accounted requests, tokens, tool calls or cost
        across recovery and finalization. Without it, ``max_hops`` controls when
        to request a final answer with tools disabled. Usage limits stop execution
        with an error; they do not reserve capacity for a final answer.
        """
        terminal: StreamEvent | None = None
        async with aclosing(
            self.stream(
                user_msg,
                request=request,
                message_history=message_history,
                instructions=instructions,
                deps=deps,
                model=model,
                config=config,
                output_type=output_type,
                tools=tools,
                extra_tools=extra_tools,
                toolsets=toolsets,
                max_hops=max_hops,
                usage_limits=usage_limits,
                session_id=session_id,
                message_id=message_id,
                user_id=user_id,
                metadata=metadata,
                persist=persist,
                persist_in_background=persist_in_background,
                print_outputs=print_outputs,
            )
        ) as events:
            async for event in events:
                if event.type is EventType.DONE:
                    terminal = event
        if terminal is not None:
            failure = terminal.data.get("exception")
            if failure is not None:
                raise failure
            result = terminal.data.get("result")
            if result is not None:
                return result
        raise RuntimeError(f"{self.agent_name}: run produced no result.")

    @traced_agent_stream
    async def stream(
        self,
        user_msg: UserPrompt | None = None,
        *,
        request: Req | None = None,
        message_history: list[ModelMessage] | None = None,
        instructions: str | Sequence[str] | None = None,
        deps: Deps | None = None,
        model: Model | str | None = None,
        config: ModelConfig | None = None,
        output_type: OutputSpec[Any] | None = None,
        tools: Sequence[Tool[Deps]] | None = None,
        extra_tools: Sequence[Tool[Deps]] | None = None,
        toolsets: Sequence[AbstractToolset[Deps]] | None = None,
        max_hops: int | None = None,
        usage_limits: UsageLimits | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
        persist_in_background: bool = False,
        print_outputs: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield model and tool events, followed by a ``DONE`` event with the outcome.

        Recoverable open-stream failures replay an unseen answer or continue
        already emitted text. Reaching ``max_hops`` starts a final phase with
        tools disabled. Cancellation and consumer closure unwind node/run scopes
        before persistence. Execution failures are carried in the terminal event;
        cancellation propagates to the caller. The terminal event precedes
        storage, so awaited writer failures can still raise after it is yielded.
        Close the iterator even when stopping at ``DONE``.

        Inputs and tool overrides follow the same rules as ``run``. Select
        ``persist=False`` to skip storage or ``persist_in_background=True`` to
        submit to the bounded background queue. Full queues raise
        PersistenceQueueFull. Await aclose after stopping invocations.
        """
        if tools is not None and extra_tools is not None:
            raise ValueError(
                "Choose a replacement tool set with tools or additions with extra_tools."
            )
        started_at = perf_counter()
        inputs = await self._merge_request(
            request,
            PreparedRequest(
                user_msg=user_msg,
                message_history=message_history,
                instructions=instructions,
                deps=deps,
                model=model,
                config=config,
                tools=tools,
                max_hops=max_hops,
                usage_limits=usage_limits,
                session_id=session_id,
                message_id=message_id,
                user_id=user_id,
                metadata=metadata,
            ),
        )
        invocation = Invocation(inputs, started_at)
        iteration_options = self._iteration_options(inputs, output_type)
        tool_override_options = tool_overrides(
            self._tools, self.agent_name, inputs.tools, extra_tools, toolsets
        )
        tool_scope = (
            self._agent.override(**tool_override_options)
            if tool_override_options
            else nullcontext()
        )
        printer = StreamPrinter() if print_outputs else None
        max_model_turns = self._max_hops if inputs.max_hops is None else inputs.max_hops
        executor = PhaseExecutor(
            self._agent,
            self.agent_name,
            self._model_request_retry_policy,
            self._max_hops_finalize_prompt,
            self._handle_node,
        )
        try:
            with tool_scope:
                try:
                    async with aclosing(
                        executor.execute(invocation, iteration_options, max_model_turns)
                    ) as events:
                        async for event in events:
                            if printer is not None:
                                printer.write(event)
                            yield event
                except Exception as failure:
                    invocation.error = failure
                    logger.exception(
                        "Agent run failed",
                        extra={
                            "event_name": "agent.run_failed",
                            "agent_name": self.agent_name,
                        },
                    )
                    terminal = failure_event(failure)
                else:
                    terminal = success_event(invocation, self.agent_name)
                if printer is not None:
                    printer.write(terminal)
                invocation.terminal_sent = True
                yield terminal
        except (asyncio.CancelledError, GeneratorExit) as interruption:
            if not invocation.terminal_sent and invocation.error is None:
                invocation.error = interruption
            raise
        finally:
            if persist:
                await self._persist_invocation(invocation, persist_in_background)

    async def _prepare_run(self, request: Req) -> PreparedRequest[Deps]:
        """Translate an application request into inputs for ``run`` and ``stream``.

        Subclasses must implement this hook to support ``request=``. Return a
        ``PreparedRequest`` containing the prompt, tool dependencies and any
        per-request settings. Explicit non-None call arguments override its fields.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _prepare_run() for request=."
        )

    async def _merge_request(
        self,
        request: Req | None,
        overrides: PreparedRequest[Deps],
    ) -> PreparedRequest[Deps]:
        """Combine prepared inputs with explicit overrides without mutating either."""
        if request is None:
            return overrides
        prepared = await self._prepare_run(request)
        # Only None inherits a prepared value; empty strings and collections
        # still express the caller's choice.
        merged_inputs = {}
        for request_field in fields(PreparedRequest):
            override_value = getattr(overrides, request_field.name)
            merged_inputs[request_field.name] = (
                getattr(prepared, request_field.name)
                if override_value is None
                else override_value
            )
        return PreparedRequest(**merged_inputs)

    def _iteration_options(
        self, inputs: PreparedRequest[Deps], output_type: OutputSpec[Any] | None
    ) -> dict[str, Any]:
        model, settings = self._resolve_run_model(inputs.model, inputs.config)
        options = dict(
            user_prompt=inputs.user_msg,
            message_history=inputs.message_history,
            deps=inputs.deps,
            model=model,
            model_settings=settings,
            output_type=output_type,
            # Keep one usage counter across recovery and finalization. Our hop
            # policy owns the default cap; explicit SDK budgets can stop earlier.
            usage=RunUsage(),
            usage_limits=(
                inputs.usage_limits
                if inputs.usage_limits is not None
                else UsageLimits(request_limit=None)
            ),
        )
        if inputs.instructions is not None:
            options["instructions"] = inputs.instructions
        return options

    def _resolve_run_model(
        self, model: Model | str | None, config: ModelConfig | None
    ) -> tuple[Model | None, ModelSettings | None]:
        """Select a per-call model and translate its effective configuration.

        With no model override, the SDK keeps the construction model and settings
        unless a per-call config is supplied. A model override uses the per-call
        config when supplied, or the construction config otherwise.
        """
        if model is None:
            return None, (
                config.to_settings(self._model_name) if config is not None else None
            )
        selected_model = (
            ModelRuntime.make_model(model) if isinstance(model, str) else model
        )
        model_name = (
            model if isinstance(model, str) else getattr(model, "model_name", "")
        )
        effective_config = self._config if config is None else config
        return (
            selected_model,
            (
                effective_config.to_settings(model_name)
                if effective_config is not None
                else None
            ),
        )

    async def _handle_node(
        self, node: Any, agent_run: AgentRun[Deps, Any], state: RunState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Customize node streaming while retaining timing and cleanup scopes."""
        async with aclosing(
            stream_node(
                node, agent_run, state, self._map_request_event, self._map_tool_event
            )
        ) as events:
            async for event in events:
                yield event

    def _map_request_event(self, event: ModelResponseStreamEvent) -> StreamEvent | None:
        """Convert a model response event; return None to omit it from the stream."""
        return map_request_event(event)

    def _map_tool_event(self, event: HandleResponseEvent) -> StreamEvent | None:
        """Convert a tool event; return None to omit it from the stream."""
        return map_tool_event(event)

    async def _persist_invocation(
        self, invocation: Invocation[Deps], background: bool
    ) -> None:
        inputs = invocation.inputs
        elapsed_ms = max(0, int((perf_counter() - invocation.started_at) * 1000))
        await self._save_messages(
            result=invocation.result,
            agent_run=invocation.run,
            state=invocation.state,
            run_error=invocation.error,
            total_latency_ms=elapsed_ms,
            session_id=inputs.session_id,
            message_id=inputs.message_id,
            user_id=inputs.user_id,
            metadata=inputs.metadata,
            partial_message_start_index=invocation.history_boundary,
            max_hops_finalized=invocation.finalized,
            persist_in_background=background,
        )

    async def _save_messages(
        self,
        *,
        result: AgentRunResult[Any] | None,
        agent_run: AgentRun[Deps, Any] | None,
        state: RunState,
        run_error: BaseException | None,
        total_latency_ms: int | None,
        session_id: str | None,
        message_id: str | None,
        user_id: str | None,
        metadata: dict[str, Any] | None,
        partial_message_start_index: int = 0,
        max_hops_finalized: bool = False,
        persist_in_background: bool = False,
    ) -> None:
        """Store completed or partial messages after execution resources close.

        Subclasses can override this hook to customize storage. The default uses
        the supplied writer and skips storage when no writer was configured.
        Awaited writer failures propagate; background writes are best effort.
        """
        await save_messages(
            self._writer,
            self.agent_name,
            self.agent_id,
            result=result,
            agent_run=agent_run,
            state=state,
            run_error=run_error,
            total_latency_ms=total_latency_ms,
            session_id=session_id,
            message_id=message_id,
            user_id=user_id,
            metadata=metadata,
            partial_message_start_index=partial_message_start_index,
            max_hops_finalized=max_hops_finalized,
            persist_in_background=persist_in_background,
            background_persistence=self._background_persistence,
        )
