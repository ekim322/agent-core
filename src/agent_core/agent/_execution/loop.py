"""Drive model/tool phases and recover interrupted open responses."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from typing import Any, Generic

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.run import AgentRun

from agent_core.streaming import StreamEvent
from agent_core.agent._execution.recovery import (
    ModelRequestRetryPolicy,
    ModelRequestRetryPlan,
    ModelRequestStreamState,
    InterruptedResponse,
    RetryableModelRequestError,
)
from agent_core.models.runtime import ModelProviderUnavailable
from agent_core.utils.telemetry.run_state import RunState
from agent_core.agent.request import Deps
from agent_core.agent._execution.invocation import Invocation

logger = logging.getLogger("agent_core._execution.phases")


class PhaseExecutor(Generic[Deps]):
    """Drive model and tool nodes, recovering interrupted answers within each phase.

    The tool phase counts model turns, including interrupted attempts. Reaching
    the cap closes its SDK run before entering a final phase with tools disabled.
    Both phases retain the invocation's history, timing and SDK usage counter.
    The supplied node hook owns event conversion and node-stream cleanup.
    """

    def __init__(
        self,
        agent: Agent[Deps, Any],
        agent_name: str,
        retry_policy: ModelRequestRetryPolicy,
        finalize_prompt: str,
        handle_node: Callable[
            [Any, AgentRun[Deps, Any], RunState], AsyncGenerator[StreamEvent, None]
        ],
    ) -> None:
        self.agent = agent
        self.agent_name = agent_name
        self.retry_policy = retry_policy
        self.finalize_prompt = finalize_prompt
        self.handle_node = handle_node

    async def execute(
        self,
        invocation: Invocation[Deps],
        initial_options: dict[str, Any],
        limit: int | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Drive a phase with an optional model-turn cap.

        A phase owns its retry budget. Re-entering a Pydantic run resumes from
        the recovery policy's history; a completed model node resets the budget.
        Recovery retains the phase's model-turn count. The final phase has its
        own recovery budget and shares invocation timing, history and usage.
        The caller disables tools for the final phase; explicit SDK usage limits
        still apply when the model-turn cap is None.
        """
        options = initial_options
        recovery_attempts = 0
        model_turns = 0
        while True:
            try:
                finalization: tuple[AgentRun[Deps, Any], Any] | None = None
                async with self.agent.iter(**options) as run:
                    invocation.run = run
                    async for node in run:
                        is_model_request = Agent.is_model_request_node(node)
                        if (
                            is_model_request
                            and limit is not None
                            and model_turns >= limit
                        ):
                            finalization = (run, node)
                            break
                        if is_model_request:
                            model_turns += 1
                        async with aclosing(
                            self._observe_node(node, run, invocation.state)
                        ) as events:
                            async for event in events:
                                yield event
                        if is_model_request:
                            recovery_attempts = 0
                    else:
                        invocation.result = run.result
                # Finalization starts another SDK run. Release this run's model,
                # toolset and concurrency scopes before acquiring the next ones.
                if finalization is not None:
                    capped_run, pending_node = finalization
                    async with aclosing(
                        self._finalize(
                            invocation, capped_run, pending_node, initial_options, limit
                        )
                    ) as events:
                        async for event in events:
                            yield event
                return
            except RetryableModelRequestError as interrupted:
                recovery_attempts += 1
                plan = self._recovery_plan(
                    interrupted.interruption, recovery_attempts, initial_options
                )
                if interrupted.interruption.reset_stream:
                    yield StreamEvent.run_status(
                        "model_stream_reset",
                        attempt=recovery_attempts,
                        reason="mid_stream_failure",
                    )
                await asyncio.sleep(plan.delay_seconds)
                invocation.text_prefix += plan.prefix_delta
                options = plan.iter_kwargs

    async def _finalize(
        self,
        invocation: Invocation[Deps],
        capped_run: AgentRun[Deps, Any],
        pending_node: Any,
        options: dict[str, Any],
        limit: int,
    ) -> AsyncGenerator[StreamEvent, None]:
        invocation.finalized = True
        history = list(capped_run.all_messages())
        pending = getattr(pending_node, "request", None)
        # A tool return may exist in the next node before the SDK appends it.
        # Retain it even if entering the final run fails.
        if isinstance(pending, ModelRequest) and (
            not history or history[-1] != pending
        ):
            history.append(pending)
            invocation.state.unrecorded_messages.append(pending)
        logger.warning(
            "Model-turn budget exhausted; starting final response with tools disabled",
            extra={
                "event_name": "agent.max_hops_finalizing",
                "agent_name": self.agent_name,
                "max_hops": limit,
            },
        )
        yield StreamEvent.run_status("max_hops_finalizing", max_hops=limit)
        final_options = {
            **options,
            "user_prompt": self.finalize_prompt,
            "message_history": history,
            "instructions": options.get("instructions"),
        }
        with self.agent.override(tools=[], toolsets=[]):
            async with aclosing(
                self.execute(invocation, final_options, None)
            ) as events:
                async for event in events:
                    yield event

    def _recovery_plan(
        self,
        interrupted: InterruptedResponse,
        attempt: int,
        options: dict[str, Any],
    ) -> ModelRequestRetryPlan:
        policy = self.retry_policy
        if not policy.can_retry(attempt - 1):
            raise interrupted.error from interrupted.error
        plan = policy.plan_retry(interrupted, attempt, options)
        logger.warning(
            "Applying recovery plan after model response interruption",
            extra={
                "event_name": "model.stream_retry",
                "agent_name": self.agent_name,
                "retry_mode": plan.mode,
                "retry_attempt": attempt,
                "retry_limit": policy.max_attempts,
                "delay_seconds": plan.delay_seconds,
                "error_type": type(interrupted.error).__name__,
            },
        )
        return plan

    async def _observe_node(
        self,
        node: Any,
        run: AgentRun[Deps, Any],
        state: RunState,
    ) -> AsyncGenerator[StreamEvent, None]:
        emitted = ModelRequestStreamState()
        state.begin_node_stream()
        try:
            with state.bind_to_run(run.ctx.state.run_id):
                async with aclosing(self.handle_node(node, run, state)) as events:
                    async for event in events:
                        if Agent.is_model_request_node(node):
                            emitted.observe(event)
                        yield event
        except Exception as failure:
            policy = self.retry_policy
            can_recover = (
                Agent.is_model_request_node(node)
                and not isinstance(failure, ModelProviderUnavailable)
                and state.model_stream_opened
                and policy.is_retryable_error(failure)
            )
            if not can_recover or (
                emitted.emitted_visible_output and not emitted.partial_response_text
            ):
                raise
            interruption = InterruptedResponse(
                error=failure,
                messages=tuple(run.all_messages()),
                visible_text=emitted.partial_response_text,
            )
            raise RetryableModelRequestError(interruption) from failure
