"""Plan replay or continuation of an interrupted agent response."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from agent_core._retry import (
    DEFAULT_MODEL_REQUEST_RETRY_ATTEMPTS,
    MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS,
    _READ_FAILURE,
    _jittered_backoff_seconds,
    _has_nonretryable_cause,
)
from agent_core._validation import count, exception_chain, seconds
from agent_core.streaming import EventType, StreamEvent
from agent_core.agent._execution.prompts import CONTINUE_PARTIAL_ASSISTANT_PROMPT


@dataclass
class ModelRequestStreamState:
    stream_opened: bool = False
    emitted_visible_output: bool = False
    text_parts: list[str] = field(default_factory=list)

    def mark_opened(self) -> None:
        self.stream_opened = True

    def observe(self, event: StreamEvent) -> None:
        if event.type in {EventType.TEXT_DELTA, EventType.REASONING_DELTA}:
            delta = str(event.data.get("delta", ""))
            if delta:
                self.emitted_visible_output = True
                if event.type == EventType.TEXT_DELTA:
                    self.text_parts.append(delta)

    @property
    def partial_response_text(self) -> str:
        return "".join(self.text_parts)


@dataclass(frozen=True)
class InterruptedResponse:
    """Capture the failure and SDK history before control leaves the stream.

    The tuple freezes history membership; message objects retain their SDK
    identity. Building continuation inputs never mutates this captured history.
    """

    error: BaseException
    messages: tuple[ModelMessage, ...]
    visible_text: str = ""
    reset_stream: bool = False

    def recovery_history(self, prompt: str) -> list[ModelMessage]:
        history = list(self.messages)
        if not self.visible_text:
            return history
        last_message = self.messages[-1] if self.messages else None
        run_id = getattr(last_message, "run_id", None)
        conversation_id = getattr(last_message, "conversation_id", None)
        if not self._text_is_recorded(last_message):
            history.append(
                ModelResponse(
                    parts=[TextPart(self.visible_text)],
                    state="interrupted",
                    run_id=run_id,
                    conversation_id=conversation_id,
                )
            )
        history.append(
            ModelRequest(
                parts=[UserPromptPart(prompt)],
                run_id=run_id,
                conversation_id=conversation_id,
            )
        )
        return history

    def _text_is_recorded(self, message: ModelMessage | None) -> bool:
        if not isinstance(message, ModelResponse):
            return False
        text = "".join(
            part.content for part in message.parts if isinstance(part, TextPart)
        )
        return text.strip() == self.visible_text.strip()


class RetryableModelRequestError(Exception):
    """Transport a captured interruption to the agent's recovery loop."""

    def __init__(self, interruption: InterruptedResponse) -> None:
        self.interruption = interruption
        super().__init__(str(interruption.error))


@dataclass(frozen=True)
class ModelRequestRetryPlan:
    iter_kwargs: dict[str, Any]
    prefix_delta: str
    delay_seconds: float
    mode: str


@dataclass(frozen=True)
class ModelRequestRetryPolicy:
    """Plan recovery from established-response read failures.

    The agent verifies stream visibility before requesting a plan. Text can be
    continued; reasoning-only output has no safe prefix and must not be replayed.
    History is copied and prior tool calls/results remain intact.
    """

    max_attempts: int = DEFAULT_MODEL_REQUEST_RETRY_ATTEMPTS
    base_delay_seconds: float = MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS
    max_delay_seconds: float = 8.0
    continuation_prompt: str = CONTINUE_PARTIAL_ASSISTANT_PROMPT

    def __post_init__(self) -> None:
        count(self.max_attempts, "max_attempts")
        seconds(self.base_delay_seconds, "base_delay_seconds")
        seconds(self.max_delay_seconds, "max_delay_seconds")
        if not self.continuation_prompt.strip():
            raise ValueError("continuation_prompt must not be blank")

    def can_retry(self, attempts_used: int) -> bool:
        return count(attempts_used, "attempts_used") < self.max_attempts

    def is_retryable_error(self, exc: BaseException) -> bool:
        return not _has_nonretryable_cause(exc) and any(
            isinstance(cause, _READ_FAILURE) for cause in exception_chain(exc)
        )

    def _delay_seconds(self, attempt: int) -> float:
        return _jittered_backoff_seconds(
            attempt, self.base_delay_seconds, self.max_delay_seconds
        )

    def plan_retry(
        self,
        interruption: InterruptedResponse,
        retry_attempt: int,
        iter_kwargs: dict[str, Any],
    ) -> ModelRequestRetryPlan:
        prefix = interruption.visible_text
        inputs = {
            **iter_kwargs,
            "user_prompt": None,
            "message_history": interruption.recovery_history(self.continuation_prompt),
        }
        return ModelRequestRetryPlan(
            inputs,
            prefix,
            self._delay_seconds(retry_attempt),
            "continuing partial response" if prefix else "replaying request",
        )
