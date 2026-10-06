"""Retry policy for provider failures before a response is exposed."""

from __future__ import annotations

from dataclasses import dataclass
import random

from pydantic_ai.exceptions import ModelHTTPError

from agent_core._retry import (
    MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS,
    RETRYABLE_MODEL_HTTP_STATUS_CODES,
    _READ_TIMEOUT,
    _jittered_backoff_seconds,
    _has_nonretryable_cause,
    _parse_retry_after_seconds,
)
from agent_core._validation import count, exception_chain, seconds


@dataclass(frozen=True)
class ProviderRequestRetryPolicy:
    """One shared retry budget for transient statuses and pre-response silence."""

    max_retries: int = 2
    base_delay_seconds: float = MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS
    max_delay_seconds: float = 8.0
    max_retry_after_seconds: float = 30.0
    retry_after_jitter_ratio: float = 0.1
    retryable_http_status_codes: frozenset[int] = RETRYABLE_MODEL_HTTP_STATUS_CODES

    def __post_init__(self) -> None:
        count(self.max_retries, "max_retries")
        for name in (
            "base_delay_seconds",
            "max_delay_seconds",
            "max_retry_after_seconds",
            "retry_after_jitter_ratio",
        ):
            seconds(getattr(self, name), name)
        codes = frozenset(self.retryable_http_status_codes)
        if any(type(code) is not int or not 100 <= code <= 599 for code in codes):
            raise ValueError(
                "retryable_http_status_codes must contain HTTP status integers"
            )
        object.__setattr__(self, "retryable_http_status_codes", codes)

    @staticmethod
    def status_code(exc: BaseException) -> int | None:
        from google.genai.errors import APIError

        for cause in exception_chain(exc):
            if isinstance(cause, ModelHTTPError):
                return cause.status_code
            if isinstance(cause, APIError):
                try:
                    return int(cause.code)
                except (TypeError, ValueError):
                    return None
        return None

    def retry_after_seconds(self, exc: BaseException) -> float | None:
        for cause in exception_chain(exc):
            headers = getattr(getattr(cause, "response", None), "headers", None)
            if headers is not None:
                for key, milliseconds in (
                    ("retry-after-ms", True),
                    ("retry-after", False),
                ):
                    delay = _parse_retry_after_seconds(
                        headers.get(key), milliseconds=milliseconds
                    )
                    if delay is not None:
                        return delay
        return None

    def is_retryable_status(self, exc: BaseException) -> bool:
        return (
            not _has_nonretryable_cause(exc)
            and self.status_code(exc) in self.retryable_http_status_codes
        )

    def is_read_timeout(self, exc: BaseException) -> bool:
        return not _has_nonretryable_cause(exc) and any(
            isinstance(cause, _READ_TIMEOUT) for cause in exception_chain(exc)
        )

    def can_retry(self, exc: BaseException, retries_used: int) -> bool:
        count(retries_used, "retries_used")
        if retries_used >= self.max_retries or _has_nonretryable_cause(exc):
            return False
        eligible = self.is_read_timeout(exc) or self.is_retryable_status(exc)
        guidance = self.retry_after_seconds(exc)
        return eligible and (
            guidance is None or guidance <= self.max_retry_after_seconds
        )

    def delay_seconds(self, exc: BaseException, retry_attempt: int) -> float:
        count(retry_attempt, "retry_attempt", minimum=1)
        guidance = self.retry_after_seconds(exc)
        if guidance is None:
            return _jittered_backoff_seconds(
                retry_attempt, self.base_delay_seconds, self.max_delay_seconds
            )
        if guidance > self.max_retry_after_seconds:
            raise ValueError("Retry-After exceeds the configured wait budget")
        jitter = min(
            max(0.1, guidance * self.retry_after_jitter_ratio),
            self.max_retry_after_seconds - guidance,
        )
        return guidance + random.uniform(0, jitter)
