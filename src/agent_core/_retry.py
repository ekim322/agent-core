"""Shared retry error classification and bounded backoff mechanics."""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import httpx2

from agent_core._validation import count, exception_chain

DEFAULT_MODEL_REQUEST_RETRY_ATTEMPTS = 2
MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_MODEL_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_NO_RETRY = tuple(
    getattr(client, kind)
    for client in (httpx, httpx2)
    for kind in (
        "ConnectError",
        "ConnectTimeout",
        "ProxyError",
        "PoolTimeout",
        "WriteError",
        "WriteTimeout",
    )
)
_READ_TIMEOUT = (httpx.ReadTimeout, httpx2.ReadTimeout)
_READ_FAILURE = tuple(
    getattr(client, kind)
    for client in (httpx, httpx2)
    for kind in ("ReadTimeout", "ReadError", "RemoteProtocolError")
)


def _has_nonretryable_cause(error: BaseException) -> bool:
    """Reject cancellation and connection, admission or upload failures in the chain."""
    return any(
        isinstance(cause, (*_NO_RETRY, asyncio.CancelledError, GeneratorExit))
        for cause in exception_chain(error)
    )


def _jittered_backoff_seconds(attempt: int, base: float, maximum: float) -> float:
    """Choose a bounded exponential wait with jitter in its upper half."""
    count(attempt, "retry_attempt", minimum=1)
    # Saturate before exponentiation: large retry counts cannot overflow.
    if base == 0 or maximum == 0:
        return 0.0
    saturation = math.log2(maximum) - math.log2(base)
    ceiling = maximum if attempt - 1 >= saturation else math.ldexp(base, attempt - 1)
    return random.uniform(ceiling / 2, ceiling)


def _parse_retry_after_seconds(
    raw: object, *, milliseconds: bool = False
) -> float | None:
    """Read a finite numeric delay or HTTP date; ignore unusable guidance."""
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        if milliseconds:
            return None
        try:
            at = parsedate_to_datetime(str(raw))
            at = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
            value = (at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(value):
        return None
    return max(0.0, value / 1000 if milliseconds else value)
