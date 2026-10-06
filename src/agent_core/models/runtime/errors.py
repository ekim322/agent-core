"""Explicit overload signals; causes distinguish local and remote failures."""

import math


class _Unavailable(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: float):
        if type(retry_after_seconds) not in {int, float} or not math.isfinite(
            retry_after_seconds
        ):
            raise ValueError("retry_after_seconds must be a finite number")
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        RuntimeError.__init__(self, message)


class ModelTransportUnavailable(_Unavailable):
    """Connection circuit rejected new setup; pooled connections can still work."""


class ModelRouteSaturated(_Unavailable):
    """Local queue or connection pool cannot admit work within its budget."""

    def __init__(self, message: str, *, retry_after_seconds: float = 1.0):
        super().__init__(message, retry_after_seconds=retry_after_seconds)


class ModelProviderUnavailable(_Unavailable):
    """Transient status or inactivity exhausted the pre-response retry budget."""

    def __init__(
        self, message: str, *, status_code: int | None, retry_after_seconds: float
    ):
        self.status_code = status_code
        super().__init__(message, retry_after_seconds=retry_after_seconds)
