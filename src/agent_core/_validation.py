"""Internal validation of budgets and traversal of layered exception causes."""

import math
from collections.abc import Iterator


def count(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def seconds(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'non-negative'}"
        )
    return value


def exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen = set()
    while id(error) not in seen:
        seen.add(id(error))
        yield error
        cause = error.__cause__ or error.__context__
        if cause is None:
            return
        error = cause
