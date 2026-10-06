"""Calendar and region-code validation for string-valued tool arguments."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from enum import StrEnum


class TimeFormat(StrEnum):
    DATE = "YYYY-MM-DD"
    DATE_MINUTE = "YYYY-MM-DD HH:MM"
    DATE_SECOND = "YYYY-MM-DD HH:MM:SS"


_DIRECTIVES = {
    TimeFormat.DATE: "%Y-%m-%d",
    TimeFormat.DATE_MINUTE: "%Y-%m-%d %H:%M",
    TimeFormat.DATE_SECOND: "%Y-%m-%d %H:%M:%S",
}


def _values(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Sequence):
        yield from (item for item in value if isinstance(item, str))


def _parse(value: str, param: str, fmt: TimeFormat) -> datetime:
    # Compare each character with the format template: strptime alone accepts
    # unpadded fields and some non-ASCII numerals.
    shape_valid = len(value) == len(fmt) and all(
        char in "0123456789" if marker.isalpha() else char == marker
        for char, marker in zip(value, fmt)
    )
    if shape_valid:
        try:
            return datetime.strptime(value, _DIRECTIVES[fmt])
        except ValueError:
            pass
    raise ValueError(f"{param} requires a real date in {fmt} form; received {value!r}")


def check_date(
    value: object, param: str, fmt: TimeFormat | str = TimeFormat.DATE
) -> None:
    """Check strings (including sequence items); leave omitted/typed values alone."""
    format_kind = TimeFormat(fmt)
    for candidate in _values(value):
        _parse(candidate, param, format_kind)


def check_date_window(
    start: object,
    end: object,
    *,
    start_param: str,
    end_param: str,
    max_days: int | None = None,
    fmt: TimeFormat | str = TimeFormat.DATE,
) -> None:
    """Validate endpoints, ordering, and an optional inclusive calendar-day cap."""
    if max_days is not None and (type(max_days) is not int or max_days < 1):
        raise ValueError("max_days must be a positive integer")
    format_kind = TimeFormat(fmt)
    check_date(start, start_param, format_kind)
    check_date(end, end_param, format_kind)
    if isinstance(start, str) and isinstance(end, str):
        first = _parse(start, start_param, format_kind)
        last = _parse(end, end_param, format_kind)
        if last < first:
            raise ValueError(f"{start_param}={start!r} is after {end_param}={end!r}")
        days = (last.date() - first.date()).days + 1
        if max_days is not None and days > max_days:
            raise ValueError(
                f"{start_param} to {end_param} spans {days} days; at most {max_days} days allowed"
            )


def check_country_code(value: object, param: str) -> None:
    """Check two or three ASCII letters, without claiming catalog membership."""
    for candidate in _values(value):
        if not (
            2 <= len(candidate) <= 3 and candidate.isascii() and candidate.isalpha()
        ):
            raise ValueError(
                f"{param} requires a two- or three-letter region code; received {candidate!r}"
            )


__all__ = ["TimeFormat", "check_date", "check_date_window", "check_country_code"]
