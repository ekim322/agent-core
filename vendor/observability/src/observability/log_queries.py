"""Read structured console logs without requiring a particular telemetry vendor.

These queries inspect retained input records, not a live metrics backend. Summary
percentiles describe only matching operation-completion logs in that input.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Iterable, Iterator

_SUMMARY_FIELDS = ('service_name', 'environment', 'operation', 'outcome')


def parse_time(value: str) -> datetime:
    """Require an explicit timezone so bug-report windows are unambiguous."""
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(
            'Use an ISO timestamp with timezone, e.g. 2026-10-03T14:00:00-04:00'
        ) from exc
    if result.tzinfo is None:
        raise ValueError('Timestamp must include Z or an explicit timezone offset')
    return result.astimezone(timezone.utc)


def read_records(lines: Iterable[str]) -> Iterator[dict]:
    """Yield JSON objects with timestamps, skipping malformed and non-object lines."""
    for line in lines:
        try:
            record = json.loads(line, parse_constant=_invalid_constant)
        except (ValueError, TypeError):
            continue
        if isinstance(record, dict) and 'timestamp' in record:
            yield record


def _invalid_constant(value: str):
    raise ValueError('Non-finite numbers are not JSON log values')


def matches(
    record: dict,
    fields: Iterable[tuple[str, str]],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> bool:
    """Match exact string fields and an optional inclusive/exclusive time window."""
    for field, expected in fields:
        if record.get(field) != expected:
            return False
    if since is not None or until is not None:
        try:
            timestamp = parse_time(str(record['timestamp']))
        except (ValueError, KeyError):
            return False
        if since is not None and timestamp < since:
            return False
        if until is not None and timestamp >= until:
            return False
    return True


def summarize(records: list[dict]) -> list[dict]:
    """Summarize valid completion durations by service, environment, operation, and outcome.

    Ignore missing, non-finite, and negative durations. Percentiles use the
    nearest rank in each group's sorted durations; results describe only the
    records supplied by the caller.
    """
    groups = _group_completion_durations(records)
    summaries = []
    for dimensions, durations in sorted(groups.items()):
        durations.sort()
        summaries.append(
            {
                **dict(zip(_SUMMARY_FIELDS, dimensions)),
                'count': len(durations),
                'mean_ms': round(sum(durations) / len(durations), 3),
                'p50_ms': durations[math.ceil(len(durations) * 0.50) - 1],
                'p95_ms': durations[math.ceil(len(durations) * 0.95) - 1],
                'max_ms': durations[-1],
            }
        )
    return summaries


def _group_completion_durations(records: list[dict]) -> dict[tuple, list[float]]:
    """Collect finite, nonnegative durations under the existing summary dimensions."""
    groups: dict[tuple, list[float]] = {}
    for record in records:
        if record.get('event_name') != 'operation.completed':
            continue
        duration = record.get('duration_ms')
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        if not math.isfinite(duration) or duration < 0:
            continue
        key = tuple(str(record.get(field, 'unknown')) for field in _SUMMARY_FIELDS)
        groups.setdefault(key, []).append(float(duration))
    return groups
