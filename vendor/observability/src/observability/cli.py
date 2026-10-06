"""Terminal arguments, input/output, and exit codes for saved-log investigation."""

import argparse
from collections import deque
from collections.abc import Iterable
from datetime import datetime
import json
from pathlib import Path
import sys

from .log_queries import matches, parse_time, read_records, summarize

_NAMED_FIELDS = (
    'service_name',
    'environment',
    'operation',
    'request_id',
    'trace_id',
    'event_name',
    'outcome',
    'severity',
)


def _timestamp_argument(value: str) -> datetime:
    try:
        return parse_time(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _field_argument(value: str) -> tuple[str, str]:
    name, separator, expected = value.partition('=')
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError(
            'Use --field NAME=VALUE with a nonempty field name'
        )
    return name, expected


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Filter and summarize saved JSON logs."
    )
    result.add_argument('command', choices=('logs', 'summary'))
    result.add_argument(
        '--file',
        type=Path,
        help='JSON lines file; default is stdin (not a follow stream)',
    )
    result.add_argument(
        '--since',
        type=_timestamp_argument,
        help='Inclusive ISO timestamp with timezone',
    )
    result.add_argument(
        '--until',
        type=_timestamp_argument,
        help='Exclusive ISO timestamp with timezone',
    )
    for field in _NAMED_FIELDS:
        if field != 'severity':
            result.add_argument(f'--{field.replace("_", "-")}')
    result.add_argument(
        '--field',
        action='append',
        type=_field_argument,
        default=[],
        metavar='NAME=VALUE',
        help='Exact string field match; repeat to require multiple fields',
    )
    result.add_argument(
        '--severity', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    )
    result.add_argument(
        '--limit', type=int, default=50, help='Maximum logs returned (default 50)'
    )
    result.add_argument(
        '--pretty',
        action='store_true',
        help='Indent JSON output for reading in a terminal',
    )
    result.add_argument(
        '--max-records',
        type=int,
        default=100000,
        help='Maximum matching records retained for summaries (default 100000)',
    )
    return result


def _retain_matching_records(
    lines: Iterable[str],
    fields: Iterable[tuple[str, str]],
    *,
    capacity: int,
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[dict], int]:
    """Scan finite input, retaining the final matches in their source order."""
    retained: deque[dict] = deque(maxlen=capacity)
    matched_count = 0
    for record in read_records(lines):
        if matches(record, fields, since=since, until=until):
            matched_count += 1
            retained.append(record)
    return list(retained), matched_count


def main(argv: list[str] | None = None) -> int:
    """Query a saved file or finite stdin stream and print the result as JSON."""
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if not 1 <= args.limit <= 10000 or not 1 <= args.max_records <= 1000000:
        argument_parser.error(
            'limit must be 1..10000 and max-records must be 1..1000000'
        )
    if args.since is not None and args.until is not None and args.since >= args.until:
        argument_parser.error('since must precede until')
    fields = list(args.field)
    for field in _NAMED_FIELDS:
        value = getattr(args, field)
        if value is not None:
            fields.append((field, value))
    stream = None
    try:
        stream = args.file.open(encoding='utf-8') if args.file else sys.stdin
        capacity = args.limit if args.command == 'logs' else args.max_records
        retained_records, matched_count = _retain_matching_records(
            stream,
            fields,
            capacity=capacity,
            since=args.since,
            until=args.until,
        )
        output = {
            'matched_records': matched_count,
            'truncated': matched_count > capacity,
            'source': 'retained_console_logs',
        }
        if args.command == 'logs':
            output['records'] = retained_records
        else:
            output['operations'] = summarize(retained_records)
            output['note'] = (
                'Durations summarize retained operation.completed logs only; not sampled/exported metric history.'
            )
        print(
            json.dumps(
                output,
                ensure_ascii=False,
                allow_nan=False,
                indent=2 if args.pretty else None,
            )
        )
    except (OSError, UnicodeError):
        print('Unable to read UTF-8 log input.', file=sys.stderr)
        return 1
    finally:
        if stream is not None and stream is not sys.stdin:
            stream.close()
    return 0
