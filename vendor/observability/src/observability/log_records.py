"""Bounded structured records shared by stdout and OTLP export.

Messages must be static operational descriptions: arbitrary free text cannot be
reliably redacted. Exception messages, payload fields and object reprs are omitted.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
import traceback
from collections.abc import Mapping
from itertools import islice

from .correlation import current_observability_context, current_trace_ids

_STANDARD = set(logging.makeLogRecord({}).__dict__) | {'message', 'asctime'}
_SENSITIVE = (
    'password',
    'secret',
    'token',
    'authorization',
    'cookie',
    'prompt',
    'payload',
    'body',
    'content',
    'headers',
    'query',
    'parameters',
    'api_key',
)


def safe_value(key: str, value: object, depth: int = 0) -> object:
    """Redact sensitive fields and bound nested values for logs and span attributes."""
    if any(part in key.lower() for part in _SENSITIVE):
        return '[REDACTED]'
    if depth > 3:
        return '[TRUNCATED]'
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1024]
    if isinstance(value, Mapping):
        return {
            str(field_name)[:100]: safe_value(str(field_name), field_value, depth + 1)
            for field_name, field_value in islice(value.items(), 30)
        }
    if isinstance(value, (list, tuple)):
        return [safe_value(key, item, depth + 1) for item in value[:30]]
    return f'<{type(value).__name__}>'


def exception_details(exc: BaseException) -> dict[str, object]:
    """Keep the exception type and recent code locations, omitting text and locals."""
    frames = traceback.extract_tb(exc.__traceback__)
    return {
        'error_type': type(exc).__name__,
        'error_frames': [
            {
                'file': Path(frame.filename).name,
                'function': frame.name,
                'line': frame.lineno,
            }
            for frame in frames[-12:]
        ],
    }


class JsonFormatter(logging.Formatter):
    """Turn an emitted Python log into JSON with request/chat and trace IDs.

    Logging handlers call this in the emitting task, where bound fields and the
    active OpenTelemetry span are available. Handlers write or export the result;
    the formatter assembles it without saving anything itself.
    """

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def record_fields(self, record: logging.LogRecord) -> dict[str, object]:
        """Combine log extras, bound fields, and active span IDs for both outputs.

        Bound fields override same-named extras; standard fields and active
        trace/span IDs take precedence afterward. Redaction applies to structured
        fields, so callers must keep secrets out of the free-text message.
        """
        log_extra_fields = {
            key: safe_value(key, value)
            for key, value in record.__dict__.items()
            if key not in _STANDARD and not key.startswith('_')
        }
        bound_fields = {
            key: safe_value(key, value)
            for key, value in current_observability_context().items()
        }
        fields = {**log_extra_fields, **bound_fields}
        fields.update(
            {
                'timestamp': datetime.fromtimestamp(
                    record.created, timezone.utc
                ).isoformat(),
                'severity': record.levelname,
                'service_name': self.settings.service_name,
                'environment': self.settings.environment,
                'service_version': self.settings.service_version,
                'logger': record.name,
                'message': record.getMessage()[:2048],
                **current_trace_ids(),
            }
        )
        if record.exc_info and record.exc_info[1]:
            fields.update(exception_details(record.exc_info[1]))
        return fields

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            self.record_fields(record), ensure_ascii=False, allow_nan=False
        )
