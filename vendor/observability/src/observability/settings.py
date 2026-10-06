"""Validated, vendor-independent telemetry configuration."""

from collections.abc import Mapping
import os
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field, field_validator


class ObservabilitySettings(BaseModel):
    """Choose service identity, log verbosity, and optional HTTP telemetry export."""

    service_name: str = 'application'
    environment: str = 'development'
    service_version: str = 'unknown'
    log_level: str = 'INFO'
    otlp_endpoint: str | None = None
    otlp_headers: dict[str, str] = Field(default_factory=dict, repr=False)
    export_timeout_seconds: float = Field(default=5, gt=0, le=30)
    metric_interval_seconds: float = Field(default=30, ge=1)

    @field_validator('log_level')
    @classmethod
    def valid_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}:
            raise ValueError('Unsupported observability log level')
        return value

    @field_validator('otlp_endpoint')
    @classmethod
    def valid_endpoint(cls, value: str | None) -> str | None:
        if not value:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {'http', 'https'}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                'OTLP endpoint must be an HTTP(S) base URL without credentials or query'
            )
        return value.rstrip('/')

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> 'ObservabilitySettings':
        """Read and validate settings from the supplied mapping or process environment."""
        env = os.environ if environ is None else environ
        headers = _parse_otlp_headers(env.get('OTEL_EXPORTER_OTLP_HEADERS', ''))
        return cls(
            service_name=env.get('OTEL_SERVICE_NAME', 'application'),
            environment=env.get('OBSERVABILITY_ENVIRONMENT', 'development'),
            service_version=env.get('OBSERVABILITY_SERVICE_VERSION', 'unknown'),
            log_level=env.get('OBSERVABILITY_LOG_LEVEL', 'INFO'),
            otlp_endpoint=env.get('OTEL_EXPORTER_OTLP_ENDPOINT') or None,
            otlp_headers=headers,
            export_timeout_seconds=env.get('OBSERVABILITY_EXPORT_TIMEOUT_SECONDS', '5'),
            metric_interval_seconds=env.get(
                'OBSERVABILITY_METRIC_INTERVAL_SECONDS', '30'
            ),
        )


def _parse_otlp_headers(encoded_headers: str) -> dict[str, str]:
    """Decode comma-separated header pairs, preserving '=' inside their values."""
    headers = {}
    for pair in encoded_headers.split(','):
        if pair.strip():
            key, separator, header_value = pair.partition('=')
            if not separator:
                raise ValueError('OTLP headers must use key=value entries')
            headers[unquote(key.strip())] = unquote(header_value.strip())
    return headers
