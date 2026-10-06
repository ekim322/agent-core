"""Immutable, validated route capacity and transport/retry budgets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from agent_core._validation import count, seconds
from agent_core.models.runtime.routes import ConnectionProvider, ModelRouting


@dataclass(frozen=True)
class ModelProviderSettings:
    """Outer admission owns backpressure; pools must accommodate admitted work.

    HTTP/2-only routes use one-connection shards with explicit stream slots.
    Other routes need one possible HTTP/1.1 connection per admitted operation.
    """

    max_in_flight: int
    max_queued: int = 128
    queue_timeout_seconds: float = 10.0
    max_connections: int = 1
    max_keepalive_connections: int = 1
    http2_streams_per_connection: int | None = None
    http2_only: bool = False

    def __post_init__(self) -> None:
        count(self.max_in_flight, "max_in_flight", minimum=1)
        count(self.max_queued, "max_queued")
        count(self.max_connections, "max_connections", minimum=1)
        count(self.max_keepalive_connections, "max_keepalive_connections")
        seconds(self.queue_timeout_seconds, "queue_timeout_seconds", positive=True)
        if type(self.http2_only) is not bool:
            raise ValueError("http2_only must be a boolean")
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError("max_keepalive_connections cannot exceed max_connections")
        if self.http2_only:
            count(
                self.http2_streams_per_connection,
                "http2_streams_per_connection",
                minimum=1,
            )
            if self.shard_count > min(
                self.max_connections, self.max_keepalive_connections
            ):
                raise ValueError(
                    "connection and keepalive limits must retain every required HTTP/2 shard"
                )
        else:
            if self.http2_streams_per_connection is not None:
                raise ValueError("http2_streams_per_connection requires http2_only")
            if self.max_connections < self.max_in_flight:
                raise ValueError(
                    "HTTP/1.1 pool must have max_connections >= max_in_flight"
                )

    @property
    def shard_count(self) -> int:
        if self.http2_only:
            streams = self.http2_streams_per_connection
            return (self.max_in_flight + streams - 1) // streams
        return 1


def _capacity(provider: ConnectionProvider) -> ModelProviderSettings:
    if provider in {"openai", "anthropic"}:
        connections = 13 if provider == "openai" else 4
        return ModelProviderSettings(
            32,
            max_connections=connections,
            max_keepalive_connections=connections,
            http2_streams_per_connection=16,
            http2_only=True,
        )
    return ModelProviderSettings(
        32, max_queued=64, max_connections=32, max_keepalive_connections=32
    )


@dataclass(frozen=True)
class ModelRuntimeSettings:
    """Finite process-local budgets. Environment overrides load explicitly.

    Read timeouts measure response inactivity, not total generation duration.
    Zero disables retries/circuits or requests immediate transport timeouts;
    queue waits must be positive. Provider SDK retries remain disabled.
    """

    openai: ModelProviderSettings = field(default_factory=lambda: _capacity("openai"))
    anthropic: ModelProviderSettings = field(
        default_factory=lambda: _capacity("anthropic")
    )
    google: ModelProviderSettings = field(default_factory=lambda: _capacity("google"))
    keepalive_expiry_seconds: float = 90.0
    connect_timeout_seconds: float = 20.0
    read_timeout_seconds: float = 30.0
    write_timeout_seconds: float = 60.0
    pool_timeout_seconds: float = 10.0
    status_max_retries: int = 2
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 8.0
    retry_after_max_seconds: float = 30.0
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 15.0

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name in ModelRouting.VALID_PROVIDERS:
                if not isinstance(value, ModelProviderSettings):
                    raise ValueError(f"{spec.name} must be ModelProviderSettings")
            elif spec.name in {"status_max_retries", "circuit_failure_threshold"}:
                count(value, spec.name)
            else:
                seconds(value, spec.name)

    def for_provider(self, provider: ConnectionProvider) -> ModelProviderSettings:
        if provider not in ModelRouting.VALID_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider!r}")
        return getattr(self, provider)

    @staticmethod
    def _number(name: str, default, *, integer: bool):
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw) if integer else float(raw)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"Invalid numeric environment setting {name}") from error
        return count(value, name) if integer else seconds(value, name)

    @classmethod
    def _provider_from_env(
        cls, provider: ConnectionProvider, defaults: ModelProviderSettings
    ) -> ModelProviderSettings:
        prefix = "LLM_MODEL_" + provider.upper() + "_"

        def read(key, fallback, *, integer=True):
            specific = prefix + key
            selected = specific if specific in os.environ else "LLM_MODEL_" + key
            return cls._number(selected, fallback, integer=integer)

        active = read("MAX_IN_FLIGHT", defaults.max_in_flight)
        connections = read("MAX_CONNECTIONS", defaults.max_connections)
        if not defaults.http2_only and prefix + "MAX_CONNECTIONS" not in os.environ:
            connections = (
                max(active, connections)
                if "LLM_MODEL_MAX_CONNECTIONS" in os.environ
                else active
            )
        if prefix + "MAX_KEEPALIVE_CONNECTIONS" in os.environ:
            keepalive = read("MAX_KEEPALIVE_CONNECTIONS", connections)
        elif defaults.http2_only:
            keepalive = cls._number(
                "LLM_MODEL_MAX_KEEPALIVE_CONNECTIONS", connections, integer=True
            )
        else:
            keepalive = connections
        streams = (
            read("HTTP2_STREAMS_PER_CONNECTION", defaults.http2_streams_per_connection)
            if defaults.http2_only
            else None
        )
        return ModelProviderSettings(
            active,
            max_queued=read("MAX_QUEUED", defaults.max_queued),
            queue_timeout_seconds=read(
                "QUEUE_TIMEOUT_SECONDS", defaults.queue_timeout_seconds, integer=False
            ),
            max_connections=connections,
            max_keepalive_connections=keepalive,
            http2_streams_per_connection=streams,
            http2_only=defaults.http2_only,
        )

    @classmethod
    def from_env(cls) -> ModelRuntimeSettings:
        defaults = cls()
        resolved = {}
        for spec in fields(defaults):
            value = getattr(defaults, spec.name)
            if spec.name in ModelRouting.VALID_PROVIDERS:
                resolved[spec.name] = cls._provider_from_env(spec.name, value)
            else:
                resolved[spec.name] = cls._number(
                    "LLM_MODEL_" + spec.name.upper(), value, integer=type(value) is int
                )
        return cls(**resolved)
