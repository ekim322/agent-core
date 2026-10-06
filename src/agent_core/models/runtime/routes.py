"""Provider/model resolution and stable identities for shared route resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import httpx2 as httpx

ConnectionProvider = Literal["openai", "anthropic", "google"]
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ModelRouting:
    VALID_PROVIDERS: tuple[ConnectionProvider, ...] = (
        "openai",
        "anthropic",
        "google",
    )

    @staticmethod
    def normalize_base_url(base_url: str | None) -> str | None:
        if base_url is None or not base_url.strip():
            return None
        try:
            url = httpx.URL(base_url.strip())
        except httpx.InvalidURL as error:
            raise ValueError("Invalid provider base_url") from error
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("Provider base_url must be an absolute HTTP(S) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "Provider base_url must not include credentials, query, or fragment"
            )
        return str(url).rstrip("/")

    @classmethod
    def _split(cls, name: str) -> tuple[ConnectionProvider | None, str]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model name must not be blank")
        prefix, separator, model = name.strip().partition(":")
        if not separator:
            return None, prefix
        if prefix.lower() not in cls.VALID_PROVIDERS:
            raise ValueError(
                f"Unknown provider prefix {prefix!r}; use one of {', '.join(cls.VALID_PROVIDERS)}"
            )
        if not model.strip():
            raise ValueError("Empty model name after provider prefix")
        return cast(ConnectionProvider, prefix.lower()), model.strip()

    @classmethod
    def infer_provider(cls, model_name: str) -> ConnectionProvider:
        explicit, model = cls._split(model_name)
        if explicit is not None:
            return explicit
        candidate = model.lower()
        for provider, hints in (
            ("openai", ("gpt", "chatgpt", "o1", "o3", "o4")),
            ("anthropic", ("claude",)),
            ("google", ("gemini",)),
        ):
            if any(hint in candidate for hint in hints):
                return cast(ConnectionProvider, provider)
        raise ValueError(
            f"Cannot infer provider for {model_name!r}; use provider:model"
        )

    @classmethod
    def unprefixed_model_name(cls, name: str) -> str:
        return cls._split(name)[1]


@dataclass(frozen=True)
class ModelRoute:
    """Provider + endpoint + optional Vertex project/region define sharing."""

    provider: ConnectionProvider
    vertex_project: str | None = None
    vertex_region: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in ModelRouting.VALID_PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.provider!r}")
        vertex = self.vertex_project is not None or self.vertex_region is not None
        if vertex:
            if (
                not self.vertex_project
                or not self.vertex_project.strip()
                or not self.vertex_region
                or not self.vertex_region.strip()
            ):
                raise ValueError("Vertex project and region must both be nonblank")
            if self.provider not in {"anthropic", "google"}:
                raise ValueError(f"Vertex is not supported for {self.provider}")
        object.__setattr__(
            self, "base_url", ModelRouting.normalize_base_url(self.base_url)
        )

    @property
    def label(self) -> str:
        parts = [self.provider]
        if self.vertex_region:
            parts.append(f"-vertex:{self.vertex_project}:{self.vertex_region}")
        if self.base_url:
            url = httpx.URL(self.base_url)
            endpoint = url.host
            if url.port is not None:
                endpoint += f":{url.port}"
            endpoint += url.path.rstrip("/")
            parts.append("@" + endpoint)
        return "".join(parts)
