"""Construct SDK providers on runtime-owned transports with SDK retries off."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

import httpx2 as httpx
from pydantic_ai.models import Model
from pydantic_ai.providers import Provider

from agent_core.models.runtime.routes import (
    DEFAULT_OPENAI_BASE_URL,
    ModelRoute,
)


class ModelProviderFactory:
    """Bind provider SDKs to shared clients; the runtime closes those clients."""

    @staticmethod
    def openai_base_url_from_env() -> str:
        return os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL

    @staticmethod
    def build_provider(
        route: ModelRoute, http_client: httpx.AsyncClient
    ) -> Provider[Any]:
        options = dict(
            base_url=route.base_url,
            http_client=http_client,
            timeout=http_client.timeout,
            max_retries=0,
        )
        if route.provider == "openai":
            from openai import AsyncOpenAI
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIProvider(openai_client=AsyncOpenAI(**options))
        if route.provider == "anthropic":
            from anthropic import AsyncAnthropic, AsyncAnthropicVertex
            from pydantic_ai.providers.anthropic import AnthropicProvider

            if route.vertex_project is None:
                sdk = AsyncAnthropic(**options)
            else:
                sdk = AsyncAnthropicVertex(
                    project_id=route.vertex_project,
                    region=route.vertex_region,
                    **options,
                )
            return AnthropicProvider(anthropic_client=sdk)
        if route.provider == "google":
            if route.vertex_project is None:
                from pydantic_ai.providers.google import GoogleProvider

                return GoogleProvider(base_url=route.base_url, http_client=http_client)
            from pydantic_ai.providers.google_cloud import GoogleCloudProvider

            return GoogleCloudProvider(
                project=route.vertex_project,
                location=route.vertex_region,
                base_url=route.base_url,
                http_client=http_client,
            )
        raise ValueError(f"No provider adapter for {route.provider!r}")

    @staticmethod
    def build_base_model(
        model_name: str, route: ModelRoute, provider: Provider[Any]
    ) -> Model:
        families = {
            "openai": ("openai", "OpenAIResponsesModel"),
            "anthropic": ("anthropic", "AnthropicModel"),
            "google": ("google", "GoogleModel"),
        }
        if route.provider not in families:
            raise ValueError(f"No model adapter for {route.provider!r}")
        module, class_name = families[route.provider]
        model_class = getattr(import_module("pydantic_ai.models." + module), class_name)
        return model_class(model_name, provider=provider)
