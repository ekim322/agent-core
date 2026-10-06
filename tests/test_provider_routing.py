"""Built-in routes reject removed providers before allocating transports."""

import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from agent_core import BaseAgent, LLMClient, ModelConfig
from agent_core.models.runtime import ModelRuntime, ModelRoute
from agent_core.models.runtime.routes import ModelRouting


@pytest.mark.parametrize(
    "name",
    [
        "xai:grok-4",
        "fireworks:kimi-k3",
        "grok-4",
        "kimi-k3",
        "deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k3",
    ],
)
def test_removed_model_names_fail_before_provider_construction(name, monkeypatch):
    def unexpected_resources(*args):
        pytest.fail("unsupported models must not allocate provider resources")

    monkeypatch.setattr(ModelRuntime, "_resources_for", unexpected_resources)
    for constructor in (BaseAgent, LLMClient, ModelRuntime.make_model):
        with pytest.raises(
            ValueError, match="Unknown provider prefix|Cannot infer provider"
        ):
            constructor(name)
    with pytest.raises(
        ValueError, match="Unknown provider prefix|Cannot infer provider"
    ):
        ModelConfig(thinking=False).to_settings(name)


@pytest.mark.parametrize("provider", ["xai", "fireworks"])
def test_removed_provider_routes_are_invalid(provider):
    with pytest.raises(ValueError, match="Unsupported provider"):
        ModelRoute(provider)


@pytest.mark.parametrize(
    "name,provider",
    [
        ("gpt-5", "openai"),
        ("claude-sonnet-4", "anthropic"),
        ("gemini-2.5-pro", "google"),
        ("openai:custom", "openai"),
        ("anthropic:custom", "anthropic"),
        ("google:custom", "google"),
    ],
)
def test_remaining_provider_names_route_normally(name, provider):
    assert ModelRouting.infer_provider(name) == provider


@pytest.mark.parametrize("name", ["xai:grok-4", "fireworks:kimi-k3"])
def test_per_call_overrides_reject_removed_providers(name):
    async def exercise():
        model = TestModel()
        agent = BaseAgent(model)
        client = LLMClient(model)
        with pytest.raises(ValueError, match="Unknown provider prefix"):
            await agent.run("hello", model=name)
        with pytest.raises(ValueError, match="Unknown provider prefix"):
            await client.run([], model=name)

    asyncio.run(exercise())


def test_caller_supplied_model_objects_remain_usable():
    model = TestModel()
    assert BaseAgent(model).agent.model is model
    assert LLMClient(model).model is model
