"""Default capacity across models on a shared provider route, without network calls."""
import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel

from agent_core.models.runtime.providers import ModelProviderFactory
from agent_core.models.runtime.runtime import ModelRuntime
from agent_core.models.runtime.settings import ModelRuntimeSettings


@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'google'])
def test_32_shared_model_calls_queue_and_release_on_cancel(monkeypatch, provider):
    async def check():
        active = 0
        peak = 0
        entered = 0
        full = asyncio.Event()
        resumed = asyncio.Event()
        release = asyncio.Event()
        async def respond(messages, info):
            nonlocal active, peak, entered
            active += 1
            entered += 1
            peak = max(peak, active)
            if entered == 32:
                full.set()
            if entered == 33:
                resumed.set()
            try:
                await release.wait()
                return ModelResponse(parts=[TextPart('done')])
            finally:
                active -= 1
        monkeypatch.setattr(ModelProviderFactory, 'build_provider', staticmethod(lambda route, client: object()))
        monkeypatch.setattr(ModelProviderFactory, 'build_base_model', staticmethod(lambda *args: FunctionModel(respond)))
        async with ModelRuntime(settings=ModelRuntimeSettings()) as runtime:
            first = ModelRuntime.make_model(f'{provider}:first', runtime=runtime, base_url='https://provider.test/v1')
            second = ModelRuntime.make_model(f'{provider}:second', runtime=runtime, base_url='https://provider.test/v1')
            tasks = [asyncio.create_task((first if i % 2 else second).request([], None, ModelRequestParameters())) for i in range(32)]
            try:
                await asyncio.wait_for(full.wait(), 5)
                extra = asyncio.create_task(second.request([], None, ModelRequestParameters()))
                tasks.append(extra)
                await asyncio.sleep(0)  # let the 33rd request reach the shared limiter
                assert entered == 32 and not extra.done()
                tasks[0].cancel()
                await asyncio.gather(tasks[0], return_exceptions=True)
                await asyncio.wait_for(resumed.wait(), 5)
                assert peak == 32
                release.set()
                await asyncio.gather(*tasks[1:])
                assert active == 0
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(check())


def test_environment_overrides_preserve_provider_pool_capacity(monkeypatch):
    monkeypatch.setenv('LLM_MODEL_MAX_IN_FLIGHT', '32')
    monkeypatch.setenv('LLM_MODEL_ANTHROPIC_MAX_IN_FLIGHT', '48')
    settings = ModelRuntimeSettings.from_env()
    assert settings.anthropic.max_in_flight == 48
    for provider in ['openai', 'google']:
        limits = settings.for_provider(provider)
        assert limits.max_in_flight == 32
        assert limits.http2_only or limits.max_connections >= 32
