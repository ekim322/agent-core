"""Provider settings preserve caller overrides and reasoning translation."""
import pytest

from agent_core import ModelConfig


@pytest.mark.parametrize('provider', ['anthropic', 'google', 'custom'])
def test_common_options_preserve_falsey_values_and_extra_precedence(provider, monkeypatch):
    from agent_core.models.config import ModelRuntime
    monkeypatch.setattr(ModelRuntime, 'infer_provider', lambda name: provider)
    settings = ModelConfig(
        thinking=False, temperature=0, max_tokens=0, seed=0, stop_sequences=[],
        parallel_tool_calls=False, extra={'temperature': 0.7, 'timeout': 9},
    ).to_settings('example')
    assert settings == {
        'thinking': False, 'parallel_tool_calls': False, 'temperature': 0.7,
        'max_tokens': 0, 'seed': 0, 'stop_sequences': [], 'timeout': 9,
    }


@pytest.mark.parametrize('thinking', [False, 'high'])
def test_openai_explicit_defaults_follow_extra_and_reasoning_controls_sampling(thinking, monkeypatch):
    from agent_core.models.config import ModelRuntime
    monkeypatch.setattr(ModelRuntime, 'infer_provider', lambda name: 'openai')
    settings = ModelConfig(thinking=thinking, extra={
        'temperature': 0.5, 'top_p': 0.8, 'presence_penalty': 1,
        'frequency_penalty': 1, 'logit_bias': {'1': 2},
        'service_tier': 'priority', 'openai_store': True,
        'openai_reasoning_summary': 'detailed',
    }).to_settings('example')
    assert settings['service_tier'] == 'default'
    assert settings['openai_store'] is False
    assert settings['openai_reasoning_summary'] == 'auto'
    sampling = {'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'logit_bias'}
    if thinking is False:
        assert sampling <= settings.keys()
    else:
        assert not sampling & settings.keys()
    omitted = ModelConfig(thinking=thinking, service_tier=None).to_settings('example')
    assert 'service_tier' not in omitted
