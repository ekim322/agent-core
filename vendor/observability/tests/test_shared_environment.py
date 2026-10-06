"""Environment configuration for console-only and remote-export applications."""

from observability import ObservabilitySettings
import pytest


def test_cloud_base_path_headers_and_intervals():
    settings = ObservabilitySettings.from_env(
        {
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'https://example.invalid/otlp/',
            'OTEL_EXPORTER_OTLP_HEADERS': 'Authorization=Basic%20abc%3D%3D,X-Scope-OrgID=test',
            'OTEL_SERVICE_NAME': 'custom-service',
            'OBSERVABILITY_ENVIRONMENT': 'production',
            'OBSERVABILITY_METRIC_INTERVAL_SECONDS': '10',
            'OBSERVABILITY_EXPORT_TIMEOUT_SECONDS': '3',
        }
    )
    assert settings.otlp_endpoint == 'https://example.invalid/otlp'
    assert settings.otlp_headers == {
        'Authorization': 'Basic abc==',
        'X-Scope-OrgID': 'test',
    }
    assert settings.service_name == 'custom-service'
    assert settings.environment == 'production'
    assert settings.metric_interval_seconds == 10
    assert settings.export_timeout_seconds == 3
    assert 'abc' not in repr(settings)


def test_standalone_or_cloud_blank_endpoint_keeps_console_only():
    assert ObservabilitySettings.from_env({}).otlp_endpoint is None
    assert (
        ObservabilitySettings.from_env(
            {'OTEL_EXPORTER_OTLP_ENDPOINT': ''}
        ).otlp_endpoint
        is None
    )


def test_header_parser_skips_blanks_and_decodes_keys_and_values():
    settings = ObservabilitySettings.from_env(
        {
            'OTEL_EXPORTER_OTLP_HEADERS': ' , X%2DName = a=b%2Cc ,Empty=,X%2DName=last%3Dvalue, ',
        }
    )
    assert settings.otlp_headers == {'X-Name': 'last=value', 'Empty': ''}


def test_header_parser_rejects_entries_without_a_separator():
    with pytest.raises(ValueError, match='OTLP headers must use key=value entries'):
        ObservabilitySettings.from_env(
            {'OTEL_EXPORTER_OTLP_HEADERS': 'Good=value,invalid'}
        )
