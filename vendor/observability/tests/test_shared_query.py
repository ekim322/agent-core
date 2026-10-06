"""Bounded local investigation works without external telemetry infrastructure."""

import json
import io

import pytest

from observability.cli import main
from observability.log_queries import parse_time, summarize


def test_summary_excludes_invalid_durations_and_keeps_groups_sorted():
    records = [
        {'event_name': 'operation.completed', 'operation': 'b', 'duration_ms': 0},
        {'event_name': 'operation.completed', 'operation': 'a', 'duration_ms': 20},
        {'event_name': 'operation.completed', 'operation': 'a', 'duration_ms': 10},
        {'event_name': 'other', 'operation': 'a', 'duration_ms': 100},
        {'event_name': 'operation.completed', 'operation': 'a'},
    ]
    records.extend(
        {'event_name': 'operation.completed', 'operation': 'a', 'duration_ms': value}
        for value in (True, '10', None, -1, float('nan'), float('inf'))
    )

    assert summarize(records) == [
        {
            'service_name': 'unknown',
            'environment': 'unknown',
            'operation': 'a',
            'outcome': 'unknown',
            'count': 2,
            'mean_ms': 15.0,
            'p50_ms': 10.0,
            'p95_ms': 20.0,
            'max_ms': 20.0,
        },
        {
            'service_name': 'unknown',
            'environment': 'unknown',
            'operation': 'b',
            'outcome': 'unknown',
            'count': 1,
            'mean_ms': 0.0,
            'p50_ms': 0.0,
            'p95_ms': 0.0,
            'max_ms': 0.0,
        },
    ]


def test_logs_scan_all_stdin_and_retain_final_matches(monkeypatch, capsys):
    rows = [
        {'timestamp': '2026-10-05T18:00:00Z', 'request_id': str(index)}
        for index in range(4)
    ]
    stream = io.StringIO('not JSON\n' + '\n'.join(map(json.dumps, rows)))
    monkeypatch.setattr('sys.stdin', stream)

    assert main(['logs', '--limit', '2']) == 0

    result = json.loads(capsys.readouterr().out)
    assert result['records'] == rows[-2:]
    assert result['matched_records'] == 4
    assert result['truncated'] is True
    assert stream.read() == ''
    assert not stream.closed


def test_query_correlates_chat_across_requests_and_timezones(tmp_path, capsys):
    rows = [
        {
            'timestamp': '2026-10-03T18:00:00Z',
            'conversation_id': 'chat-a',
            'request_id': 'a',
        },
        {
            'timestamp': '2026-10-03T18:01:00Z',
            'conversation_id': 'chat-b',
            'request_id': 'b',
        },
        {
            'timestamp': '2026-10-03T18:02:00Z',
            'conversation_id': 'chat-a',
            'request_id': 'c',
        },
    ]
    path = tmp_path / 'logs.jsonl'
    path.write_text('not JSON\n' + '\n'.join(map(json.dumps, rows)))
    assert (
        main(
            [
                'logs',
                '--file',
                str(path),
                '--field',
                'conversation_id=chat-a',
                '--since',
                '2026-10-03T14:00:00-04:00',
                '--until',
                '2026-10-03T14:03:00-04:00',
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert [r['request_id'] for r in result['records']] == ['a', 'c']
    assert result['matched_records'] == 2


def test_summary_counts_completions_only_and_discloses_truncation(tmp_path, capsys):
    rows = [
        dict(
            timestamp='2026-10-03T18:00:00Z',
            event_name='operation.completed',
            operation='chat.run',
            outcome='success',
            duration_ms=n,
        )
        for n in (10, 20, 30)
    ]
    rows.append(
        dict(
            timestamp='2026-10-03T18:00:00Z', event_name='chat.started', duration_ms=100
        )
    )
    path = tmp_path / 'logs.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    main(['summary', '--file', str(path), '--max-records', '3'])
    result = json.loads(capsys.readouterr().out)
    assert result['truncated']
    assert result['operations'][0]['count'] == 2
    assert result['operations'][0]['mean_ms'] == 25
    assert result['operations'][0]['p95_ms'] == 30


def test_requires_timezone_and_valid_window():
    with pytest.raises(ValueError):
        parse_time('2026-10-03T14:00:00')
    with pytest.raises(SystemExit):
        main(
            [
                'logs',
                '--since',
                '2026-10-03T18:00:00Z',
                '--until',
                '2026-10-03T17:00:00Z',
            ]
        )


def test_query_filters_graph_event_and_outcome(tmp_path, capsys):
    rows = [
        dict(
            timestamp='2026-10-03T18:00:00Z',
            graph_id=graph,
            event_name=event,
            outcome=outcome,
        )
        for graph, event, outcome in [
            ('graph-a', 'operation.completed', 'error'),
            ('graph-b', 'operation.completed', 'error'),
            ('graph-a', 'operation.completed', 'ok'),
            ('graph-a', 'indexing.started', 'error'),
        ]
    ]
    path = tmp_path / 'logs.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert (
        main(
            [
                'logs',
                '--file',
                str(path),
                '--field',
                'graph_id=graph-a',
                '--event-name',
                'operation.completed',
                '--outcome',
                'error',
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result['records'] == [rows[0]]


def test_query_emitted_project_and_artifact_events(tmp_path, capsys):
    import io
    import logging
    from observability import (
        Observability,
        ObservabilitySettings,
        bind_observability_context,
    )

    output = io.StringIO()
    runtime = Observability(ObservabilitySettings(), stream=output)
    runtime.start()
    try:
        for artifact_id in ('artifact-a', 'artifact-b'):
            with bind_observability_context(
                project_id='project-a', artifact_id=artifact_id
            ):
                logging.getLogger(__name__).info(
                    'Artifact operation finished',
                    extra={'event_name': 'artifact.finished'},
                )
        with bind_observability_context(
            project_id='project-b', artifact_id='artifact-a'
        ):
            logging.getLogger(__name__).info(
                'Artifact operation finished', extra={'event_name': 'artifact.finished'}
            )
    finally:
        runtime.close()
    path = tmp_path / 'logs.jsonl'
    path.write_text(output.getvalue())
    assert (
        main(
            [
                'logs',
                '--file',
                str(path),
                '--field',
                'project_id=project-a',
                '--field',
                'artifact_id=artifact-a',
                '--event-name',
                'artifact.finished',
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result['matched_records'] == 1
    assert result['records'][0]['project_id'] == 'project-a'
    assert result['records'][0]['artifact_id'] == 'artifact-a'


@pytest.mark.parametrize(
    'field',
    [
        'artifact_operation_id',
        'artifact_edit_id',
        'graph_edit_draft_id',
    ],
)
def test_query_retained_operation_ids(tmp_path, capsys, field):
    rows = [
        dict(
            timestamp='2026-10-03T18:00:00Z',
            **{field: value},
            request_id='http-request',
            event_name='operation.completed',
        )
        for value in ('first', 'second')
    ]
    path = tmp_path / 'operations.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert (
        main(
            [
                'logs',
                '--file',
                str(path),
                '--field',
                f'{field}=second',
                '--request-id',
                'http-request',
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result['matched_records'] == 1
    assert result['records'] == [rows[1]]


def test_custom_fields_require_all_matches_and_preserve_values(tmp_path, capsys):
    rows = [
        {'timestamp': '2026-10-05T18:00:00Z', 'custom_id': 'a=b', 'tag': ''},
        {'timestamp': '2026-10-05T18:00:00Z', 'custom_id': 'a=b', 'tag': 'other'},
        {'timestamp': '2026-10-05T18:00:00Z', 'tag': ''},
    ]
    path = tmp_path / 'custom.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert (
        main(
            ['logs', '--file', str(path), '--field', 'custom_id=a=b', '--field', 'tag=']
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)['records'] == [rows[0]]


@pytest.mark.parametrize('value', ['missing-separator', '=value', '  =value'])
def test_invalid_custom_field_arguments_fail_before_reading_input(value):
    with pytest.raises(SystemExit) as exc:
        main(['logs', '--field', value])
    assert exc.value.code == 2
