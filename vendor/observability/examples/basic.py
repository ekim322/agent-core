"""Emit correlated JSON logs: run from the checkout with uv run examples/basic.py."""

import logging

from observability import (
    Observability,
    ObservabilitySettings,
    bind_observability_context,
    observe_operation,
)


def main() -> None:
    # Explicit settings keep this demonstration console-only even if OTLP
    # environment variables are set for another application.
    runtime = Observability(ObservabilitySettings(service_name='demo'))
    runtime.start()
    try:
        with bind_observability_context(run_id='example'):
            with observe_operation('demo.work'):
                logging.getLogger(__name__).info(
                    'Demo work started',
                    extra={'event_name': 'demo.started'},
                )
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
