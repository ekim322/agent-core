"""Record and connect application diagnostics using Python logging and OpenTelemetry.

Start Observability to configure console output and optional export. Use
bind_observability_context to attach IDs to the current task, logger calls to
emit events, and observe_operation to record a timed span and completion metrics.
Binding alone stores fields in memory; it neither emits nor saves a record.
"""

from .correlation import bind_observability_context, current_trace_ids
from .operation_telemetry import ObservedOperation, observe_operation
from .runtime import Observability, get_meter, get_tracer
from .settings import ObservabilitySettings

__all__ = [
    'Observability',
    'ObservabilitySettings',
    'ObservedOperation',
    'bind_observability_context',
    'current_trace_ids',
    'get_meter',
    'get_tracer',
    'observe_operation',
]
