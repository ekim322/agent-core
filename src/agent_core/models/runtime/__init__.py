"""Runtime-owned provider connections, admission and overload signals.

Own an isolated lifetime with async with ModelRuntime(), or call
ModelRuntime.close_default() after process-default model work has stopped.
"""

from agent_core.models.runtime.errors import (
    ModelProviderUnavailable,
    ModelRouteSaturated,
    ModelTransportUnavailable,
)
from agent_core.models.runtime.managed import ManagedModel
from agent_core.models.runtime.routes import (
    DEFAULT_OPENAI_BASE_URL,
    ConnectionProvider,
    ModelRoute,
)
from agent_core.models.runtime.runtime import ModelRuntime
from agent_core.models.runtime.settings import (
    ModelProviderSettings,
    ModelRuntimeSettings,
)

__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "ConnectionProvider",
    "ManagedModel",
    "ModelProviderSettings",
    "ModelProviderUnavailable",
    "ModelRoute",
    "ModelRouteSaturated",
    "ModelRuntime",
    "ModelRuntimeSettings",
    "ModelTransportUnavailable",
]
