"""Mark recoverable tool failures without putting error text in telemetry."""

from collections.abc import Mapping

TOOL_ERROR_METADATA_KEY = "tool_error"


def tool_error_metadata() -> dict[str, bool]:
    """Return a fresh marker so callers may attach other metadata safely."""
    return dict(tool_error=True)


def has_tool_error_metadata(metadata: object) -> bool:
    """Only the explicit boolean marker indicates a failed execution."""
    if not isinstance(metadata, Mapping):
        return False
    return metadata.get(TOOL_ERROR_METADATA_KEY, False) is True
