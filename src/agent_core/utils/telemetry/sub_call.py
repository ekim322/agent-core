"""Persistence correlation supplied explicitly by nested-call consumers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SubCallTrace:
    """Link a nested completion to its parent through storage metadata.

    LLMClient keeps these identifiers in metadata alongside the completion's
    own session/message fields. Applications decide how to query that linkage.
    """

    parent_tool_call_id: str | None = None
    session_id: str | None = None
    message_id: str | None = None

    def to_metadata(self) -> dict[str, str | None]:
        return asdict(self)
