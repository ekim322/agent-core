"""Remove credential values from captured payloads and independently buffered text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_REDACTED = "[redacted]"
_CREDENTIAL_FIELDS = (
    "password",
    "authorization",
    "cookie",
    "secret",
    "credential",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "bearertoken",
    "privatekey",
)


@dataclass
class _SecretPrefix:
    children: dict[str, _SecretPrefix] = field(default_factory=dict)
    complete: bool = False


class TraceRedactor:
    """Match original text once and keep literal credentials out of payloads.

    The prefix tree shares work between credentials with common prefixes. Each
    text stream owns its unfinished suffix, so parallel spans cannot combine
    unrelated chunks. Replacement markers never enter the matcher again.
    """

    def __init__(self, secrets: tuple[str, ...]):
        self._root = _SecretPrefix()
        for secret in set(secrets):
            if not secret:
                continue
            node = self._root
            for character in secret:
                child = node.children.get(character)
                if child is None:
                    child = _SecretPrefix()
                    node.children[character] = child
                node = child
            node.complete = True
        self._starts = None
        if self._root.children:
            characters = re.escape("".join(self._root.children))
            self._starts = re.compile(f"[{characters}]")

    def text(self, value: str) -> str:
        redacted, _ = self._scan(value, final=True)
        return redacted

    def payload(self, value: Any) -> Any:
        """Sanitize a JSON-compatible payload without changing its source."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.payload(item) for item in value]
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                name = str(key)
                normalized = "".join(char for char in name.casefold() if char.isalnum())
                sensitive = any(field in normalized for field in _CREDENTIAL_FIELDS)
                if sensitive:
                    sanitized[self.text(name)] = _REDACTED
                else:
                    sanitized[self.text(name)] = self.payload(item)
            return sanitized
        return value

    def stream(self) -> RedactedText:
        return RedactedText(self)

    def _scan(self, value: str, *, final: bool) -> tuple[str, str]:
        """Return safe text and the raw suffix that needs another chunk."""
        if self._starts is None:
            return value, ""
        fragments = []
        position = 0
        while match := self._starts.search(value, position):
            start = match.start()
            fragments.append(value[position:start])
            node = self._root
            cursor = start
            matched_end = None
            while cursor < len(value):
                child = node.children.get(value[cursor])
                if child is None:
                    break
                node = child
                cursor += 1
                if node.complete:
                    matched_end = cursor
            if cursor == len(value) and node.children and not final:
                # A complete short secret may still be the beginning of a
                # longer credential. Wait before deciding which one to hide.
                return "".join(fragments), value[start:]
            if matched_end is None:
                fragments.append(value[start])
                position = start + 1
            else:
                fragments.append(_REDACTED)
                position = matched_end
        fragments.append(value[position:])
        return "".join(fragments), ""


class RedactedText:
    """Buffer only an unfinished credential prefix for one streamed response."""

    def __init__(self, redactor: TraceRedactor):
        self._redactor = redactor
        self._pending = ""

    def feed(self, delta: str) -> str:
        safe, self._pending = self._redactor._scan(self._pending + delta, final=False)
        return safe

    def finish(self) -> str:
        if not self._pending:
            return ""
        self._pending = ""
        return _REDACTED
