"""Select and disclose admitted tools from a validated catalog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar

from pydantic_ai import RunContext, Tool


class LoadToolDeps(Protocol):
    available_gated_tool_names: set[str] | None
    opened_tool_names: set[str]


DepsT = TypeVar("DepsT", bound=LoadToolDeps)


@dataclass(frozen=True)
class LoadTool(Generic[DepsT]):
    """Snapshot catalog membership; caller dependencies own per-run unlocks.

    Tool objects themselves remain SDK-owned and must not be renamed after
    registration. Options preserve their declared order. Admission is applied
    every time the loader runs, including when an option was opened earlier.
    """

    category_id: str
    category_prompt: str
    tools: tuple[Tool[DepsT], ...]
    tools_by_option: Mapping[str, tuple[Tool[DepsT], ...]] = field(default_factory=dict)
    tool_prompts: tuple[tuple[Tool[DepsT], str], ...] = ()
    _catalog: Mapping[str, Tool[DepsT]] = field(init=False, repr=False)
    _guidance: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for label, text in (
            ("category_id", self.category_id),
            ("category_prompt", self.category_prompt),
        ):
            if not text.strip():
                raise ValueError(f"{label} must not be blank")
        tools = tuple(self.tools)
        catalog = self._index_tools(tools)
        options = self._validate_options(catalog)
        guidance = self._collect_guidance(catalog)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "tools_by_option", MappingProxyType(options))
        object.__setattr__(self, "tool_prompts", tuple(self.tool_prompts))
        object.__setattr__(self, "_catalog", MappingProxyType(catalog))
        object.__setattr__(self, "_guidance", MappingProxyType(guidance))

    def _validate_options(
        self, catalog: Mapping[str, Tool[DepsT]]
    ) -> dict[str, tuple[Tool[DepsT], ...]]:
        options = {}
        for key, selection in self.tools_by_option.items():
            if not key.strip():
                raise ValueError("option names must not be blank")
            chosen = tuple(selection)
            self._index_tools(chosen)
            self._require_members(chosen, catalog)
            options[key] = chosen
        return options

    def _collect_guidance(self, catalog: Mapping[str, Tool[DepsT]]) -> dict[str, str]:
        guidance = {}
        for tool, prompt in self.tool_prompts:
            self._require_members((tool,), catalog)
            if tool.name in guidance:
                raise ValueError("tool_prompts contains duplicate tools")
            guidance[tool.name] = prompt.strip()
        return guidance

    @staticmethod
    def _index_tools(tools: Iterable[Tool[DepsT]]) -> dict[str, Tool[DepsT]]:
        result = {}
        for tool in tools:
            if tool.name in result:
                raise ValueError(f"tools contain duplicate names: {tool.name}")
            result[tool.name] = tool
        return result

    @staticmethod
    def _require_members(
        tools: Iterable[Tool[DepsT]], catalog: Mapping[str, Tool[DepsT]]
    ) -> None:
        for tool in tools:
            if catalog.get(tool.name) is not tool:
                raise ValueError(
                    f"tool {tool.name!r} is not the registered category tool"
                )

    @property
    def loader_name(self) -> str:
        return "load_" + self.category_id + "_tools"

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._catalog)

    def tools_for(self, option: str | Enum | None = None) -> tuple[Tool[DepsT], ...]:
        if option is None:
            return self.tools
        key = str(option.value) if isinstance(option, Enum) else str(option)
        if key not in self.tools_by_option:
            raise ValueError(
                f"{self.category_id} option must be one of: {', '.join(self.tools_by_option) or 'none'}"
            )
        return self.tools_by_option[key]

    def admitted_tools(
        self, deps: DepsT, option: str | Enum | None = None
    ) -> tuple[Tool[DepsT], ...]:
        selected = self.tools_for(option)
        allowed = deps.available_gated_tool_names
        return tuple(
            tool for tool in selected if allowed is None or tool.name in allowed
        )

    async def load(
        self, ctx: RunContext[DepsT], option: str | Enum | None = None
    ) -> str:
        """Open only selected admitted names, and return their ordered guidance."""
        selection = self.admitted_tools(ctx.deps, option)
        ctx.deps.opened_tool_names.update(item.name for item in selection)
        paragraphs = [self.category_prompt.strip()]
        if selection:
            paragraphs.extend(
                self._guidance[item.name]
                for item in selection
                if self._guidance.get(item.name)
            )
        else:
            paragraphs.append(
                "This selection opens no tools under the request's admission rules."
            )
        return "\n\n".join(paragraphs)


__all__ = ["LoadTool", "LoadToolDeps"]
