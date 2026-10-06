"""Tool visibility policies; applications supply catalogs and request state.

Visibility controls model disclosure, not authorization inside a tool. Policies
may edit a definition and are evaluated in order without running the tool body.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence, Set
from copy import deepcopy
from inspect import isawaitable
from typing import Any, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.tools import Tool, ToolDefinition, ToolPrepareFunc
from pydantic_ai.usage import RunUsage

DepsT = TypeVar("DepsT")


async def _evaluate_gate(gate, ctx, definition) -> ToolDefinition | None:
    outcome = gate(ctx, definition)
    if isawaitable(outcome):
        outcome = await outcome
    if outcome is not None and not isinstance(outcome, ToolDefinition):
        raise TypeError("a tool gate must return ToolDefinition or None")
    return outcome


def _gate_from_predicate(
    predicate: Callable[[RunContext[DepsT], ToolDefinition], bool], name: str
) -> ToolPrepareFunc[DepsT]:
    async def prepare(
        ctx: RunContext[DepsT], tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        if predicate(ctx, tool_def):
            return tool_def
        return None

    prepare.__name__ = name
    return prepare


def _gate_name(kind: str, label: str | None) -> str:
    return kind if label is None else f"{kind}[{label}]"


def _dependency_selector(selector, attribute: str, default):
    if selector is not None:
        return selector
    return lambda deps: getattr(deps, attribute, default)


async def always_on(
    ctx: RunContext[Any], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Return the definition unchanged so this policy always permits disclosure."""
    return tool_def


def requires_client(
    selector: Callable[[DepsT], object], *, label: str | None = None
) -> ToolPrepareFunc[DepsT]:
    """Disclose a tool when the selected dependency is truthy."""
    return _gate_from_predicate(
        lambda ctx, definition: bool(selector(ctx.deps)),
        _gate_name("requires_client", label),
    )


def requires_feature_flag(flag_name: str) -> ToolPrepareFunc[Any]:
    """Disclose a tool when dependencies' user_ff_map enables the named flag."""
    if not flag_name.strip():
        raise ValueError("flag_name must not be blank")
    return _gate_from_predicate(
        lambda ctx, definition: bool(
            (getattr(ctx.deps, "user_ff_map", None) or {}).get(flag_name, False)
        ),
        _gate_name("requires_feature_flag", flag_name),
    )


def gate_all(*gates: ToolPrepareFunc[DepsT]) -> ToolPrepareFunc[DepsT]:
    """Compose sync or async gates, preserving edits and stopping at denial."""

    async def prepare(
        ctx: RunContext[DepsT], tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        definition = tool_def
        for policy in gates:
            definition = await _evaluate_gate(policy, ctx, definition)
            if definition is None:
                break
        return definition

    prepare.__name__ = (
        "gate_all["
        + ",".join(getattr(gate, "__name__", "policy") for gate in gates)
        + "]"
    )
    return prepare


def make_category_gate(
    category_ids_for_tool: Callable[[str], Set[str]],
    *,
    opened_categories: Callable[[DepsT], Set[str]] | None = None,
    label: str | None = None,
) -> ToolPrepareFunc[DepsT]:
    """Disclose uncategorized tools or tools in at least one opened category.

    By default, read ``opened_tool_categories`` from request dependencies.
    """
    opened = _dependency_selector(
        opened_categories, "opened_tool_categories", frozenset()
    )

    def visible(ctx, definition):
        categories = category_ids_for_tool(definition.name)
        return not categories or not categories.isdisjoint(opened(ctx.deps))

    return _gate_from_predicate(visible, _gate_name("gate_by_category", label))


def _append_prepare_gate(tool: Tool[DepsT], policy: ToolPrepareFunc[DepsT]) -> None:
    previous = tool.prepare
    tool.prepare = policy if previous is None else gate_all(previous, policy)


def install_category_gate(
    tool: Tool[DepsT],
    *,
    category_ids_for_tool: Callable[[str], Set[str]],
    opened_categories: Callable[[DepsT], Set[str]] | None = None,
) -> None:
    """Append a category gate to a categorized tool's existing preparation policy."""
    if category_ids_for_tool(tool.name):
        _append_prepare_gate(
            tool,
            make_category_gate(
                category_ids_for_tool, opened_categories=opened_categories
            ),
        )


def make_tool_unlock_gate(
    *,
    opened_tools: Callable[[DepsT], Set[str]] | None = None,
    label: str | None = None,
) -> ToolPrepareFunc[DepsT]:
    """Disclose tools listed in dependencies' ``opened_tool_names`` by default."""
    opened = _dependency_selector(opened_tools, "opened_tool_names", frozenset())
    return _gate_from_predicate(
        lambda ctx, definition: definition.name in opened(ctx.deps),
        _gate_name("gate_by_open_tool", label),
    )


def install_tool_unlock_gate(
    tool: Tool[DepsT], *, opened_tools: Callable[[DepsT], Set[str]] | None = None
) -> None:
    """Append an unlock gate while preserving the tool's existing preparation."""
    _append_prepare_gate(tool, make_tool_unlock_gate(opened_tools=opened_tools))


def make_load_tool_gate(
    tools_for_load_tool: Callable[[str], Set[str]],
    *,
    available_tools: Callable[[DepsT], Set[str] | None] | None = None,
    label: str | None = None,
) -> ToolPrepareFunc[DepsT]:
    """Disclose a loader when at least one of its candidate tools is admitted.

    By default, read ``available_gated_tool_names`` from dependencies. None
    permits all candidates; an empty set hides every loader.
    """
    available = _dependency_selector(
        available_tools, "available_gated_tool_names", None
    )

    def visible(ctx, definition):
        candidates = tools_for_load_tool(definition.name)
        admitted = available(ctx.deps)
        return (
            bool(candidates)
            if admitted is None
            else not candidates.isdisjoint(admitted)
        )

    return _gate_from_predicate(visible, _gate_name("gate_load_tool", label))


async def compute_admitted_tool_names(
    deps: DepsT, tools: Iterable[Tool[DepsT]]
) -> list[str]:
    """Evaluate dependency-only gates using a synthetic SDK context.

    Model/usage-sensitive policies should be evaluated by the actual agent.
    Definitions are copied so preview preparation cannot mutate SDK schemas.
    Returned names identify registered tools even if a gate edits their schema.
    """
    context = RunContext(deps=deps, model=None, usage=RunUsage())  # type: ignore[arg-type]
    names = []
    for tool in tools:
        if tool.prepare is None:
            names.append(tool.name)
            continue
        definition = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=deepcopy(tool.function_schema.json_schema),
            strict=tool.strict,
            sequential=tool.sequential,
            metadata=deepcopy(tool.metadata),
            timeout=tool.timeout,
            defer_loading=tool.defer_loading,
        )
        if await _evaluate_gate(tool.prepare, context, definition) is not None:
            names.append(tool.name)
    return names


def is_tool_enabled(
    tool_name: str,
    requested_sources: Sequence[str] | None,
    tool_to_sources: Mapping[str, frozenset[str]],
    *,
    and_combos: Mapping[str, frozenset[str]] | None = None,
) -> bool:
    """Any-source and all-source entries are alternative ways to enable a tool.

    None disables filtering; an empty requested list still filters cataloged
    tools. Uncataloged tools remain visible. An empty all-source entry is true.
    """
    if requested_sources is None:
        return True
    any_sources = tool_to_sources.get(tool_name)
    all_sources = (and_combos or {}).get(tool_name)
    if any_sources is None and all_sources is None:
        return True
    requested_set = frozenset(requested_sources)
    return bool(
        (any_sources is not None and requested_set.intersection(any_sources))
        or (all_sources is not None and all_sources <= requested_set)
    )


def make_data_source_gate(
    tool_to_sources: Mapping[str, frozenset[str]],
    *,
    and_combos: Mapping[str, frozenset[str]] | None = None,
    requested: Callable[[DepsT], Sequence[str] | None] | None = None,
    label: str | None = None,
) -> ToolPrepareFunc[DepsT]:
    """Snapshot source rules; read each request's chosen sources at preparation."""
    catalog = {name: frozenset(sources) for name, sources in tool_to_sources.items()}
    combinations = {
        name: frozenset(sources) for name, sources in (and_combos or {}).items()
    }
    sources = _dependency_selector(requested, "data_sources", None)
    return _gate_from_predicate(
        lambda ctx, definition: is_tool_enabled(
            definition.name, sources(ctx.deps), catalog, and_combos=combinations
        ),
        _gate_name("gate_by_data_source", label),
    )
