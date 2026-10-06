"""Tool contracts at catalog, policy and argument boundaries."""

import asyncio
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import pytest
from pydantic_ai import Tool
from pydantic_ai.tools import ToolDefinition

from agent_core.streaming import DoneReason, StreamEvent
from agent_core.streaming import StreamPrinter
from agent_core.tools.validation import TimeFormat, check_country_code, check_date, check_date_window
from agent_core.tools.catalog import LoadTool
from agent_core.tools.visibility import compute_admitted_tool_names, gate_all, is_tool_enabled, make_category_gate, make_data_source_gate, make_load_tool_gate, make_tool_unlock_gate


def lookup(value: int) -> int:
    return value


@pytest.mark.parametrize(
    "value", ["2026-2-01", "2026-02-29", "２０２６-02-01", "2026-01-01\n"]
)
def test_dates_require_exact_ascii_calendar_form(value):
    with pytest.raises(ValueError, match="date"):
        check_date(value, "as_of")


def test_dates_and_windows_handle_optional_typed_values_and_inclusive_days():
    check_date([None, 4, "2024-02-29"], "dates")
    check_date("2026-01-01 23:59:59", "instant", TimeFormat.DATE_SECOND)
    check_date_window(
        "2026-01-01", "2026-01-31", start_param="first", end_param="last", max_days=31
    )
    with pytest.raises(ValueError, match="31 days"):
        check_date_window(
            "2026-01-01",
            "2026-01-31",
            start_param="first",
            end_param="last",
            max_days=30,
        )
    with pytest.raises(ValueError, match="after"):
        check_date_window(
            "2026-01-02", "2026-01-01", start_param="first", end_param="last"
        )
    with pytest.raises(ValueError, match="positive integer"):
        check_date_window(
            None, None, start_param="first", end_param="last", max_days=True
        )
    check_country_code(["us", "EUZ", None], "region")
    with pytest.raises(ValueError):
        check_country_code("uſ", "region")


def test_gate_composition_preserves_edits_short_circuits_and_propagates_cancellation():
    definition = ToolDefinition(name="lookup")
    calls = []

    def edit(ctx, value):
        calls.append("edit")
        return replace(value, description="prepared")

    async def deny(ctx, value):
        calls.append(value.description)
        return None

    async def unreachable(ctx, value):
        pytest.fail("a denied chain must stop")

    assert asyncio.run(gate_all(edit, deny, unreachable)(None, definition)) is None
    assert calls == ["edit", "prepared"]
    assert asyncio.run(gate_all()(None, definition)) is definition

    async def cancelled(ctx, value):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(gate_all(cancelled)(None, definition))


def test_admission_preview_cannot_mutate_registered_schema():
    def prepare(ctx, value):
        value.parameters_json_schema["properties"].clear()
        return value

    tool = Tool(lookup, prepare=prepare)
    assert asyncio.run(compute_admitted_tool_names(None, [tool])) == ["lookup"]
    assert "value" in tool.function_schema.json_schema["properties"]


def test_source_policy_snapshots_rules_and_keeps_alternative_any_all_semantics():
    rules = {"lookup": frozenset({"a"})}
    gate = make_data_source_gate(rules, and_combos={"lookup": frozenset({"b", "c"})})
    rules["lookup"] = frozenset({"mutated"})
    definition = ToolDefinition(name="lookup")
    for sources, enabled in [
        (["a"], True),
        (["b", "c"], True),
        (["b"], False),
        ([], False),
        (None, True),
    ]:
        ctx = SimpleNamespace(deps=SimpleNamespace(data_sources=sources))
        assert (asyncio.run(gate(ctx, definition)) is not None) is enabled
    assert is_tool_enabled("unknown", [], {})


def test_disclosure_gates_distinguish_empty_admission_from_no_filter():
    tool = ToolDefinition(name="lookup")
    loader = ToolDefinition(name="load_example_tools")
    deps = SimpleNamespace(
        opened_tool_names={"lookup"},
        opened_tool_categories={"example"},
        available_gated_tool_names=set(),
    )
    ctx = SimpleNamespace(deps=deps)
    assert asyncio.run(make_tool_unlock_gate()(ctx, tool)) is tool
    assert asyncio.run(make_category_gate(lambda name: {"example"})(ctx, tool)) is tool
    gate = make_load_tool_gate(lambda name: {"lookup"})
    assert asyncio.run(gate(ctx, loader)) is None
    deps.available_gated_tool_names = None
    assert asyncio.run(gate(ctx, loader)) is loader


def test_loader_snapshots_membership_and_guidance_and_opens_only_admitted_names():
    first, second = Tool(lookup, name="first"), Tool(lookup, name="second")
    options = {"selected": (second, first)}
    catalog = LoadTool(
        "example",
        "Guidance",
        (first, second),
        options,
        ((first, "First help"), (second, "Second help")),
    )
    options.clear()
    deps = SimpleNamespace(
        available_gated_tool_names={"second"}, opened_tool_names={"already"}
    )
    result = asyncio.run(catalog.load(SimpleNamespace(deps=deps), "selected"))
    assert result == "Guidance\n\nSecond help"
    assert deps.opened_tool_names == {"already", "second"}
    with pytest.raises(TypeError):
        catalog.tools_by_option["new"] = (first,)
    with pytest.raises(ValueError, match="registered"):
        LoadTool(
            "example", "Guidance", (first,), {"bad": (Tool(lookup, name="first"),)}
        )


def test_printer_orders_delta_and_tool_blocks_without_ansi_in_non_tty_output():
    output = StringIO()
    printer = StreamPrinter(output)
    printer.write(StreamEvent.text_delta("Hello"))
    printer.write(StreamEvent.text_delta(" world"))
    printer.write(StreamEvent.tool_call("lookup", "call", '{"value": 1}'))
    printer.write(StreamEvent.done(DoneReason.COMPLETE))
    value = output.getvalue()
    assert "\x1b" not in value
    assert value.startswith("Hello world\nlookup\n")
    assert "call_id: call" in value
    assert value.endswith("[done] reason=complete\n")
