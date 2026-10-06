"""Loader catalog validation catches ambiguous names before a run starts."""

import pytest
from pydantic_ai import Tool

from agent_core.tools.catalog import LoadTool


def lookup() -> str:
    return "result"


@pytest.mark.parametrize("location", ["category", "option"])
def test_duplicate_names_rejected_even_for_distinct_tool_objects(location):
    first, second = Tool(lookup, name="lookup"), Tool(lookup, name="lookup")
    tools = (first, second) if location == "category" else (first,)
    options = {} if location == "category" else {"selected": (first, second)}
    with pytest.raises(ValueError, match="duplicate names"):
        LoadTool(
            category_id="example", category_prompt="Example tools",
            tools=tools, tools_by_option=options,
        )


def test_valid_option_keeps_catalog_order():
    first, second = Tool(lookup, name="first"), Tool(lookup, name="second")
    loader = LoadTool(
        category_id="example", category_prompt="Example tools",
        tools=(first, second), tools_by_option={"selected": (second, first)},
    )
    assert loader.tools_for("selected") == (second, first)
