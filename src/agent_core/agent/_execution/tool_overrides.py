"""Copy recoverable tool registrations and assemble task-local overrides."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from copy import copy
from dataclasses import replace
from typing import Any

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import AbstractToolset

from agent_core.tools.outcomes import tool_error_metadata
from agent_core.agent.request import Deps

logger = logging.getLogger("agent_core._execution.tools")


def recoverable_tool(tool: Tool[Deps], agent_name: str) -> Tool[Deps]:
    """Copy a tool registration and return ordinary execution failures to the model.

    Keep the validated schema and SDK registration settings. ModelRetry remains
    an SDK correction request; cancellation propagates. Synchronous tools run
    in a worker thread, which can outlive cancellation of the awaiting task.
    """
    schema = tool.function_schema

    async def invoke(*args: Any, **kwargs: Any) -> Any:
        try:
            if schema.is_async:
                return await schema.function(*args, **kwargs)
            return await asyncio.to_thread(schema.function, *args, **kwargs)
        except ModelRetry:
            # Let the SDK ask for corrected arguments and enforce its retry budget.
            raise
        except Exception as failure:
            logger.exception(
                "Tool exception converted to an error result for the next model turn",
                extra={
                    "event_name": "agent.tool_failed",
                    "agent_name": agent_name,
                    "tool_name": tool.name,
                },
            )
            positional = (
                args[1:] if args and type(args[0]).__name__ == "RunContext" else args
            )
            arguments = [repr(value) for value in positional]
            arguments += [f"{name}={value!r}" for name, value in kwargs.items()]
            description = f"{tool.name}({', '.join(arguments)}) failed: {failure}"
            return ToolReturn(return_value=description, metadata=tool_error_metadata())

    # Preserve the SDK tool's complete registration settings and validated
    # schema, without rebuilding its arguments from the variadic wrapper.
    wrapped = copy(tool)
    wrapped.function = invoke
    wrapped.function_schema = replace(schema, function=invoke, is_async=True)
    return wrapped


def tool_overrides(
    construction: Sequence[Tool[Deps]],
    agent_name: str,
    replacement: Sequence[Tool[Deps]] | None,
    additions: Sequence[Tool[Deps]] | None,
    toolsets: Sequence[AbstractToolset[Deps]] | None,
) -> dict[str, Any]:
    """Build per-call tool replacements or additions without mutating registrations.

    Existing construction tools win name collisions with additions, then the
    first addition wins. None inherits toolsets; an empty sequence clears them.
    """
    overrides: dict[str, Any] = {}
    if replacement is not None:
        overrides["tools"] = tuple(
            recoverable_tool(tool, agent_name) for tool in replacement
        )
    elif additions is not None:
        merged = list(construction)
        names = {tool.name for tool in merged}
        for tool in additions:
            # Construction tools and the first dynamic occurrence win.
            if tool.name in names:
                logger.warning(
                    "Dropping duplicate per-call tool",
                    extra={
                        "agent_name": agent_name,
                        "tool_name": tool.name,
                    },
                )
                continue
            names.add(tool.name)
            merged.append(recoverable_tool(tool, agent_name))
        overrides["tools"] = tuple(merged)
    if toolsets is not None:
        overrides["toolsets"] = toolsets
    return overrides
