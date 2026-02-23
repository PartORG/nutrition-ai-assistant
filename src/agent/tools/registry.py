"""
agent.tools.registry - Tool registration, discovery, and invocation.

Central registry that manages all available tools and provides
LangChain-compatible tool wrappers.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import StructuredTool

from application.context import SessionContext
from agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages tool registration and invocation."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def all(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    async def invoke(self, name: str, ctx: SessionContext, **kwargs) -> str:
        """Invoke a tool by name and auto-store results in ctx.scratch.

        Returns the string output (what the LLM sees).
        """
        tool = self.get(name)
        result = await tool.execute(ctx, **kwargs)

        if result.store_as and result.data is not None:
            ctx.scratch[result.store_as] = result.data
            logger.debug("Stored result in ctx.scratch['%s']", result.store_as)

        return result.output

    def to_langchain_tools(self, ctx: SessionContext) -> list[StructuredTool]:
        """Convert all registered tools to LangChain StructuredTools.

        Binds the SessionContext so LangChain's agent can call them.
        """
        lc_tools = []
        for tool in self._tools.values():
            schema = tool.get_schema()

            # Create a closure that captures both tool and ctx
            def _make_func(t: BaseTool, context: SessionContext):
                def func(**kwargs: Any) -> str:
                    import asyncio
                    # This function always runs inside a thread-pool worker
                    # (via run_in_executor), so there is no running event loop
                    # in this thread — asyncio.run() creates a fresh one.
                    result = asyncio.run(t.execute(context, **kwargs))

                    if result.store_as and result.data is not None:
                        context.scratch[result.store_as] = result.data

                    return result.output
                return func

            lc_tools.append(StructuredTool.from_function(
                func=_make_func(tool, ctx),
                name=tool.name,
                description=tool.description,
                args_schema=schema,
            ))
        return lc_tools
