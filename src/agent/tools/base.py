"""
agent.tools.base - Base tool interface and result container.

All agent tools inherit from BaseTool and return ToolResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel

from application.context import SessionContext


@dataclass
class ToolResult:
    """Result returned by a tool execution.

    output:    String shown to the agent/user (markdown, text).
    data:      Structured data for inter-tool communication (not passed through LLM).
    store_as:  If set, data is automatically stored in ctx.scratch[store_as].
    """
    output: str
    data: Any = None
    store_as: Optional[str] = None


class BaseTool(ABC):
    """Abstract base for all agent tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, ctx: SessionContext, **kwargs) -> ToolResult:
        """Execute the tool with the given session context and arguments."""
        ...

    @abstractmethod
    def get_schema(self) -> type[BaseModel]:
        """Return the Pydantic schema for this tool's input arguments."""
        ...
