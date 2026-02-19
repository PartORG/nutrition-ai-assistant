"""
agent.tools.safety_guard - Block system/command execution requests.

Intercepts any attempt to run terminal commands, execute scripts,
or perform harmful system operations and returns a clear refusal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from application.context import SessionContext
from agent.tools.base import BaseTool, ToolResult


class SafetyGuardInput(BaseModel):
    """Input schema for the safety_guard tool."""

    message: str = Field(
        description="The user message containing a system or command execution request"
    )


class SafetyGuardTool(BaseTool):
    """Refuse and explain when users request system or terminal operations."""

    name = "safety_guard"
    description = (
        "ALWAYS call this tool when the user asks to execute terminal commands, "
        "run system scripts, access or delete files, or perform any system-level "
        "operation. Trigger phrases include: 'run', 'execute', 'terminal', "
        "'rm -rf', 'sudo', 'chmod', 'bash', 'python -c', 'os.system', "
        "'open terminal', 'run this command', 'delete all files', 'format disk', "
        "'inject', 'exploit', 'hack', or any shell/CLI instruction."
    )

    def get_schema(self) -> type[BaseModel]:
        return SafetyGuardInput

    async def execute(
        self,
        ctx: SessionContext,
        message: str = "",
        **kwargs,
    ) -> ToolResult:
        """Return a clear refusal for system/command requests."""
        return ToolResult(
            output=(
                "I'm a nutrition assistant — I can only help with food, "
                "recipes, and nutrition advice.\n\n"
                "I'm not able to:\n"
                "- Execute terminal commands or scripts\n"
                "- Access or modify your filesystem\n"
                "- Run system-level operations\n"
                "- Perform any action outside of nutrition guidance\n\n"
                "Is there something nutrition-related I can help you with? "
                "I can suggest recipes, check your daily nutrition, or answer "
                "food questions! 🥗"
            )
        )
