"""
agent.tools.crisis_support - Mental health crisis intervention.

Responds with compassion and crisis hotline numbers when a user
expresses suicidal thoughts or severe emotional distress.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolResult
from application.context import SessionContext

_CRISIS_RESOURCES = """\
**Crisis Support Lines (free, available 24/7)**

- 🌍 **International directory**: https://www.iasp.info/resources/Crisis_Centres/
- 🇺🇸 **USA — 988 Suicide & Crisis Lifeline**: Call or text **988**
- 🇺🇸 **USA — Crisis Text Line**: Text **HOME** to **741741**
- 🇬🇧 **UK — Samaritans**: Call **116 123**
- 🇨🇦 **Canada — Talk Suicide Canada**: Call **1-833-456-4566**
- 🇦🇺 **Australia — Lifeline**: Call **13 11 14**
- 🌐 **Befrienders Worldwide**: https://www.befrienders.org
- 🚨 **Emergency services**: **112** (EU) · **911** (US) · **999** (UK)\
"""


class CrisisSupportInput(BaseModel):
    """Input schema for the crisis_support tool."""

    message: str = Field(
        description="The user message expressing distress or suicidal thoughts"
    )


class CrisisSupportTool(BaseTool):
    """Respond with compassion and crisis resources to users in distress."""

    name = "crisis_support"
    description = (
        "IMMEDIATELY call this tool when the user expresses suicidal thoughts, "
        "desire to self-harm, or severe emotional distress. "
        "Trigger phrases: 'I want to die', 'kill myself', 'end my life', "
        "'hurt myself', 'don't want to live', 'no reason to live', "
        "'can't go on', 'suicidal', 'take my own life', 'self-harm'."
    )

    def get_schema(self) -> type[BaseModel]:
        return CrisisSupportInput

    async def execute(
        self,
        ctx: SessionContext,
        message: str = "",
        **kwargs,
    ) -> ToolResult:
        """Return a compassionate response with crisis hotline numbers."""
        return ToolResult(
            output=(
                "I hear you, and I'm really glad you reached out. "
                "What you're feeling matters deeply, and you are not alone. 💙\n\n"
                "Please reach out to a crisis support line right now — "
                "trained counselors are there to listen, free of charge:\n\n"
                + _CRISIS_RESOURCES
                + "\n\n---\n"
                "You deserve support and care. When you're ready, "
                "I'm here to help with your nutrition and wellbeing too. 💙"
            )
        )
