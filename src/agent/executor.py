"""
agent.executor - Agent execution engine.

The single class that runs the LLM + tool selection loop.
No component construction, no global state, no business logic.

Extracted from host_agent.py create_nutrition_agent() and chat_loop().
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from langchain.agents import AgentExecutor as LangChainAgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel

from application.context import SessionContext
from agent.tools.registry import ToolRegistry
from agent.memory import ConversationMemory

if TYPE_CHECKING:
    from application.services.chat_history import ChatHistoryService

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Runs the LLM + tool selection loop.

    Constructed by factory.py with all dependencies injected.
    Stateless per call — all state flows through SessionContext.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: ToolRegistry,
        memory: ConversationMemory,
        system_prompt: str,
        max_iterations: int = 3,
        chat_history_service: ChatHistoryService | None = None,
    ):
        self._tools = tools
        self._memory = memory
        self._system_prompt = system_prompt
        self._llm = llm
        self._max_iterations = max_iterations
        self._executor: LangChainAgentExecutor | None = None
        self._chat_history_service = chat_history_service

    def _build_executor(self, ctx: SessionContext) -> LangChainAgentExecutor:
        """Build the LangChain agent executor with tools bound to context."""
        lc_tools = self._tools.to_langchain_tools(ctx)

        # Augment system prompt with the user's health profile so the agent
        # can refer to their conditions, restrictions, and foods to avoid
        # throughout the conversation without the user repeating themselves.
        system_prompt = self._system_prompt
        health_section = _build_health_context(ctx)
        if health_section:
            system_prompt = system_prompt + "\n\n" + health_section

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=lc_tools,
            prompt=prompt,
        )

        return LangChainAgentExecutor(
            agent=agent,
            tools=lc_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=self._max_iterations,
            return_intermediate_steps=True,
        )

    async def run(self, ctx: SessionContext, user_input: str) -> str:
        """Process a user message and return the agent's response.

        On first call: loads previous conversation from DB (if persistence
        is configured) and builds the LangChain executor.

        After each turn: persists both user and assistant messages to DB.

        Args:
            ctx:        Session context (user info, scratch space).
            user_input: The user's message text.

        Returns:
            The agent's response string.
        """
        if self._executor is None:
            # First call — load history and build executor
            if self._chat_history_service:
                await self._chat_history_service.ensure_conversation(ctx)
                await self._memory.load_from_db(ctx.conversation_id)
            self._executor = self._build_executor(ctx)

        logger.info(
            "Agent processing (user=%d, conversation=%s): %s",
            ctx.user_id, ctx.conversation_id, user_input[:80],
        )

        # Guarantee the original user message is available to tools via
        # ctx.scratch so they can pass it verbatim to the recommendation
        # pipeline — bypassing any LLM rewriting of the tool argument.
        ctx.scratch["original_query"] = user_input

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                self._executor.invoke,
                {"input": user_input, "chat_history": self._memory.messages},
            )
        except Exception:
            logger.exception(
                "Agent execution failed for user %d — returning friendly error",
                ctx.user_id,
            )
            error_output = (
                "I'm sorry, I ran into a problem while processing your request. "
                "Please try again, or rephrase your question."
            )
            self._memory.add_user_message(user_input)
            self._memory.add_ai_message(error_output)
            if self._chat_history_service:
                try:
                    await self._chat_history_service.save_user_message(ctx, user_input)
                    await self._chat_history_service.save_assistant_message(ctx, error_output)
                except Exception:
                    pass
            return error_output

        # For recipe/image tools, return the tool observation verbatim.
        # The LLM's "final answer" step tends to summarise or truncate the
        # full recipe details — bypassing it guarantees the user sees every
        # recipe with all ingredients, instructions and nutrition data.
        _DIRECT_OUTPUT_TOOLS = {"search_recipes", "analyze_image"}
        output = response.get("output", "I couldn't generate a response.")
        for action, observation in response.get("intermediate_steps", []):
            if getattr(action, "tool", "") in _DIRECT_OUTPUT_TOOLS:
                output = str(observation)
                break

        # Update in-memory history manually (no ConversationBufferMemory)
        self._memory.add_user_message(user_input)
        self._memory.add_ai_message(output)

        # Persist both messages to DB (non-blocking failure)
        if self._chat_history_service:
            try:
                await self._chat_history_service.save_user_message(ctx, user_input)
                await self._chat_history_service.save_assistant_message(ctx, output)
            except Exception:
                logger.exception(
                    "Failed to persist chat messages — continuing without save",
                )

        logger.debug("Agent response: %s", output[:100])
        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_health_context(ctx: SessionContext) -> str:
    """Build a USER HEALTH PROFILE block from session context user_data.

    Returns an empty string if the user has no saved profile data.
    """
    conditions = ctx.user_data.get("health_conditions", [])
    restrictions = ctx.user_data.get("restrictions", [])
    raw_avoid = ctx.user_data.get("avoid", [])
    preferences = ctx.user_data.get("preferences", [])

    if not any([conditions, restrictions, raw_avoid, preferences]):
        return ""

    # Flatten avoid: each item may itself be a comma-separated string
    flat_avoid: list[str] = []
    for item in raw_avoid:
        if isinstance(item, str):
            flat_avoid.extend(s.strip() for s in item.split(",") if s.strip())
        else:
            flat_avoid.append(str(item))

    lines = ["USER HEALTH PROFILE (loaded from their account — always respect these):"]
    if conditions:
        lines.append(f"- Health conditions: {', '.join(conditions)}")
    if restrictions:
        lines.append(f"- Dietary restrictions: {', '.join(restrictions)}")
    if flat_avoid:
        lines.append(f"- Foods to avoid: {', '.join(flat_avoid)}")
    if preferences:
        lines.append(f"- Food preferences: {', '.join(preferences)}")
    lines.append("Keep these constraints in mind for every recipe recommendation.")
    return "\n".join(lines)
