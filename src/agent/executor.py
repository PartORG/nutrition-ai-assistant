"""
agent.executor - Agent execution engine.

The single class that runs the LLM + tool selection loop.
No component construction, no global state, no business logic.

Extracted from host_agent.py create_nutrition_agent() and chat_loop().
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from langchain.agents import AgentExecutor as LangChainAgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel

from application.context import SessionContext
from agent.tools.registry import ToolRegistry  # also used in type hints below
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

        # Clear the previous turn's tool-call cache so the backup logic
        # below only triggers for a tool called *this* turn.
        ctx.scratch.pop("_last_tool_call", None)

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

        # For these tools, return the tool observation verbatim instead of
        # letting the LLM rewrite or summarise it.
        # - search_recipes / analyze_image: full recipe markdown must not be truncated
        # - nutrition_status: structured table must reach the user unchanged
        # - safety_guard / crisis_support: critical messages must not be softened
        _DIRECT_OUTPUT_TOOLS = {
            "search_recipes",
            "analyze_image",
            "nutrition_status",
            "safety_guard",
            "crisis_support",
            "save_recipe",
            "show_recipe",
        }
        output = response.get("output", "I couldn't generate a response.")
        steps = response.get("intermediate_steps", [])
        logger.info(
            "Agent finished: %d tool step(s), output starts with: %s",
            len(steps),
            output[:80],
        )
        found_direct_tool = False
        for action, observation in steps:
            tool_name = getattr(action, "tool", "")
            logger.info(
                "Step: action_type=%s tool=%r obs_type=%s",
                type(action).__name__, tool_name, type(observation).__name__,
            )
            if tool_name in _DIRECT_OUTPUT_TOOLS:
                # LangChain >= 0.3 may return a BaseMessage object as observation;
                # extract .content to get the plain text tool output.
                if hasattr(observation, "content"):
                    output = observation.content
                else:
                    output = str(observation)
                logger.info("Using direct output from tool '%s'", tool_name)
                found_direct_tool = True
                break

        if not found_direct_tool:
            # Fallback: some Ollama models output a raw JSON tool-call string
            # instead of invoking the tool via the function-calling API.
            # Detect that pattern and invoke the tool manually.
            fallback_result = await _try_raw_tool_call_fallback(output, ctx, self._tools)
            if fallback_result is not None:
                output = fallback_result
            else:
                # Secondary backup: if the tool WAS executed (captured in ctx.scratch
                # by the registry) but intermediate_steps didn't expose it correctly,
                # use the cached output directly.
                last_call = ctx.scratch.get("_last_tool_call")
                if last_call:
                    last_name, last_output = last_call
                    if last_name in _DIRECT_OUTPUT_TOOLS and last_output:
                        logger.warning(
                            "Direct tool '%s' not found in intermediate_steps — "
                            "using ctx.scratch backup output",
                            last_name,
                        )
                        output = last_output

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


async def _try_raw_tool_call_fallback(
    output: str,
    ctx: SessionContext,
    tools: "ToolRegistry",
) -> str | None:
    """Detect and execute a raw JSON tool-call that the model emitted as text.

    Some Ollama models don't support the function-calling API and instead output
    something like:
        {"name": "analyze_image", "parameters": {"image_path": "..."}}

    When that happens LangChain treats it as a final answer instead of a tool
    invocation.  This function detects that pattern and manually invokes the
    tool so the user gets a real response.

    Returns the tool output string, or None if the output is not a raw tool call.
    """
    raw = output.strip()

    # Strip markdown code fences FIRST so the '{' check below works
    # e.g.  ```json\n{"name": ...}\n```
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\s*```$", "", raw, flags=re.DOTALL).strip()

    # Find first '{' — skip any preamble text the model may have added
    # (e.g. "I'll analyze this image.\n{...}")
    brace_pos = raw.find("{")
    if brace_pos == -1:
        return None
    raw = raw[brace_pos:]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Windows paths contain backslashes that aren't valid JSON escape
        # sequences (e.g. C:\Users → \U is invalid in JSON).
        # Try to repair by escaping lone backslashes before re-parsing.
        try:
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
            parsed = json.loads(fixed)
            logger.info("Raw tool-call fallback: repaired JSON backslash escaping")
        except json.JSONDecodeError:
            logger.info(
                "Raw tool-call fallback: JSON parse failed — output: %s", raw[:200]
            )
            return None

    if not isinstance(parsed, dict) or "name" not in parsed:
        return None

    tool_name = parsed.get("name", "")
    # Support both "parameters" (Ollama style) and "arguments" (OpenAI style)
    tool_args = parsed.get("parameters") or parsed.get("arguments") or {}
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            tool_args = {}

    if tool_name not in tools.names():
        logger.info("Raw tool-call fallback: unknown tool '%s'", tool_name)
        return None

    # For analyze_image: always extract the image path directly from the
    # original query stored in ctx.scratch.  This is the most reliable
    # source and avoids any path-escaping issues in the model's JSON output.
    if tool_name == "analyze_image":
        original = ctx.scratch.get("original_query", "")
        path_match = re.search(r"\[IMAGE:([^\]]+)\]", original, re.IGNORECASE)
        if path_match:
            extracted = path_match.group(1).strip()
            tool_args = dict(tool_args)
            tool_args["image_path"] = extracted
            logger.info(
                "Raw tool-call fallback: extracted image_path from original query: %s",
                extracted,
            )
        elif not tool_args.get("image_path"):
            logger.warning(
                "Raw tool-call fallback: analyze_image has no image_path "
                "in args or original query — skipping"
            )
            return None

    logger.warning(
        "Raw tool-call fallback triggered for tool '%s' — "
        "model does not support native function calling",
        tool_name,
    )
    try:
        return await tools.invoke(tool_name, ctx, **tool_args)
    except Exception:
        logger.exception("Raw tool-call fallback failed for tool '%s'", tool_name)
        return None
