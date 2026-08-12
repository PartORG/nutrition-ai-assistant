"""
agent.executor - Agent execution engine (LangGraph).

Runs the LLM + tool-selection loop using LangGraph's `create_react_agent`
(the supported replacement for the deprecated LangChain `AgentExecutor`).

No business logic, no global state — all state flows through SessionContext.
Tools are real async coroutines (see ToolRegistry.to_langchain_tools), so the
graph runs fully async with no thread hops or `asyncio.run`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agent.memory import ConversationMemory
from agent.tools.registry import ToolRegistry
from application.context import SessionContext
from domain.models import NutritionValues, Recipe

if TYPE_CHECKING:
    from application.services.chat_history import ChatHistoryService

logger = logging.getLogger(__name__)

# Tools whose output must reach the user verbatim — the LLM must not rewrite or
# summarise them:
#   - search_recipes / analyze_image: full recipe markdown must not be truncated
#   - nutrition_status: structured table must reach the user unchanged
#   - safety_guard / crisis_support: critical messages must not be softened
#   - save_recipe / show_recipe: deterministic confirmations / listings
_DIRECT_OUTPUT_TOOLS = {
    "search_recipes",
    "analyze_image",
    "nutrition_status",
    "safety_guard",
    "crisis_support",
    "save_recipe",
    "show_recipe",
}


class AgentExecutor:
    """Runs the LLM + tool-selection loop via a LangGraph ReAct agent.

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
        self._graph = None
        self._chat_history_service = chat_history_service

    def _build_graph(self, ctx: SessionContext):
        """Build the LangGraph ReAct agent with tools bound to the context."""
        lc_tools = self._tools.to_langchain_tools(ctx)

        # Augment the system prompt with the user's health profile so the agent
        # honours their conditions/restrictions without the user repeating them.
        system_prompt = self._system_prompt
        health_section = _build_health_context(ctx)
        if health_section:
            system_prompt = system_prompt + "\n\n" + health_section

        return _create_react_agent(self._llm, lc_tools, system_prompt)

    async def _restore_recipe_cache(self, ctx: SessionContext) -> None:
        """Load cached recipes from DB into ctx.scratch on session resume."""
        if not self._chat_history_service:
            return
        try:
            cache_json = await self._chat_history_service.load_recipe_cache(
                ctx.conversation_id
            )
            if cache_json:
                recipes = _deserialize_recipes(cache_json)
                if recipes:
                    ctx.scratch["_cached_safe_recipes"] = recipes
                    logger.info(
                        "Restored %d cached recipe(s) for conversation %s",
                        len(recipes), ctx.conversation_id,
                    )
        except Exception:
            logger.warning("Failed to restore recipe cache — continuing without it")

    async def _persist_recipe_cache(self, ctx: SessionContext) -> None:
        """Persist the current recipe list to DB so it survives reconnects."""
        if not self._chat_history_service:
            return
        rec_result = ctx.scratch.get("last_recommendations")
        if rec_result is None:
            return
        recipes = rec_result.safe_recipes
        if not recipes:
            return
        try:
            cache_json = _serialize_recipes(recipes)
            await self._chat_history_service.save_recipe_cache(ctx, cache_json)
        except Exception:
            logger.warning(
                "Failed to persist recipe cache — save may not work after reconnect"
            )

    async def run(self, ctx: SessionContext, user_input: str) -> str:
        """Process a user message and return the agent's response.

        On first call: loads previous conversation from DB (if persistence is
        configured) and builds the LangGraph agent. After each turn: persists
        both user and assistant messages to DB.
        """
        if self._graph is None:
            # First call — load history and build the graph.
            if self._chat_history_service:
                await self._chat_history_service.ensure_conversation(ctx)
                await self._memory.load_from_db(ctx.conversation_id)
                await self._restore_recipe_cache(ctx)
            self._graph = self._build_graph(ctx)

        logger.info(
            "Agent processing (user=%d, conversation=%s): %s",
            ctx.user_id, ctx.conversation_id, user_input[:80],
        )

        # Make the original message available to tools so they can pass it
        # verbatim to the recommendation pipeline (bypassing LLM rewriting).
        ctx.scratch["original_query"] = user_input

        # ── Deterministic image fast-path ────────────────────────────────────
        # When the message carries an [IMAGE:...] tag we invoke analyze_image
        # directly. This is more reliable than prompt instructions because the
        # LLM may otherwise pick search_recipes when text sounds recipe-like.
        image_output = await _route_image_if_present(user_input, ctx, self._tools)
        if image_output is not None:
            return await self._finalize(ctx, user_input, image_output)

        messages = list(self._memory.messages) + [HumanMessage(content=user_input)]
        try:
            result = await self._graph.ainvoke(
                {"messages": messages},
                config={"recursion_limit": 2 * self._max_iterations + 1},
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
            return await self._finalize(ctx, user_input, error_output)

        output = self._extract_output(result.get("messages", []))
        return await self._finalize(ctx, user_input, output)

    @staticmethod
    def _extract_output(messages: list) -> str:
        """Pick the user-facing response from the graph's message list.

        Direct-output tools win: if one ran, its result is returned verbatim
        (not the LLM's summary). Otherwise the final assistant message is used.
        """
        for msg in messages:
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") in _DIRECT_OUTPUT_TOOLS:
                content = msg.content
                if content:
                    return content if isinstance(content, str) else str(content)

        if messages:
            content = getattr(messages[-1], "content", "")
            if content:
                return content if isinstance(content, str) else str(content)
        return "I couldn't generate a response."

    async def _finalize(self, ctx: SessionContext, user_input: str, output: str) -> str:
        """Update memory, persist messages + recipe cache, and return output."""
        self._memory.add_user_message(user_input)
        self._memory.add_ai_message(output)

        if self._chat_history_service:
            try:
                await self._chat_history_service.save_user_message(ctx, user_input)
                await self._chat_history_service.save_assistant_message(ctx, output)
            except Exception:
                logger.exception(
                    "Failed to persist chat messages — continuing without save",
                )

        await self._persist_recipe_cache(ctx)
        logger.debug("Agent response: %s", output[:100])
        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_react_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    """Build a LangGraph ReAct agent, tolerant of the prompt kwarg's name.

    The kwarg that injects a system prompt was renamed across LangGraph
    versions (`messages_modifier` → `state_modifier` → `prompt`).
    """
    for kwarg in ("prompt", "state_modifier", "messages_modifier"):
        try:
            return create_react_agent(llm, tools, **{kwarg: system_prompt})
        except TypeError:
            continue
    # Last resort: no system prompt injection supported by this version.
    logger.warning("create_react_agent: no supported system-prompt kwarg found")
    return create_react_agent(llm, tools)


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


async def _route_image_if_present(
    user_input: str,
    ctx: SessionContext,
    tools: ToolRegistry,
) -> str | None:
    """Fast-path: if user_input contains [IMAGE:...] invoke analyze_image directly.

    Returns the tool output string when an image was found and the tool exists,
    otherwise None so the normal LLM routing proceeds.
    """
    path_match = re.search(r"\[IMAGE:([^\]]+)\]", user_input, re.IGNORECASE)
    if path_match is None:
        return None
    if "analyze_image" not in tools.names():
        return None

    image_path = path_match.group(1).strip()
    logger.info(
        "Image fast-path triggered for user %d — invoking analyze_image directly (path=%s)",
        ctx.user_id, image_path,
    )
    try:
        return await tools.invoke("analyze_image", ctx, image_path=image_path)
    except Exception:
        logger.exception("Image fast-path failed — falling back to normal LLM routing")
        return None


# ---------------------------------------------------------------------------
# Recipe cache serialization (for cross-session persistence)
# ---------------------------------------------------------------------------

def _serialize_recipes(recipes: list[Recipe]) -> str:
    """Convert a list of Recipe objects to a JSON string for DB storage."""
    return json.dumps([
        {
            "name": r.name,
            "ingredients": list(r.ingredients),
            "why_recommended": r.why_recommended,
            "servings": r.servings,
            "prep_time": r.prep_time,
            "cook_instructions": r.cook_instructions,
            "nutrition": {
                "calories": r.nutrition.calories,
                "protein_g": r.nutrition.protein_g,
                "carbs_g": r.nutrition.carbs_g,
                "fat_g": r.nutrition.fat_g,
                "fiber_g": r.nutrition.fiber_g,
                "sodium_mg": r.nutrition.sodium_mg,
                "sugar_g": r.nutrition.sugar_g,
            },
        }
        for r in recipes
    ])


def _deserialize_recipes(json_str: str) -> list[Recipe]:
    """Reconstruct a list of Recipe objects from a JSON string."""
    data = json.loads(json_str)
    result: list[Recipe] = []
    for d in data:
        n = d.get("nutrition", {})
        result.append(Recipe(
            name=d.get("name", ""),
            ingredients=d.get("ingredients", []),
            why_recommended=d.get("why_recommended", ""),
            servings=d.get("servings", 0),
            prep_time=d.get("prep_time", ""),
            cook_instructions=d.get("cook_instructions", ""),
            nutrition=NutritionValues(
                calories=n.get("calories"),
                protein_g=n.get("protein_g"),
                carbs_g=n.get("carbs_g"),
                fat_g=n.get("fat_g"),
                fiber_g=n.get("fiber_g"),
                sodium_mg=n.get("sodium_mg"),
                sugar_g=n.get("sugar_g"),
            ),
        ))
    return result
