"""
Unit tests for agent/tools/nutrition_advice.py.

Covers:
- NutritionAdviceInput schema (query field, message alias)
- NutritionAdviceTool metadata (name, description, schema)
- execute() happy path: forwards query to Medical RAG, returns ToolResult
- execute() prefers ctx.scratch["original_query"] over the query argument
- execute() falls back to query argument when scratch is empty
- execute() handles Medical RAG exceptions gracefully
- execute() returns no data / no store_as (no DB writes)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.tools.nutrition_advice import NutritionAdviceTool, NutritionAdviceInput
from agent.tools.base import ToolResult
from application.context import SessionContext


# ---------------------------------------------------------------------------
# NutritionAdviceInput schema
# ---------------------------------------------------------------------------

class TestNutritionAdviceInput:
    def test_query_default_empty(self):
        inp = NutritionAdviceInput()
        assert inp.query == ""

    def test_query_set(self):
        inp = NutritionAdviceInput(query="How should someone with diabetes eat?")
        assert inp.query == "How should someone with diabetes eat?"

    def test_message_alias_accepted(self):
        inp = NutritionAdviceInput(message="test message")
        assert inp.message == "test message"

    def test_message_excluded_from_dict(self):
        inp = NutritionAdviceInput(query="q", message="m")
        d = inp.model_dump()
        assert "message" not in d
        assert d["query"] == "q"


# ---------------------------------------------------------------------------
# NutritionAdviceTool metadata
# ---------------------------------------------------------------------------

class TestNutritionAdviceToolMeta:
    @pytest.fixture
    def tool(self):
        return NutritionAdviceTool(medical_rag=MagicMock())

    def test_name(self, tool):
        assert tool.name == "nutrition_advice"

    def test_description_mentions_third_person(self, tool):
        assert "third person" in tool.description.lower()

    def test_description_excludes_personal(self, tool):
        assert "I have" in tool.description

    def test_get_schema(self, tool):
        assert tool.get_schema() is NutritionAdviceInput


# ---------------------------------------------------------------------------
# NutritionAdviceTool.execute()
# ---------------------------------------------------------------------------

class TestNutritionAdviceToolExecute:

    def _make_tool(self, ask_return: str = '{"dietary_goals": "eat healthy"}'):
        """Create a tool with a mock Medical RAG."""
        mock_rag = MagicMock()
        mock_rag.ask.return_value = ask_return
        return NutritionAdviceTool(medical_rag=mock_rag), mock_rag

    def _make_ctx(self, original_query: str | None = None) -> SessionContext:
        ctx = SessionContext(user_id=1, conversation_id="test-conv")
        if original_query is not None:
            ctx.scratch["original_query"] = original_query
        return ctx

    async def test_happy_path_returns_rag_answer(self):
        expected = '{"dietary_goals": "reduce sugar"}'
        tool, mock_rag = self._make_tool(ask_return=expected)
        ctx = self._make_ctx(original_query="How should someone with diabetes eat?")

        result = await tool.execute(ctx, query="ignored")

        assert isinstance(result, ToolResult)
        assert result.output == expected
        mock_rag.ask.assert_called_once_with("How should someone with diabetes eat?")

    async def test_prefers_original_query_from_scratch(self):
        tool, mock_rag = self._make_tool()
        ctx = self._make_ctx(original_query="original message")

        await tool.execute(ctx, query="llm rewritten query")

        mock_rag.ask.assert_called_once_with("original message")

    async def test_falls_back_to_query_param(self):
        tool, mock_rag = self._make_tool()
        ctx = self._make_ctx()  # no original_query in scratch

        await tool.execute(ctx, query="fallback query")

        mock_rag.ask.assert_called_once_with("fallback query")

    async def test_no_data_returned(self):
        tool, _ = self._make_tool()
        ctx = self._make_ctx(original_query="test")

        result = await tool.execute(ctx)

        assert result.data is None

    async def test_no_store_as(self):
        tool, _ = self._make_tool()
        ctx = self._make_ctx(original_query="test")

        result = await tool.execute(ctx)

        assert result.store_as is None

    async def test_scratch_not_modified(self):
        tool, _ = self._make_tool()
        ctx = self._make_ctx(original_query="test")
        scratch_before = dict(ctx.scratch)

        await tool.execute(ctx)

        # Only original_query should be in scratch — no new keys added
        assert set(ctx.scratch.keys()) == set(scratch_before.keys())

    async def test_rag_exception_returns_friendly_error(self):
        mock_rag = MagicMock()
        mock_rag.ask.side_effect = RuntimeError("connection failed")
        tool = NutritionAdviceTool(medical_rag=mock_rag)
        ctx = self._make_ctx(original_query="test")

        result = await tool.execute(ctx)

        assert "sorry" in result.output.lower()
        assert "try again" in result.output.lower()
        assert result.data is None
        assert result.store_as is None

    async def test_empty_query_still_calls_rag(self):
        tool, mock_rag = self._make_tool()
        ctx = self._make_ctx()  # no scratch, no query param

        await tool.execute(ctx)

        mock_rag.ask.assert_called_once_with("")
