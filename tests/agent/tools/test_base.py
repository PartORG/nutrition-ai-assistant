"""
Unit tests for agent/tools/base.py — ToolResult dataclass and BaseTool interface.
"""

import pytest
from agent.tools.base import ToolResult, BaseTool
from application.context import SessionContext
from pydantic import BaseModel


class TestToolResult:
    def test_output_required(self):
        tr = ToolResult(output="Hello")
        assert tr.output == "Hello"

    def test_data_defaults_to_none(self):
        tr = ToolResult(output="Hi")
        assert tr.data is None

    def test_store_as_defaults_to_none(self):
        tr = ToolResult(output="Hi")
        assert tr.store_as is None

    def test_with_all_fields(self):
        tr = ToolResult(output="recipes", data={"key": "val"}, store_as="last_recommendations")
        assert tr.data == {"key": "val"}
        assert tr.store_as == "last_recommendations"

    def test_output_can_be_multiline_markdown(self):
        md = "## Title\n- item 1\n- item 2"
        tr = ToolResult(output=md)
        assert "## Title" in tr.output


class TestBaseToolIsAbstract:
    def test_cannot_instantiate_directly(self):
        """BaseTool is abstract — instantiation without subclassing must fail."""
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_execute(self):
        """A subclass that forgets execute() cannot be instantiated."""
        class IncompleteToolNoExecute(BaseTool):
            name = "incomplete"
            description = "missing execute"

            def get_schema(self):
                return BaseModel

        with pytest.raises(TypeError):
            IncompleteToolNoExecute()

    def test_concrete_subclass_must_implement_get_schema(self):
        class IncompleteToolNoSchema(BaseTool):
            name = "incomplete"
            description = "missing schema"

            async def execute(self, ctx, **kwargs):
                return ToolResult(output="ok")

        with pytest.raises(TypeError):
            IncompleteToolNoSchema()

    def test_valid_concrete_subclass_instantiates(self):
        class MySchema(BaseModel):
            pass

        class ConcreteTool(BaseTool):
            name = "concrete"
            description = "a real tool"

            def get_schema(self):
                return MySchema

            async def execute(self, ctx: SessionContext, **kwargs) -> ToolResult:
                return ToolResult(output="done")

        tool = ConcreteTool()
        assert tool.name == "concrete"

    async def test_concrete_tool_execute_returns_tool_result(self):
        class EchoSchema(BaseModel):
            pass

        class EchoTool(BaseTool):
            name = "echo"
            description = "echoes"

            def get_schema(self):
                return EchoSchema

            async def execute(self, ctx: SessionContext, **kwargs) -> ToolResult:
                return ToolResult(output="echoed", data=42)

        ctx = SessionContext(user_id=1, conversation_id="c")
        result = await EchoTool().execute(ctx)
        assert result.output == "echoed"
        assert result.data == 42
