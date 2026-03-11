"""
Unit tests for application/context.py — SessionContext dataclass.
No external dependencies.
"""

from application.context import SessionContext


class TestSessionContext:
    def test_creates_with_required_fields(self):
        ctx = SessionContext(user_id=42, conversation_id="conv-1")
        assert ctx.user_id == 42
        assert ctx.conversation_id == "conv-1"

    def test_request_id_auto_generated(self):
        ctx = SessionContext(user_id=1, conversation_id="c")
        assert len(ctx.request_id) == 32  # uuid4().hex is 32 hex chars

    def test_two_instances_get_different_request_ids(self):
        ctx1 = SessionContext(user_id=1, conversation_id="c")
        ctx2 = SessionContext(user_id=1, conversation_id="c")
        assert ctx1.request_id != ctx2.request_id

    def test_scratch_starts_empty(self):
        ctx = SessionContext(user_id=1, conversation_id="c")
        assert ctx.scratch == {}

    def test_new_request_resets_request_id(self):
        ctx = SessionContext(user_id=1, conversation_id="c")
        old_id = ctx.request_id
        ctx.new_request()
        assert ctx.request_id != old_id

    def test_new_request_preserves_scratch(self):
        ctx = SessionContext(user_id=1, conversation_id="c")
        ctx.scratch["last_recommendations"] = {"recipe": "Salad"}
        ctx.new_request()
        assert ctx.scratch.get("last_recommendations") == {"recipe": "Salad"}

    def test_new_request_preserves_user_id_and_conversation(self):
        ctx = SessionContext(user_id=99, conversation_id="conv-xyz")
        ctx.new_request()
        assert ctx.user_id == 99
        assert ctx.conversation_id == "conv-xyz"

    def test_user_data_defaults_to_empty_dict(self):
        ctx = SessionContext(user_id=1, conversation_id="c")
        assert ctx.user_data == {}

    def test_user_data_can_be_set(self):
        data = {"name": "Alice", "restrictions": ["vegan"]}
        ctx = SessionContext(user_id=1, conversation_id="c", user_data=data)
        assert ctx.user_data["name"] == "Alice"
