"""
Unit tests for domain/entities.py — persistence-aware dataclasses.
No external dependencies.
"""

from domain.entities import (
    User,
    Authentication,
    MedicalAdvice,
    RecipeHistory,
    NutritionHistory,
    Conversation,
    ChatMessage,
    UserProfileHistory,
)


class TestUser:
    def test_defaults(self):
        u = User()
        assert u.id is None
        assert u.name == ""
        assert u.age == 0

    def test_with_values(self):
        u = User(id=1, name="Alice", surname="Smith", age=30, gender="female")
        assert u.id == 1
        assert u.name == "Alice"
        assert u.age == 30


class TestAuthentication:
    def test_defaults(self):
        a = Authentication()
        assert a.id is None
        assert a.login == ""
        assert a.role == ""
        assert a.user_id is None

    def test_with_values(self):
        a = Authentication(login="alice@example.com", password="hashed", role="user", user_id=5)
        assert a.login == "alice@example.com"
        assert a.role == "user"
        assert a.user_id == 5


class TestMedicalAdvice:
    def test_defaults(self):
        m = MedicalAdvice()
        assert m.id is None
        assert m.health_condition == ""
        assert m.avoid == ""

    def test_with_values(self):
        m = MedicalAdvice(health_condition="diabetes", avoid="sugar", user_id=2)
        assert m.health_condition == "diabetes"
        assert m.user_id == 2


class TestRecipeHistory:
    def test_defaults(self):
        r = RecipeHistory()
        assert r.id is None
        assert r.recipe_name == ""
        assert r.servings == 0
        assert r.rating is None

    def test_with_values(self):
        r = RecipeHistory(recipe_name="Salad", servings=2, rating=4, user_id=1)
        assert r.recipe_name == "Salad"
        assert r.servings == 2
        assert r.rating == 4


class TestNutritionHistory:
    def test_float_defaults(self):
        n = NutritionHistory()
        assert n.calories == 0.0
        assert n.protein == 0.0
        assert n.sugar == 0.0

    def test_with_values(self):
        n = NutritionHistory(calories=350.0, protein=28.5, user_id=1)
        assert n.calories == 350.0
        assert n.protein == 28.5


class TestConversation:
    def test_defaults(self):
        c = Conversation()
        assert c.id is None
        assert c.conversation_id == ""
        assert c.title == ""

    def test_with_values(self):
        c = Conversation(conversation_id="abc-123", title="My Chat", user_id=7)
        assert c.conversation_id == "abc-123"
        assert c.title == "My Chat"


class TestChatMessage:
    def test_defaults(self):
        m = ChatMessage()
        assert m.role == ""
        assert m.content == ""

    def test_with_values(self):
        m = ChatMessage(role="user", content="Hello!", conversation_id="abc", user_id=1)
        assert m.role == "user"
        assert m.content == "Hello!"


class TestUserProfileHistory:
    def test_defaults(self):
        p = UserProfileHistory()
        assert p.id is None
        assert p.preferences == ""
        assert p.restrictions == ""

    def test_with_values(self):
        p = UserProfileHistory(preferences="vegan", health_condition="diabetes", user_id=3)
        assert p.preferences == "vegan"
        assert p.health_condition == "diabetes"
