"""
Integration tests for infrastructure/persistence/migrations.py.

Uses a temporary file-based SQLite database via pytest's tmp_path fixture.

WHY NOT :memory:?
SQLite's `:memory:` creates a brand-new empty database on every
`aiosqlite.connect()` call.  Since AsyncSQLiteConnection calls
`connect()` each time `.acquire()` is used, migrations and
assertions would target different databases and the assertions
would always see empty tables.  A real (but temporary) file avoids
this: the same on-disk DB is opened and closed between acquire() calls.

These verify that:
- All tables are created by run_migrations()
- run_migrations() is idempotent (safe to call multiple times)
- The recipe_cache column patch is applied to conversations
- The rating column patch is applied to recipe_history
"""

from __future__ import annotations

import pytest
import aiosqlite

from infrastructure.persistence.connection import AsyncSQLiteConnection
from infrastructure.persistence.migrations import run_migrations


# ---------------------------------------------------------------------------
# Fixture: temp-file connection
# ---------------------------------------------------------------------------

@pytest.fixture
async def in_memory_connection(tmp_path) -> AsyncSQLiteConnection:
    """Return an AsyncSQLiteConnection backed by a temp file.

    Using a real file (tmp_path) instead of :memory: because SQLite
    creates a separate empty database for each connect(':memory:') call,
    which would make migrations and assertions target different databases.
    """
    db_file = str(tmp_path / "test.db")
    return AsyncSQLiteConnection(db_file)


# ---------------------------------------------------------------------------
# Helper: introspect a table's columns
# ---------------------------------------------------------------------------

async def get_columns(conn, table: str) -> set[str]:
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def get_tables(conn) -> set[str]:
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    rows = await cursor.fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunMigrations:
    async def test_all_tables_created(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            tables = await get_tables(conn)

        expected = {
            "users",
            "authentication",
            "medical_advice",
            "user_profile_history",
            "recipe_history",
            "nutrition_history",
            "conversations",
            "chat_messages",
        }
        assert expected.issubset(tables)

    async def test_idempotent_second_call_does_not_raise(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        # Should not raise 'table already exists' or similar
        await run_migrations(in_memory_connection)

    async def test_recipe_cache_column_added_to_conversations(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            cols = await get_columns(conn, "conversations")
        assert "recipe_cache" in cols

    async def test_rating_column_added_to_recipe_history(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            cols = await get_columns(conn, "recipe_history")
        assert "rating" in cols

    async def test_users_table_has_expected_columns(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            cols = await get_columns(conn, "users")
        for col in ("id", "name", "surname", "user_name", "age", "gender"):
            assert col in cols, f"Missing column: {col}"

    async def test_authentication_table_has_expected_columns(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            cols = await get_columns(conn, "authentication")
        for col in ("id", "login", "password", "role", "user_id"):
            assert col in cols, f"Missing column: {col}"

    async def test_chat_messages_table_has_expected_columns(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            cols = await get_columns(conn, "chat_messages")
        for col in ("id", "user_id", "conversation_id", "role", "content"):
            assert col in cols, f"Missing column: {col}"

    async def test_nutrition_history_has_macro_columns(self, in_memory_connection):
        await run_migrations(in_memory_connection)
        async with in_memory_connection.acquire() as conn:
            cols = await get_columns(conn, "nutrition_history")
        for col in ("calories", "protein", "fat", "carbohydrates", "fiber", "sugar", "sodium"):
            assert col in cols, f"Missing column: {col}"
