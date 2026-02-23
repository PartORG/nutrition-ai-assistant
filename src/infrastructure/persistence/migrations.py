"""
infrastructure.persistence.migrations - Database schema creation.

Extracted from db.py create_*_table() methods. Called once at startup
by the adapter or factory.
"""

from __future__ import annotations

import logging

from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)

_TABLES = [
    """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        surname TEXT,
        user_name TEXT UNIQUE,
        caretaker TEXT,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        age INTEGER,
        gender TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS medical_advice (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        health_condition TEXT,
        medical_advice TEXT,
        dietary_limit TEXT,
        avoid TEXT,
        dietary_constraints TEXT,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        user_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""",
    """CREATE TABLE IF NOT EXISTS authentication (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        login TEXT UNIQUE,
        password TEXT,
        role TEXT,
        created_at TEXT,
        deleted_at TEXT,
        updated_at TEXT,
        user_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""",
    """CREATE TABLE IF NOT EXISTS user_profile_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        preferences TEXT,
        user_id INTEGER,
        health_condition TEXT,
        restrictions TEXT,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""",
    """CREATE TABLE IF NOT EXISTS recipe_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        recipe_id INTEGER,
        rating INTEGER,
        recipe_name TEXT,
        cook_instructions TEXT,
        servings INTEGER,
        ingredients TEXT,
        prep_time TEXT,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""",
    """CREATE TABLE IF NOT EXISTS nutrition_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        recipe_id INTEGER,
        calories REAL,
        protein REAL,
        fat REAL,
        carbohydrates REAL,
        fiber REAL,
        sugar REAL,
        sodium REAL,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""",
    """CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        conversation_id TEXT UNIQUE,
        title TEXT,
        last_message_at TEXT,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""",
    """CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        conversation_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT,
        updated_at TEXT,
        deleted_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
    )""",
]


async def run_migrations(connection: AsyncSQLiteConnection) -> None:
    """Create all tables if they don't exist.

    Safe to call multiple times (uses IF NOT EXISTS).
    """
    async with connection.acquire() as conn:
        for ddl in _TABLES:
            await conn.execute(ddl)
    logger.info("All tables created (or already exist).")
