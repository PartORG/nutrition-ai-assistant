"""
infrastructure.persistence.connection - Async SQLite connection manager.

Wraps aiosqlite with a context manager pattern matching the original
UserDBHandler._get_connection() but async.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

logger = logging.getLogger(__name__)


class AsyncSQLiteConnection:
    """Async SQLite connection provider with auto-commit/rollback."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield an async SQLite connection with FK support.

        Commits on success, rolls back on exception.
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = aiosqlite.Row
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                logger.exception("Database operation failed, transaction rolled back.")
                raise
