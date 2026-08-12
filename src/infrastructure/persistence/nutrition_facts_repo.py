"""
infrastructure.persistence.nutrition_facts_repo - Food composition lookups.

Implements NutritionFactsRepository. Replaces the 36 MB nutrition FAISS index:
name lookups are exact with an FTS5 fallback for near-misses, and "highest in
iron" is an ORDER BY rather than an approximate nearest-neighbour search.
"""

from __future__ import annotations

import logging
import re

from domain.models import NutritionFact
from infrastructure.persistence.connection import AsyncSQLiteConnection
from infrastructure.persistence.nutrition_facts_seed import (
    FTS_TABLE,
    TABLE,
    UNITS_TABLE,
)

logger = logging.getLogger(__name__)

# Identifiers reaching SQL are validated against the table's real columns, but
# reject anything obviously unsafe before it gets that far.
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_NON_NUTRIENT_COLUMNS = {"id", "name", "serving_size"}


class SQLiteNutritionFactsRepository:
    """Async SQLite implementation of NutritionFactsRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection
        self._units: dict[str, str] | None = None
        self._columns: set[str] | None = None

    # ------------------------------------------------------------------
    # Schema introspection (cached — reference data does not change at runtime)
    # ------------------------------------------------------------------

    async def _load_schema(self, conn) -> None:
        if self._units is not None and self._columns is not None:
            return
        cursor = await conn.execute(f"PRAGMA table_info({TABLE})")
        self._columns = {
            row["name"] for row in await cursor.fetchall()
        } - _NON_NUTRIENT_COLUMNS

        cursor = await conn.execute(f"SELECT column_name, unit FROM {UNITS_TABLE}")
        self._units = {
            row["column_name"]: row["unit"] or "" for row in await cursor.fetchall()
        }

    async def available_nutrients(self) -> list[str]:
        """Every nutrient column, for validating a caller's request."""
        async with self._conn.acquire() as conn:
            await self._load_schema(conn)
        return sorted(self._columns or set())

    def _to_fact(self, row) -> NutritionFact:
        nutrients = {
            key: row[key]
            for key in row.keys()
            if key not in _NON_NUTRIENT_COLUMNS and row[key] is not None
        }
        return NutritionFact(
            name=row["name"],
            serving_size=row["serving_size"] or "",
            nutrients=nutrients,
            units={k: (self._units or {}).get(k, "") for k in nutrients},
        )

    # ------------------------------------------------------------------
    # Port implementation
    # ------------------------------------------------------------------

    async def lookup(self, food_name: str) -> NutritionFact | None:
        """Exact match (case-insensitive), then fuzzy. None if nothing matches."""
        name = (food_name or "").strip()
        if not name:
            return None

        async with self._conn.acquire() as conn:
            await self._load_schema(conn)
            cursor = await conn.execute(
                f"SELECT * FROM {TABLE} WHERE name = ? COLLATE NOCASE LIMIT 1",
                (name,),
            )
            row = await cursor.fetchone()
            if row is not None:
                return self._to_fact(row)

        matches = await self.search(name, limit=1)
        return matches[0] if matches else None

    @staticmethod
    def _rank_key(name: str, term: str) -> tuple:
        """Order candidates the way a person would expect.

        Raw bm25 puts 'Oil, avocado' above 'Avocados, California, raw' for the
        query "avocado", because it favours shorter documents. Someone asking
        about avocado means the fruit. Prefer an exact name, then a name that
        starts with the query, then the shortest — which keeps base ingredients
        ahead of branded products built from them.
        """
        lowered = name.lower()
        return (
            lowered != term,
            not lowered.startswith(term),
            len(name),
        )

    async def search(self, query: str, limit: int = 5) -> list[NutritionFact]:
        """Name search over FTS5 (token and prefix matching), falling back to LIKE.

        This matches words, not meanings: "corn" finds Cornstarch and "thyme"
        finds "Spices, dried, thyme", but a synonym like "cornflour" finds
        nothing, because no lexical index relates it to "Cornstarch". Callers
        should treat an empty result as "unknown food", not "no such food".
        """
        term = (query or "").strip()
        if not term:
            return []
        lowered = term.lower()

        async with self._conn.acquire() as conn:
            await self._load_schema(conn)

            # Over-fetch so the reordering below has candidates to choose from;
            # bm25 alone picks the wrong one often enough to matter.
            pool = max(limit * 5, 25)

            # Quote the term so FTS5 reads it as a phrase rather than syntax,
            # and prefix-match so partial names still hit.
            safe = term.replace('"', " ").strip()
            rows = []
            if safe:
                try:
                    cursor = await conn.execute(
                        f"""SELECT f.* FROM {FTS_TABLE} fts
                            JOIN {TABLE} f ON f.id = fts.rowid
                            WHERE {FTS_TABLE} MATCH ?
                            ORDER BY bm25({FTS_TABLE})
                            LIMIT ?""",
                        (f'"{safe}"*', pool),
                    )
                    rows = await cursor.fetchall()
                except Exception as e:
                    logger.debug("FTS query failed for %r (%s) — falling back to LIKE", term, e)

            if not rows:
                cursor = await conn.execute(
                    f"SELECT * FROM {TABLE} WHERE name LIKE ? ORDER BY length(name) LIMIT ?",
                    (f"%{term}%", pool),
                )
                rows = await cursor.fetchall()

            ranked = sorted(rows, key=lambda r: self._rank_key(r["name"], lowered))
            return [self._to_fact(r) for r in ranked[:limit]]

    async def top_by_nutrient(self, nutrient: str, limit: int = 10) -> list[NutritionFact]:
        """Foods highest in one nutrient. Exact, where the vector index guessed."""
        column = (nutrient or "").strip().lower()
        if not _IDENTIFIER_RE.match(column):
            raise ValueError(f"Invalid nutrient name: {nutrient!r}")

        async with self._conn.acquire() as conn:
            await self._load_schema(conn)
            if column not in (self._columns or set()):
                raise ValueError(f"Unknown nutrient: {nutrient!r}")

            # `column` is validated against the live schema above, so this
            # interpolation cannot inject; SQLite has no placeholder for
            # identifiers.
            cursor = await conn.execute(
                f"""SELECT * FROM {TABLE}
                    WHERE {column} IS NOT NULL
                    ORDER BY {column} DESC
                    LIMIT ?""",
                (limit,),
            )
            return [self._to_fact(r) for r in await cursor.fetchall()]
