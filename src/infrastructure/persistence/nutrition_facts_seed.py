"""
infrastructure.persistence.nutrition_facts_seed - Load food composition into SQL.

`data/cleaned_nutrition.csv` is 8,789 rows x 76 columns of exact numeric
nutrients keyed by food name, with no duplicate names. That is a relational
table, and it was previously embedded into a 36 MB FAISS index — where each row
rendered as ~25 lines of near-identical template, so the only token
distinguishing one food from another was its name. Nothing ever filtered on the
typed metadata the loader extracted.

This loads it into `nutrition_facts` instead, with an FTS5 index for fuzzy name
matching ("cornflour" -> "Cornstarch"). Lookups become exact, and "foods highest
in iron" becomes ORDER BY rather than an approximate nearest-neighbour search.

Units live inside the CSV values ("9.00 mg", "0.1g", "0"). They are stripped and
normalised per column here, and the canonical unit is recorded in
`nutrition_facts_units` so no caller has to guess.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

TABLE = "nutrition_facts"
UNITS_TABLE = "nutrition_facts_units"
FTS_TABLE = "nutrition_facts_fts"

# Text columns kept verbatim; everything else becomes REAL.
_TEXT_COLUMNS = {"name", "serving_size"}

_VALUE_RE = re.compile(r"^\s*(-?[\d.]+)\s*([a-zA-Z]*)\s*$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Mass units are interconvertible; IU is not — it depends on the compound, so a
# cell whose unit cannot be converted to its column's canonical unit is stored
# as NULL rather than as a confidently wrong number.
_TO_MILLIGRAMS = {"g": 1000.0, "mg": 1.0, "mcg": 0.001, "ug": 0.001}


def parse_value(raw: str) -> tuple[float | None, str | None]:
    """Split a CSV cell into (number, unit). Returns (None, None) if unparseable."""
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    m = _VALUE_RE.match(text)
    if not m:
        return None, None
    try:
        value = float(m.group(1))
    except ValueError:
        return None, None
    unit = (m.group(2) or "").lower() or None
    return value, unit


def convert(value: float, from_unit: str | None, to_unit: str | None) -> float | None:
    """Convert between units, or None when the conversion is not meaningful.

    A bare number (no unit) is taken at face value — in this dataset those are
    all zeros, which are zero in any unit.
    """
    if from_unit == to_unit or from_unit is None:
        return value
    if to_unit is None:
        return value
    if from_unit in _TO_MILLIGRAMS and to_unit in _TO_MILLIGRAMS:
        return value * _TO_MILLIGRAMS[from_unit] / _TO_MILLIGRAMS[to_unit]
    return None


def _canonical_units(rows: list[dict[str, str]], columns: list[str]) -> dict[str, str | None]:
    """Pick each column's dominant unit — the one most of its cells carry."""
    units: dict[str, str | None] = {}
    for col in columns:
        seen: Counter[str] = Counter()
        for row in rows:
            _, unit = parse_value(row.get(col, ""))
            if unit:
                seen[unit] += 1
        units[col] = seen.most_common(1)[0][0] if seen else None
    return units


def read_nutrition_csv(csv_path: Path) -> tuple[list[str], dict[str, str | None], list[dict]]:
    """Parse the CSV into (numeric_columns, units, normalised rows).

    Raises ValueError if a column name is not a safe SQL identifier — the schema
    is generated from the header, so this is the boundary that keeps it safe.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")

    header = [c.strip() for c in rows[0]]
    bad = [c for c in header if not _IDENTIFIER_RE.match(c)]
    if bad:
        raise ValueError(f"Unsafe column names in {csv_path}: {bad}")
    if "name" not in header:
        raise ValueError(f"{csv_path} has no 'name' column")

    numeric_columns = [c for c in header if c not in _TEXT_COLUMNS]
    units = _canonical_units(rows, numeric_columns)

    normalised: list[dict] = []
    dropped: Counter[str] = Counter()
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        record: dict = {
            "name": name,
            "serving_size": (row.get("serving_size") or "").strip() or None,
        }
        for col in numeric_columns:
            value, unit = parse_value(row.get(col, ""))
            if value is None:
                record[col] = None
                continue
            converted = convert(value, unit, units[col])
            if converted is None:
                dropped[col] += 1
            record[col] = converted
        normalised.append(record)

    for col, count in dropped.items():
        logger.warning(
            "%s: %d value(s) dropped — unit not convertible to '%s' "
            "(e.g. IU cannot be converted to a mass unit without knowing the compound)",
            col, count, units[col],
        )

    return numeric_columns, units, normalised


def build_schema_sql(numeric_columns: list[str]) -> list[str]:
    """DDL for the facts table, its unit map, and the FTS index over names."""
    columns = ",\n        ".join(f"{c} REAL" for c in numeric_columns)
    return [
        f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        serving_size TEXT,
        {columns}
    )""",
        f"""CREATE TABLE IF NOT EXISTS {UNITS_TABLE} (
        column_name TEXT PRIMARY KEY,
        unit TEXT
    )""",
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE}
        USING fts5(name, content='{TABLE}', content_rowid='id')""",
    ]


async def seed_nutrition_facts(
    conn: aiosqlite.Connection,
    csv_path: Path,
    *,
    force: bool = False,
) -> int:
    """Create and populate nutrition_facts. Idempotent unless force=True.

    Returns the number of rows in the table afterwards.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        logger.warning("Nutrition CSV not found at %s — nutrition lookup disabled", csv_path)
        return 0

    numeric_columns, units, rows = read_nutrition_csv(csv_path)

    for statement in build_schema_sql(numeric_columns):
        await conn.execute(statement)

    cursor = await conn.execute(f"SELECT COUNT(*) FROM {TABLE}")
    existing = (await cursor.fetchone())[0]
    if existing and not force:
        logger.info("nutrition_facts already populated (%d rows) — skipping seed", existing)
        return existing

    if force:
        await conn.execute(f"DELETE FROM {TABLE}")

    columns = ["name", "serving_size", *numeric_columns]
    placeholders = ", ".join("?" for _ in columns)
    await conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE} ({', '.join(columns)}) VALUES ({placeholders})",
        [tuple(row.get(c) for c in columns) for row in rows],
    )

    await conn.executemany(
        f"INSERT OR REPLACE INTO {UNITS_TABLE} (column_name, unit) VALUES (?, ?)",
        list(units.items()),
    )

    # External-content FTS: rebuild rather than trigger-syncing, since this table
    # is reference data that changes only on a reseed.
    await conn.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES('rebuild')")
    await conn.commit()

    logger.info("Seeded %d nutrition facts from %s", len(rows), csv_path.name)
    return len(rows)
