"""
Tests for the nutrition_facts table that replaced the 36 MB nutrition FAISS index.

Covers the three things that could silently corrupt answers: unit normalisation,
SQL identifier handling in ORDER BY, and inexact name matches being reported as
if they were exact.
"""

from __future__ import annotations

import csv

import pytest

from agent.tools.lookup_nutrition import (
    LookupNutritionTool,
    normalize_nutrient,
)
from application.context import SessionContext
from infrastructure.persistence.connection import AsyncSQLiteConnection
from infrastructure.persistence.nutrition_facts_repo import (
    SQLiteNutritionFactsRepository,
)
from infrastructure.persistence.nutrition_facts_seed import (
    convert,
    parse_value,
    read_nutrition_csv,
    seed_nutrition_facts,
)

ROWS = [
    # name,        serving, calories, protein, sodium,    irom,      vitamin_a
    ("Cornstarch", "100 g", "381",    "0.26 g", "9.00 mg", "0.47 mg", "0.00 IU"),
    ("Avocado",    "100 g", "160",    "2.00 g", "7.00 mg", "0.55 mg", "146 IU"),
    ("Thyme",      "100 g", "101",    "5.56 g", "9.00 mg", "17.45 mg", "4751 IU"),
    # grams in a milligram column — must be converted, not stored raw
    ("Oddity",     "100 g", "50",     "1.00 g", "0.002 g", "0.10 mg", "12 mcg"),
]
HEADER = ["name", "serving_size", "calories", "protein", "sodium", "irom", "vitamin_a"]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "cleaned_nutrition.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(ROWS)
    return path


@pytest.fixture
async def repo(tmp_path, csv_path):
    conn_mgr = AsyncSQLiteConnection(str(tmp_path / "test.db"))
    async with conn_mgr.acquire() as conn:
        await seed_nutrition_facts(conn, csv_path)
    return SQLiteNutritionFactsRepository(conn_mgr)


# ---------------------------------------------------------------------------
# Unit parsing — where a silent 1000x error would live
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9.00 mg", (9.0, "mg")),
        ("0.1g", (0.1, "g")),
        ("0", (0.0, None)),
        ("", (None, None)),
        ("  4751 IU ", (4751.0, "iu")),
        ("not a number", (None, None)),
    ],
)
def test_parse_value(raw, expected):
    assert parse_value(raw) == expected


def test_grams_convert_to_milligrams():
    assert convert(0.002, "g", "mg") == pytest.approx(2.0)


def test_bare_number_is_taken_at_face_value():
    """Unitless cells in this dataset are all zeros — zero in any unit."""
    assert convert(0.0, None, "mg") == 0.0


def test_incompatible_units_are_not_guessed():
    """IU -> mcg depends on the compound, so it must yield None, not a number."""
    assert convert(12.0, "mcg", "iu") is None
    assert convert(12.0, "iu", "mg") is None


def test_unconvertible_cell_becomes_null_not_wrong(csv_path):
    """The mcg row in an IU column must be dropped rather than stored raw."""
    _, units, rows = read_nutrition_csv(csv_path)
    assert units["vitamin_a"] == "iu"

    oddity = next(r for r in rows if r["name"] == "Oddity")
    assert oddity["vitamin_a"] is None
    # ...while its grams-in-a-mg-column value was converted, not dropped
    assert oddity["sodium"] == pytest.approx(2.0)


def test_rejects_unsafe_column_names(tmp_path):
    """The schema is generated from the header, so this is the injection boundary."""
    bad = tmp_path / "bad.csv"
    bad.write_text('name,"evil; DROP TABLE x"\nfoo,1\n')
    with pytest.raises(ValueError, match="Unsafe column names"):
        read_nutrition_csv(bad)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

async def test_exact_lookup_returns_typed_values(repo):
    fact = await repo.lookup("Cornstarch")
    assert fact is not None
    assert fact.name == "Cornstarch"
    assert fact.get("calories") == 381
    assert fact.get("protein") == pytest.approx(0.26)
    assert fact.unit_for("protein") == "g"
    assert fact.format_amount("sodium") == "9 mg"


async def test_lookup_is_case_insensitive(repo):
    assert (await repo.lookup("cornstarch")).name == "Cornstarch"


async def test_unknown_food_returns_none(repo):
    assert await repo.lookup("wildebeest tartare") is None


async def test_top_by_nutrient_orders_descending(repo):
    top = await repo.top_by_nutrient("irom", limit=3)
    assert [f.name for f in top] == ["Thyme", "Avocado", "Cornstarch"]


async def test_top_by_nutrient_rejects_unknown_column(repo):
    with pytest.raises(ValueError, match="Unknown nutrient"):
        await repo.top_by_nutrient("unobtainium")


async def test_top_by_nutrient_rejects_sql_injection(repo):
    """`nutrient` is interpolated into ORDER BY, so it must be validated."""
    with pytest.raises(ValueError):
        await repo.top_by_nutrient("irom; DROP TABLE nutrition_facts")


async def test_seed_is_idempotent(tmp_path, csv_path):
    conn_mgr = AsyncSQLiteConnection(str(tmp_path / "twice.db"))
    async with conn_mgr.acquire() as conn:
        first = await seed_nutrition_facts(conn, csv_path)
        second = await seed_nutrition_facts(conn, csv_path)
    assert first == second == len(ROWS)


async def test_missing_csv_degrades_quietly(tmp_path):
    conn_mgr = AsyncSQLiteConnection(str(tmp_path / "none.db"))
    async with conn_mgr.acquire() as conn:
        assert await seed_nutrition_facts(conn, tmp_path / "nope.csv") == 0


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------

def test_nutrient_aliases_cover_the_dataset_typos():
    """The CSV ships 'irom' and 'zink'; users type 'iron' and 'zinc'."""
    assert normalize_nutrient("iron") == "irom"
    assert normalize_nutrient("zinc") == "zink"
    assert normalize_nutrient("vitamin c") == "vitamin_c"
    assert normalize_nutrient("Saturated Fat") == "saturated_fat"


async def test_tool_reports_macros_for_an_exact_match(repo):
    result = await LookupNutritionTool(repo).execute(
        SessionContext(user_id=1, conversation_id="t"), food="Cornstarch",
    )
    assert "Cornstarch" in result.output
    assert "381" in result.output
    assert "Closest match" not in result.output


async def test_tool_flags_an_inexact_match(repo):
    """Answering about the wrong food silently is the failure mode to avoid."""
    result = await LookupNutritionTool(repo).execute(
        SessionContext(user_id=1, conversation_id="t"), food="corn",
    )
    assert "Closest match for 'corn'" in result.output
    assert "Cornstarch" in result.output


async def test_tool_answers_a_single_nutrient(repo):
    result = await LookupNutritionTool(repo).execute(
        SessionContext(user_id=1, conversation_id="t"), food="Thyme", nutrient="iron",
    )
    assert "17.45 mg" in result.output


async def test_tool_ranks_foods_by_nutrient(repo):
    result = await LookupNutritionTool(repo).execute(
        SessionContext(user_id=1, conversation_id="t"), nutrient="iron",
    )
    assert "Thyme" in result.output
    assert result.output.index("Thyme") < result.output.index("Avocado")


async def test_tool_handles_unknown_food_without_crashing(repo):
    result = await LookupNutritionTool(repo).execute(
        SessionContext(user_id=1, conversation_id="t"), food="wildebeest tartare",
    )
    assert "not in the nutrition database" in result.output
