"""
infrastructure.llm.safety_rules - Rule tables and the prompt for safety checking.

The keyword map drives the deterministic first pass; the prompt drives the LLM
pass that catches what literal matching cannot ("prosciutto is pork"). Both are
data, edited far more often than the filter logic that consumes them, so they
live here rather than inline in safety_filter.py.
"""

from __future__ import annotations

# Restriction-to-ingredient mapping for rule-based checks
# ---------------------------------------------------------------------------

RESTRICTION_INGREDIENT_MAP: dict[str, list[str]] = {
    "vegetarian": [
        "chicken", "beef", "pork", "lamb", "fish", "meat", "turkey",
        "bacon", "ham", "sausage", "prosciutto", "salami", "anchovy",
        "shrimp", "crab", "lobster", "duck", "veal", "venison",
    ],
    "vegan": [
        "chicken", "beef", "pork", "lamb", "fish", "meat", "turkey",
        "bacon", "ham", "sausage", "prosciutto", "salami", "anchovy",
        "shrimp", "crab", "lobster", "duck", "veal", "venison",
        "egg", "milk", "cheese", "butter", "cream", "yogurt", "honey",
        "whey", "casein", "ghee", "gelatin",
    ],
    "gluten-free": [
        "wheat", "flour", "bread", "pasta", "barley", "rye",
        "couscous", "semolina", "bulgur", "farro", "spelt",
    ],
    "lactose-free": [
        "milk", "cheese", "butter", "cream", "yogurt", "whey",
        "casein", "ghee", "ice cream",
    ],
    "pescatarian": [
        "chicken", "beef", "pork", "lamb", "meat", "turkey",
        "bacon", "ham", "sausage", "prosciutto", "salami",
        "duck", "veal", "venison",
    ],
}


SAFETY_CHECK_PROMPT = """You are a medical dietary safety checker. You ONLY flag hidden ingredient issues that simple keyword matching cannot detect.

Your ONLY job: catch ingredients whose common name does NOT reveal a relevant violation. Examples:
- "prosciutto" → contains pork (flag only if pork is in Restrictions or AVOID list)
- "ghee" → contains dairy (flag only if lactose-free is in Restrictions)
- "soy sauce" → contains gluten (flag only if gluten-free is in Restrictions)

STRICT RULES — read carefully:
1. ONLY check against the exact items in "Restrictions/allergies" and "Foods to AVOID" provided below. Do NOT apply any restriction that is not explicitly listed there.
2. Do NOT flag foods in the "Foods to LIMIT" list — those are soft guidelines, never violations.
3. Do NOT flag general health concerns ("sugar is bad for diabetes", "fat is bad for cholesterol") — the recipe RAG already handles nutrition optimization.
4. Do NOT flag obvious name matches — if the ingredient name already contains the banned word, rule-based checks handled it.
5. When in doubt, return an empty issues list. False negatives are always better than false positives.
6. severity MUST always be "medium" — these are advisory flags, not hard blocks.

For EACH recipe, check ONLY:
1. Hidden avoid foods — an ingredient that IS an avoided item but its name does not make this obvious (category: "hidden_ingredient", severity: "medium")
2. Hidden restriction violations — an ingredient that violates an explicitly listed restriction in a non-obvious way (category: "restriction_violation", severity: "medium")

Return JSON:
{{
  "recipe_checks": [
    {{
      "recipe_name": "name",
      "issues": [
{{
  "category": "hidden_ingredient|restriction_violation",
  "severity": "medium",
  "description": "short explanation referencing the specific restriction or avoid item",
  "detail": "specific detail"
}}
      ]
    }}
  ]
}}

If a recipe has NO issues, return an empty issues list. The VAST MAJORITY of recipes should have NO issues.
Return ONLY valid JSON."""
