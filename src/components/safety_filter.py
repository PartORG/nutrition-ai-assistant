"""
safety_filter - Validates recipe recommendations against user medical/dietary constraints.

This module checks the MARKDOWN TEXT output from RecipesNutritionRAG against
a user's medical restrictions, dietary preferences, allergies, and nutritional
limits. It uses a hybrid approach:

  1. LLM-based parsing  — extracts structured recipe data (ingredients, nutrition)
                          from the markdown output.
  2. Rule-based checks  — fast, deterministic checks for avoid-lists, dietary
                          restriction violations, and nutrition limit breaches.
  3. LLM semantic check — catches subtle issues that string matching cannot
                          (e.g. "prosciutto" is pork, "ghee" is dairy).

Unsafe recipes are automatically removed from the output. Borderline recipes
(with only medium-severity issues) are kept but flagged with warnings.

Usage:
    from components.safety_filter import SafetyFilter
    from components.intent_retriever import UserIntent

    sf = SafetyFilter(model_name="llama3.2")
    result = sf.check(
        recipe_markdown=rag_output,
        medical_constraints=medical_rag.get_constraints(conditions),
        user_intent=intent,
    )
    print(result.summary)
    print(result.safe_recipes_markdown)
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from settings import LLM_MODEL
from components.intent_retriever import UserIntent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SafetyVerdict(str, Enum):
    """Overall safety classification for a recipe."""
    SAFE = "safe"
    WARNING = "warning"   # borderline — has minor issues, shown but flagged
    UNSAFE = "unsafe"     # must be removed from output


@dataclass
class NutritionValues:
    """Parsed nutrition values from a recipe's markdown output."""
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    sugar_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None


@dataclass
class ParsedRecipe:
    """A single recipe extracted from the markdown output."""
    name: str = ""
    ingredients: List[str] = field(default_factory=list)
    nutrition: NutritionValues = field(default_factory=NutritionValues)
    why_recommended: str = ""


@dataclass
class SafetyIssue:
    """A single safety concern found during recipe checking.

    Attributes:
        category:    Type of issue — "allergen", "avoid_food",
                     "nutrition_limit", or "restriction_violation".
        severity:    "critical", "high", or "medium".
        description: Human-readable explanation.
        detail:      Technical detail, e.g. "sodium_mg: 1200 > max 800".
    """
    category: str
    severity: str
    description: str
    detail: str = ""


@dataclass
class RecipeSafetyVerdict:
    """Safety check result for one recipe."""
    recipe_name: str
    verdict: SafetyVerdict
    issues: List[SafetyIssue] = field(default_factory=list)
    parsed_recipe: Optional[ParsedRecipe] = None

    @property
    def is_safe(self) -> bool:
        return self.verdict == SafetyVerdict.SAFE


@dataclass
class SafetyCheckResult:
    """Aggregate result for all recipes in the markdown output.

    Attributes:
        recipe_verdicts:       Per-recipe safety verdicts.
        safe_recipes_markdown: Filtered markdown containing only safe/warning recipes.
        summary:               Human-readable summary of the check.
    """
    recipe_verdicts: List[RecipeSafetyVerdict] = field(default_factory=list)
    safe_recipes_markdown: str = ""
    summary: str = ""

    @property
    def safe_count(self) -> int:
        return sum(1 for v in self.recipe_verdicts if v.is_safe or v.verdict == SafetyVerdict.WARNING)

    @property
    def total_count(self) -> int:
        return len(self.recipe_verdicts)


# ---------------------------------------------------------------------------
# Restriction-to-ingredient mapping for rule-based checks
# ---------------------------------------------------------------------------

RESTRICTION_INGREDIENT_MAP = {
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


# ---------------------------------------------------------------------------
# SafetyFilter
# ---------------------------------------------------------------------------

class SafetyFilter:
    """Checks recipe recommendations against user medical/dietary constraints.

    This filter operates on the MARKDOWN TEXT output from RecipesNutritionRAG,
    not on raw Document objects. It uses an LLM to:
      1. Parse each recipe from the markdown into structured data.
      2. Check ingredients against avoid-lists and allergens.
      3. Check nutrition values against numeric constraints.
      4. Perform a semantic check for subtle issues rules can't catch.

    Unsafe recipes are automatically removed from the returned markdown.
    """

    def __init__(self, model_name: str = "llama3.2", debug: bool = False):
        self.debug = debug
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0,
            format="json",
        )
        self.parser = JsonOutputParser()
        self._parse_chain = self._build_parse_chain()
        self._check_chain = self._build_check_chain()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        recipe_markdown: str,
        medical_constraints: Dict[str, Any],
        user_intent: UserIntent,
    ) -> SafetyCheckResult:
        """Main entry point: parse recipes, check each one, return results.

        Args:
            recipe_markdown:    Full markdown text from RecipesNutritionRAG.ask().
            medical_constraints: Dict from MedicalRAG.get_constraints() with keys
                                 "avoid", "limit", "constraints", "notes", etc.
            user_intent:        Parsed UserIntent with restrictions, health_condition, etc.

        Returns:
            SafetyCheckResult with per-recipe verdicts and filtered markdown.
        """
        # Step 1: Parse recipes from markdown using LLM
        parsed_recipes = self._parse_recipes(recipe_markdown)

        if not parsed_recipes:
            return SafetyCheckResult(
                safe_recipes_markdown=recipe_markdown,
                summary="Could not parse any recipes from the output.",
            )

        if self.debug:
            print(f"\n[SAFETY] Parsed {len(parsed_recipes)} recipes from markdown")
            for r in parsed_recipes:
                print(f"  - {r.name}: {len(r.ingredients)} ingredients")

        # Gather constraint data
        avoid_foods = medical_constraints.get("avoid", [])
        constraint_rules = medical_constraints.get("constraints", {})
        restrictions = user_intent.restrictions_list

        # Step 2 & 3: Rule-based checks (ingredients + nutrition)
        rule_issues_per_recipe: List[List[SafetyIssue]] = []
        for recipe in parsed_recipes:
            issues = []
            issues.extend(self._check_ingredients(recipe.ingredients, avoid_foods, restrictions))
            issues.extend(self._check_nutrition(recipe.nutrition, constraint_rules))
            rule_issues_per_recipe.append(issues)

        # Step 4: LLM semantic check for subtle issues
        llm_issues_map = self._llm_semantic_check(parsed_recipes, medical_constraints, user_intent)

        # Step 5: Merge and determine verdicts
        recipe_verdicts = []
        safe_names = []

        for i, recipe in enumerate(parsed_recipes):
            all_issues = rule_issues_per_recipe[i] + llm_issues_map.get(recipe.name, [])

            # Determine verdict based on highest severity
            if any(iss.severity in ("critical", "high") for iss in all_issues):
                verdict = SafetyVerdict.UNSAFE
            elif any(iss.severity == "medium" for iss in all_issues):
                verdict = SafetyVerdict.WARNING
            else:
                verdict = SafetyVerdict.SAFE

            rv = RecipeSafetyVerdict(
                recipe_name=recipe.name,
                verdict=verdict,
                issues=all_issues,
                parsed_recipe=recipe,
            )
            recipe_verdicts.append(rv)

            # Keep safe and warning recipes in the output
            if verdict != SafetyVerdict.UNSAFE:
                safe_names.append(recipe.name)

            if self.debug:
                status = verdict.value.upper()
                print(f"  [{status}] {recipe.name} — {len(all_issues)} issue(s)")
                for iss in all_issues:
                    print(f"    [{iss.severity}] {iss.description}")

        # Step 6: Build filtered markdown with only safe recipes
        safe_markdown = self._build_safe_markdown(recipe_markdown, safe_names)

        # Build summary
        summary_parts = [f"{len(safe_names)}/{len(parsed_recipes)} recipes passed safety check."]
        for rv in recipe_verdicts:
            if rv.verdict == SafetyVerdict.UNSAFE:
                reasons = "; ".join(iss.description for iss in rv.issues[:3])
                summary_parts.append(f"  REJECTED '{rv.recipe_name}': {reasons}")
            elif rv.verdict == SafetyVerdict.WARNING:
                reasons = "; ".join(iss.description for iss in rv.issues[:3])
                summary_parts.append(f"  WARNING '{rv.recipe_name}': {reasons}")

        return SafetyCheckResult(
            recipe_verdicts=recipe_verdicts,
            safe_recipes_markdown=safe_markdown,
            summary="\n".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # LLM chain builders
    # ------------------------------------------------------------------

    def _build_parse_chain(self):
        """Build LLM chain for parsing markdown recipes into structured JSON."""
        system = """You are a recipe data extractor. Given markdown text containing recipe recommendations, extract EACH recipe into structured JSON.

Return a JSON object with key "recipes" containing a list. For EACH recipe extract:
{{
  "recipes": [
    {{
      "name": "recipe name",
      "ingredients": ["ingredient1", "ingredient2"],
      "nutrition": {{
        "calories": <number or null>,
        "protein_g": <number or null>,
        "carbs_g": <number or null>,
        "fat_g": <number or null>,
        "fiber_g": <number or null>,
        "sodium_mg": <number or null>,
        "sugar_g": <number or null>,
        "saturated_fat_g": <number or null>
      }},
      "why_recommended": "brief text"
    }}
  ]
}}

RULES:
- Extract ingredient NAMES only (strip quantities and units). E.g., "200g chicken breast" -> "chicken breast".
- Parse numeric nutrition values from the per-serving info. Use null if not present.
- If the markdown has 3 recipes, return 3 items in the list.
- Return ONLY valid JSON, nothing else."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "{recipe_markdown}")
        ])
        return prompt | self.llm | self.parser

    def _build_check_chain(self):
        """Build LLM chain for semantic safety checking of parsed recipes."""
        system = """You are a medical dietary safety checker. You ONLY flag CLEAR, DEFINITE safety violations. Do NOT flag minor or speculative concerns.

Your job is to catch issues that simple keyword matching would miss. For example:
- "prosciutto" is pork (relevant if pork is restricted)
- "ghee" is dairy (relevant if lactose-free)
- "soy sauce" contains gluten (relevant if gluten-free)

IMPORTANT RULES:
- Only flag an issue if an ingredient DIRECTLY and CLEARLY violates a stated constraint.
- Do NOT flag general health concerns (e.g., "oil is high in fat" for a diabetic patient).
- Do NOT flag LIMIT foods — those are advisory, not violations.
- Do NOT re-flag obvious ingredient matches (e.g., "peanut butter" matching "peanuts") — those are already handled by rules.
- When in doubt, do NOT flag it. Prefer false negatives over false positives.

For EACH recipe, check ONLY:
1. Hidden allergens/avoid foods that are not obvious from the ingredient name (category: "hidden_ingredient", severity: "high")
2. Ingredients that violate dietary restrictions in a non-obvious way (category: "restriction_violation", severity: "high")

Return JSON:
{{
  "recipe_checks": [
    {{
      "recipe_name": "name",
      "issues": [
        {{
          "category": "hidden_ingredient|restriction_violation",
          "severity": "high",
          "description": "human-readable explanation",
          "detail": "specific detail"
        }}
      ]
    }}
  ]
}}

If a recipe has NO issues, return an empty issues list for it. Most recipes should have NO issues.
Return ONLY valid JSON."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "PARSED RECIPES:\n{recipes_json}\n\nPATIENT CONSTRAINTS:\n- Restrictions/allergies: {restrictions}\n- Foods to AVOID: {avoid_foods}\n- Foods to LIMIT: {limit_foods}\n- Health conditions: {health_conditions}\n- Meal instructions: {instructions}\n- Nutrition limits: {nutrition_constraints}")
        ])
        return prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _parse_recipes(self, recipe_markdown: str) -> List[ParsedRecipe]:
        """Use the LLM parse chain to extract structured recipes from markdown."""
        try:
            result = self._parse_chain.invoke({"recipe_markdown": recipe_markdown})
            recipes = []
            for r in result.get("recipes", []):
                name = r.get("name", "").strip()
                ingredients = r.get("ingredients", [])
                # Skip entries with no name or no ingredients (e.g. LLM refusal messages)
                if not name or not ingredients:
                    continue
                nutrition_data = r.get("nutrition", {})
                nutrition = NutritionValues(
                    calories=nutrition_data.get("calories"),
                    protein_g=nutrition_data.get("protein_g"),
                    carbs_g=nutrition_data.get("carbs_g"),
                    fat_g=nutrition_data.get("fat_g"),
                    fiber_g=nutrition_data.get("fiber_g"),
                    sodium_mg=nutrition_data.get("sodium_mg"),
                    sugar_g=nutrition_data.get("sugar_g"),
                    saturated_fat_g=nutrition_data.get("saturated_fat_g"),
                )
                recipes.append(ParsedRecipe(
                    name=name,
                    ingredients=ingredients,
                    nutrition=nutrition,
                    why_recommended=r.get("why_recommended", ""),
                ))
            return recipes
        except Exception as e:
            if self.debug:
                print(f"[SAFETY] Error parsing recipes: {e}")
            return []

    @staticmethod
    def _word_match(word: str, text: str) -> bool:
        """Check if 'word' appears in 'text' as a whole word (not a substring).

        Examples:
            _word_match("salt", "salted butter") -> False
            _word_match("salt", "sea salt")      -> True
            _word_match("peanut", "peanut butter") -> True
            _word_match("sugar", "sugar snap peas") -> True  (word boundary)
        """
        return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))

    def _check_ingredients(
        self,
        ingredients: List[str],
        avoid_foods: List[str],
        restrictions: List[str],
    ) -> List[SafetyIssue]:
        """Rule-based ingredient checking against avoid-list and dietary restrictions."""
        issues = []
        restrictions_lower = [r.lower() for r in restrictions]

        for ingredient in ingredients:
            ing_lower = ingredient.lower()

            # Check against avoid list using word-boundary matching
            for avoid in avoid_foods:
                # Skip vague/multi-word avoid phrases like "high sugar foods"
                # Only match concrete food names
                if len(avoid.split()) > 2:
                    continue
                if self._word_match(avoid, ing_lower):
                    issues.append(SafetyIssue(
                        category="avoid_food",
                        severity="critical",
                        description=f"Ingredient '{ingredient}' matches avoid item '{avoid}'",
                        detail=f"avoid_match: '{avoid}' in '{ingredient}'",
                    ))

            # Check against dietary restrictions using the ingredient map
            for restriction in restrictions_lower:
                banned_items = RESTRICTION_INGREDIENT_MAP.get(restriction, [])
                for banned in banned_items:
                    if self._word_match(banned, ing_lower):
                        issues.append(SafetyIssue(
                            category="restriction_violation",
                            severity="critical",
                            description=f"Ingredient '{ingredient}' violates '{restriction}' restriction",
                            detail=f"banned_ingredient: '{banned}'",
                        ))
                        break  # one match per restriction is enough

        return issues

    def _check_nutrition(
        self,
        nutrition: NutritionValues,
        constraint_rules: Dict[str, Dict],
    ) -> List[SafetyIssue]:
        """Rule-based nutrition limit checking."""
        issues = []
        # Map constraint keys to NutritionValues fields
        field_map = {
            "sugar_g": nutrition.sugar_g,
            "sodium_mg": nutrition.sodium_mg,
            "fiber_g": nutrition.fiber_g,
            "protein_g": nutrition.protein_g,
            "saturated_fat_g": nutrition.saturated_fat_g,
            "calories": nutrition.calories,
            "carbs_g": nutrition.carbs_g,
            "fat_g": nutrition.fat_g,
        }

        for nutrient, rule in constraint_rules.items():
            value = field_map.get(nutrient)
            if value is None:
                continue

            max_val = rule.get("max")
            min_val = rule.get("min")

            if max_val is not None and value > max_val:
                issues.append(SafetyIssue(
                    category="nutrition_limit",
                    severity="medium",
                    description=f"{nutrient} ({value}) exceeds maximum ({max_val})",
                    detail=f"{nutrient}: {value} > max {max_val}",
                ))

            if min_val is not None and value < min_val:
                issues.append(SafetyIssue(
                    category="nutrition_limit",
                    severity="medium",
                    description=f"{nutrient} ({value}) below minimum ({min_val})",
                    detail=f"{nutrient}: {value} < min {min_val}",
                ))

        return issues

    def _llm_semantic_check(
        self,
        recipes: List[ParsedRecipe],
        medical_constraints: Dict[str, Any],
        user_intent: UserIntent,
    ) -> Dict[str, List[SafetyIssue]]:
        """LLM-based semantic check for subtle issues that rules can't catch.

        Returns a dict mapping recipe name -> list of SafetyIssue.
        """
        # Build a simplified JSON of parsed recipes for the LLM
        recipes_for_llm = []
        for r in recipes:
            recipes_for_llm.append({
                "name": r.name,
                "ingredients": r.ingredients,
                "nutrition": {
                    "calories": r.nutrition.calories,
                    "protein_g": r.nutrition.protein_g,
                    "carbs_g": r.nutrition.carbs_g,
                    "fat_g": r.nutrition.fat_g,
                    "sugar_g": r.nutrition.sugar_g,
                    "sodium_mg": r.nutrition.sodium_mg,
                },
            })

        constraint_rules = medical_constraints.get("constraints", {})
        nutrition_str = json.dumps(constraint_rules) if constraint_rules else "None"

        try:
            result = self._check_chain.invoke({
                "recipes_json": json.dumps(recipes_for_llm, indent=2),
                "restrictions": ", ".join(user_intent.restrictions_list) or "None",
                "avoid_foods": ", ".join(medical_constraints.get("avoid", [])) or "None",
                "limit_foods": ", ".join(medical_constraints.get("limit", [])) or "None",
                "health_conditions": user_intent.health_condition or "None",
                "instructions": user_intent.instructions or "None",
                "nutrition_constraints": nutrition_str,
            })

            # Parse LLM response into SafetyIssue objects
            issues_map: Dict[str, List[SafetyIssue]] = {}
            for check in result.get("recipe_checks", []):
                name = check.get("recipe_name", "")
                issues = []
                for iss in check.get("issues", []):
                    issues.append(SafetyIssue(
                        category=iss.get("category", "unknown"),
                        severity=iss.get("severity", "medium"),
                        description=iss.get("description", ""),
                        detail=iss.get("detail", ""),
                    ))
                if issues:
                    issues_map[name] = issues
            return issues_map

        except Exception as e:
            if self.debug:
                print(f"[SAFETY] LLM semantic check error: {e}")
            return {}

    @staticmethod
    def _build_safe_markdown(recipe_markdown: str, safe_recipe_names: List[str]) -> str:
        """Extract only safe/warning recipes from the original markdown output.

        Splits the markdown by common recipe separators (--- or numbered headers)
        and keeps only sections whose recipe name matches a safe recipe.
        """
        if not safe_recipe_names:
            return ""

        # Try splitting by --- separator (common in RAG output)
        sections = re.split(r'\n-{3,}\n', recipe_markdown)

        # If no splits, try splitting by numbered recipe headers like "**1." or "## 1."
        if len(sections) <= 1:
            sections = re.split(r'\n(?=(?:\*\*\d+\.|\#{1,3}\s*\d+\.))', recipe_markdown)

        safe_sections = []
        for section in sections:
            for name in safe_recipe_names:
                if name.lower() in section.lower():
                    safe_sections.append(section.strip())
                    break

        if safe_sections:
            return "\n\n---\n\n".join(safe_sections)

        # Fallback: return original if parsing fails
        return recipe_markdown


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    safety_filter = SafetyFilter(model_name=LLM_MODEL, debug=True)
    print("Safety Filter ready!\n")

    # Test with dummy markdown output (simulating RecipesNutritionRAG output)
    dummy_markdown = """
**1. Grilled Chicken Salad**

**Why this recipe:** High in protein, low in carbs, suitable for diabetes management.

**Nutritional info per serving:**
- Calories: 380
- Protein: 32g
- Carbs: 15g
- Fat: 18g
- Fiber: 6g
- Sodium: 420mg
- Sugar: 5g

**Ingredients:**
- 200g chicken breast
- 100g mixed greens
- 50g cherry tomatoes
- 30g feta cheese
- 15ml olive oil
- 10ml lemon juice

**Instructions:**
1. Grill the chicken breast at 200C for 15 minutes.
2. Chop the greens and tomatoes.
3. Combine and top with crumbled feta and dressing.

**Time:** Prep 10 min | Cook 15 min | Total 25 min

---

**2. Peanut Butter Banana Smoothie**

**Why this recipe:** Quick breakfast option with good energy.

**Nutritional info per serving:**
- Calories: 450
- Protein: 15g
- Carbs: 55g
- Fat: 20g
- Fiber: 4g
- Sodium: 150mg
- Sugar: 35g

**Ingredients:**
- 2 tablespoons peanut butter
- 1 banana
- 250ml milk
- 1 tablespoon honey
- Ice cubes

**Instructions:**
1. Blend all ingredients until smooth.
2. Serve immediately.

**Time:** Prep 5 min | Total 5 min

---

**3. Steamed Salmon with Vegetables**

**Why this recipe:** Rich in omega-3, heart-healthy, low sodium.

**Nutritional info per serving:**
- Calories: 350
- Protein: 35g
- Carbs: 12g
- Fat: 16g
- Fiber: 5g
- Sodium: 380mg
- Sugar: 3g

**Ingredients:**
- 180g salmon fillet
- 100g broccoli
- 80g carrots
- 50g zucchini
- 10ml soy sauce
- 5g ginger

**Instructions:**
1. Steam salmon at 180C for 12 minutes.
2. Steam vegetables for 8 minutes.
3. Drizzle with soy sauce and ginger.

**Time:** Prep 10 min | Cook 12 min | Total 22 min
"""

    # Simulate constraints: diabetes + peanut allergy
    test_constraints = {
        "avoid": ["peanuts", "high sugar foods"],
        "limit": ["sodium", "refined carbs"],
        "constraints": {
            "sugar_g": {"max": 25},
            "sodium_mg": {"max": 600},
        },
        "notes": "Patient has diabetes. Limit sugar and refined carbohydrates.",
    }

    test_intent = UserIntent(
        restrictions="peanuts",
        health_condition="diabetes",
        instructions="healthy dinner",
    )

    result = safety_filter.check(
        recipe_markdown=dummy_markdown,
        medical_constraints=test_constraints,
        user_intent=test_intent,
    )

    print(f"\n{'='*60}")
    print("SAFETY CHECK RESULT")
    print(f"{'='*60}")
    print(result.summary)
    print(f"\n--- Safe Recipes Markdown ---")
    print(result.safe_recipes_markdown[:500] + "..." if len(result.safe_recipes_markdown) > 500 else result.safe_recipes_markdown)
