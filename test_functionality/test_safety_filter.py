"""
Test Safety Filter
"""
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from infrastructure.config import Settings
from factory import ServiceFactory
from domain.models import Recipe, NutritionValues, NutritionConstraints, UserIntent

# --- Instructions ---
# recipes: List of Recipe objects to check for safety. Each Recipe should have:
#   - name: str (recipe name)
#   - ingredients: List[str] (ingredients used)
#   - cook_instructions: str (instructions for cooking)
# constraints: NutritionConstraints object specifying dietary restrictions.
#   - avoid: List[str] (foods to avoid, e.g. allergens)
#   - constraints: Dict (numeric nutrition limits, can be empty for test)
# intent: UserIntent object specifying user restrictions and context.
#   - restrictions: List[str] (e.g. allergies, dietary restrictions)
#   - preferences, health_conditions, instructions, name, surname, caretaker: can be empty for test
#
# You can modify the dummy data below to test different scenarios.

async def main():
    settings = Settings.from_env()
    factory = ServiceFactory(settings)
    await factory.initialize()
    filter = factory._safety_filter
    # Improved dummy data for testing
    recipes = [
        Recipe(
            name="Chicken Caesar Salad",
            ingredients=["chicken breast", "romaine lettuce", "parmesan cheese", "croutons", "caesar dressing", "anchovy"],
            nutrition=NutritionValues(
                calories=520,
                protein_g=38,
                carbs_g=22,
                fat_g=30,
                saturated_fat_g=8,
                fiber_g=3,
                sodium_mg=1820,    # exceeds max 1500 → should trigger warning
                sugar_g=6,
            ),
            cook_instructions="Grill chicken, toss with lettuce, cheese, croutons, and dressing.",
            servings=1,
        ),
        Recipe(
            name="Vegan Stir Fry",
            ingredients=["tofu", "broccoli", "soy sauce", "carrot", "bell pepper", "ginger"],
            nutrition=NutritionValues(
                calories=310,
                protein_g=18,
                carbs_g=34,
                fat_g=10,
                saturated_fat_g=1.5,
                fiber_g=7,
                sodium_mg=980,
                sugar_g=12,
            ),
            cook_instructions="Stir fry vegetables and tofu with soy sauce and ginger.",
            servings=2,
        ),
        Recipe(
            name="Gluten-Free Pancakes",
            ingredients=["gluten-free flour", "egg", "milk", "baking powder", "butter"],
            nutrition=NutritionValues(
                calories=440,
                protein_g=10,
                carbs_g=62,
                fat_g=16,
                saturated_fat_g=9,
                fiber_g=2,
                sodium_mg=540,
                sugar_g=28,            # exceeds max 25 → should trigger warning
            ),
            cook_instructions="Mix ingredients, cook pancakes on skillet.",
            servings=2,
        ),
    ]
    constraints = NutritionConstraints(
        avoid=["anchovy", "soy sauce"],
        constraints={
            "sodium_mg": {"max": 1500},
            "sugar_g": {"max": 25},
        },
        dietary_goals=["low sodium", "low sugar"],
        foods_to_increase=["vegetables"],
        limit=["cheese", "butter"],
        notes="Testing multiple constraints"
    )
    intent = UserIntent(
        restrictions=["vegan", "gluten-free", "lactose-free"],
        preferences=["low sodium"],
        health_conditions=["hypertension"],
        instructions=["avoid processed foods"],
        name="TestUser",
        surname="Demo",
        caretaker=""
    )
    result = await filter.check(recipes, constraints, intent)
    print("Safety Filter Result Summary:\n", result.summary)
    for verdict in result.recipe_verdicts:
        print(f"\nRecipe: {verdict.recipe_name}")
        print(f"Verdict: {verdict.verdict}")
        for issue in verdict.issues:
            print(f"- Issue: {issue.category} [{issue.severity}] - {issue.description}")

if __name__ == "__main__":
    asyncio.run(main())
