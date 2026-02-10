from typing import List, Dict, Any

from langchain.schema import Document

class SafetyFilter:
    """Filters foods based on allergies, avoid-lists, and nutrition limits."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def filter(
        self,
        candidates: List[Document],
        allergies: List[str],
        constraints: Dict[str, Any],
        avoid_foods: List[str] = None,
    ) -> List[Document]:
        """Filter candidates by safety constraints."""
        avoid_foods = avoid_foods or []
        filtered = []

        allergies_lower = [a.lower() for a in allergies]
        avoid_lower = [a.lower() for a in avoid_foods]
        constraint_rules = constraints.get("constraints", {})

        if self.debug:
            print(f"\n[DEBUG] Filtering {len(candidates)} candidates")
            print(f"[DEBUG] Allergies: {allergies_lower}")
            print(f"[DEBUG] Avoid foods: {avoid_lower}")

        for doc in candidates:
            name_lower = doc.metadata.get("name", "").lower()

            # Check allergies
            if any(allergen in name_lower for allergen in allergies_lower):
                if self.debug:
                    print(f"  REJECTED (allergen): {name_lower}")
                continue

            # Check avoid list
            if any(avoid in name_lower for avoid in avoid_lower):
                if self.debug:
                    print(f"  REJECTED (avoid): {name_lower}")
                continue

            # Check nutrition limits
            rejected = False
            for nutrient, rule in constraint_rules.items():
                value = doc.metadata.get(nutrient, 0)
                if rule.get("max") and value > rule["max"]:
                    if self.debug:
                        print(f"  REJECTED (nutrition {nutrient}={value} > {rule['max']}): {name_lower}")
                    rejected = True
                    break
            if rejected:
                continue

            filtered.append(doc)

        return filtered

if __name__ == "__main__":
    safety_filter = SafetyFilter(debug=True)
    print("Safety Filter ready!")

    # test SafetyFilter with dummy data
    dummy_candidates = [
        Document(page_content="Grilled Chicken Salad", metadata={"name": "Grilled Chicken Salad", "calories": 400, "protein_g": 30, "carbs_g": 20, "fat_g": 15}),
        Document(page_content="Tomato Soup", metadata={"name": "Tomato Soup", "calories": 150, "protein_g": 5, "carbs_g": 10, "fat_g": 5}),
        Document(page_content="Peanut Butter Sandwich", metadata={"name": "Peanut Butter Sandwich", "calories": 500, "protein_g": 20, "carbs_g": 50, "fat_g": 25}),
        ]
    allergies = ["peanut"]
    constraints = {"constraints": {"calories": {"max": 450}}}
    avoid_foods = ["tomato"]
    filtered = safety_filter.filter(dummy_candidates, allergies, constraints, avoid_foods)
    print("\nFiltered candidates:")
    for doc in filtered:
        print(f"  {doc.metadata['name']} - {doc.metadata['calories']} cal")
    