# rag_pipeline/rag_main.py — full pipeline with SQLite integration

from typing import List, Dict, Any
from dataclasses import dataclass, field
import sys
from pathlib import Path
from datetime import datetime
import uuid

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from pipeline.config import (
    PDF_DIR, DATA_DIR,
    MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH,
    LLM_MODEL,
)
from pipeline.intent_retriever import UserIntent, IntentParser
from pipeline.recipes_nutrition_rag import RecipesNutritionRAG
from pipeline.medical_rag import MedicalRAG
from pipeline.safety_filter import SafetyFilter
from pipeline.db import UserDBHandler, User, MedicalAdvice

@dataclass
class PipelineResult:
    intent: UserIntent
    constraints: Dict[str, Any]
    augmented_query: str
    llm_recommendation: str
    # candidates_count: int
    # filtered_count: int
    # safe_foods: List[Dict[str, Any]]

    def display(self):
        print("\n" + "=" * 60)
        print("PIPELINE RESULT")
        print("=" * 60)
        print(f"\n--- Intent ---\n{self.intent}")
        print(f"\n--- Constraints ---")
        for k, v in self.constraints.get("constraints", {}).items():
            print(f"  {k}: {v}")
        print(f"\n--- LLM Recommendation ---\n{self.llm_recommendation}")
        # print(f"\n--- Safe Foods ({self.filtered_count}/{self.candidates_count} passed) ---")
        # for i, food in enumerate(self.safe_foods, 1):
        #     print(f"  {i}. {food['name']}  |  {food['calories']} cal  |  P:{food['protein_g']}g  C:{food['carbs_g']}g  F:{food['fat_g']}g")


class RAGPipeline:
    def __init__(self, intent_parser, medical_rag, nutrition_rag, safety_filter):
        self.intent_parser = intent_parser
        self.medical_rag = medical_rag
        self.nutrition_rag = nutrition_rag
        self.safety_filter = safety_filter
        self.db_handler = UserDBHandler()
        self.db_handler.create_users_table()
        self.db_handler.create_medical_advice_table()

    def _step_handle_user(self) -> tuple:
        csv_input = input("Enter user data (name,surname,caretaker,health_condition,age,gender): ")
        name, surname, caretaker, health_condition, age, gender = csv_input.strip().split(",")
        print(f"[DB] Handling user: {name} {surname}, caretaker: {caretaker}, health condition: {health_condition}, age: {age},")
        existing = self.db_handler.read_user(name.strip(), surname.strip())
        print(f"[DB] Existing user query result: {existing}")
        now = datetime.now().isoformat()
        #todo: try to add a loop function here to handle multiple users in a row, until user types "exit"

        if existing:
            user_id = existing[0] # The ID from the fetched tuple
            print(f"[DB] Found existing user with ID: {user_id}")
            return user_id, False  # is_new = False
        else:
            user = User(
                name=name.strip(),
                surname=surname.strip(),
                preferences="",
                restrictions="",
                health_condition=health_condition.strip(),
                caretaker=caretaker.strip(),
                created_at=now,
                updated_at=now,
                deleted_at="",
                age=int(age),
                gender=gender.strip().capitalize(),
            )
            # Create new user logic as you already have it...
            user_id = self.db_handler.insert_user(user)
            return user_id, True   # is_new = True
    
    def step_update_user(self, user_id: int, intent: UserIntent):
        print(f"[DB] Updating user ID {user_id} with intent data...")
        print(f"  → Preferences: {intent.ingredients_list}")
        print(f"  → Restrictions: {intent.restrictions_list}")
        print(f"  → Health conditions: {intent.health_condition}")
        self.db_handler.update_user(user_id, "preferences", ", ".join(intent.ingredients_list))
        self.db_handler.update_user(user_id, "restrictions", ", ".join(intent.restrictions_list))
        if isinstance(intent.health_condition, list):
            self.db_handler.update_user(user_id, "health_condition", ", ".join(intent.health_condition))
        else:
            self.db_handler.update_user(user_id, "health_condition", ", ".join(intent.health_condition.split(",")))

    def process(self, user_query: str, top_k: int = 10):
        print("\n" + "=" * 60)
        print("PROCESSING QUERY...")
        print("=" * 60)
        print(f"Query: {user_query}\n")

        user_id, is_new = self._step_handle_user()
        intent = self._step_parse_intent(user_query)
        print(f"[INTENT] content: {intent}")
        self.step_update_user(user_id, intent)
        
        if not is_new:
            print("[PIPELINE] Bypassing Medical RAG. Fetching existing advice...")
            # Get the latest advice tuple from DB
            db_advice = self.db_handler.get_medical_advice_by_user(user_id)
            
            if db_advice:
                # Assuming medical_advice column stores the notes
                # To skip MedicalRAG, we reconstruct the constraints dict
                latest_advice = db_advice[-1] # Get most recent
                constraints = {
                    "notes": latest_advice[2], # The medical_advice column
                    "constraints": {}, # Optional: Add logic to store/parse specific limits
                    "avoid": [],
                    "limit": []
                }
                print(f"[PIPELINE] Retrieved advice from DB: {constraints['notes']}")
            else:
                # Fallback if user exists but has no advice yet
                constraints = self._step_get_constraints(intent)
        else:
            # Full flow for new users
            constraints = self._step_get_constraints(intent)
            
        # constraints = self._step_get_constraints(intent)

        advice_text = constraints.get("notes", "")
        if advice_text:
            advice = MedicalAdvice(
                health_condition=intent.health_condition or "",
                medical_advice=advice_text,
                user_id=user_id
            )
            self.db_handler.insert_medical_advice(advice)

        augmented_q = self._step_build_augmented_query(user_query, intent, constraints)
        llm_rec = self._step_get_recommendation(augmented_q)
        # safe_foods = self._step_safety_filter(intent, constraints, top_k)

        return PipelineResult(
            intent=intent,
            constraints=constraints,
            augmented_query=augmented_q,
            llm_recommendation=llm_rec,
            # candidates_count=0,
            # filtered_count=0,
            # safe_foods=safe_foods["results"],
        )

    def _step_parse_intent(self, user_query: str) -> UserIntent:
        print("[Step 1] Intent Parser — extracting structured data...")
        intent = self.intent_parser.parse(user_query)
        print(f"  → Medical conditions: {intent.health_condition or '(none)'}")
        print(f"  → Dietary restrictions: {intent.restrictions or '(none)'}")
        # print(f"  → Allergies: {intent.allergies or '(none)'}")
        #print(f"  → Ingredients: {intent.ingredients_list or '(none)'}")
        #print(f"  → Cooking style: {intent.cooking_style or '(none)'}")
        return intent

    def _step_get_constraints(self, intent: UserIntent) -> Dict:
        print("\n[Step 2] Medical RAG — getting nutrition constraints...")
        print(f"  → User medical conditions: {intent.health_condition or '(none)'}")
        constraints = self.medical_rag.get_constraints(intent.health_condition)
        print(f"  → Constraints keys: {list(constraints.get('constraints', {}).keys())}")
        print(f"  → Avoid foods: {constraints.get('avoid', [])}")
        print(f"  → Limit foods: {constraints.get('limit', [])}")
        return constraints

    def _step_build_augmented_query(self, original_query: str, intent: UserIntent, constraints: Dict) -> str:
        print("\n[Step 3] Building augmented query...")
        constraint_rules = constraints.get("constraints", {})
        constraint_text = []

        if constraint_rules.get("sugar_g", {}).get("max"):
            constraint_text.append(f"prefer foods lower in sugar (daily limit: {constraint_rules['sugar_g']['max']}g)")
        if constraint_rules.get("sodium_mg", {}).get("max"):
            constraint_text.append(f"prefer foods lower in sodium (daily limit: {constraint_rules['sodium_mg']['max']}mg)")
        if constraint_rules.get("fiber_g", {}).get("min"):
            constraint_text.append(f"prefer foods with good fiber content (daily goal: {constraint_rules['fiber_g']['min']}g)")
        if constraint_rules.get("protein_g", {}).get("max"):
            constraint_text.append(f"moderate protein intake (daily limit: {constraint_rules['protein_g']['max']}g)")

        sections = [f"USER REQUEST:\n{original_query}"]

        if intent.health_condition:
            sections.append("MEDICAL CONDITIONS:\n- " + "\n- ".join(intent.health_condition.split(",")))

        if intent.restrictions:
            sections.append("DIETARY RESTRICTIONS:\n- " + "\n- ".join(intent.restrictions))

        if intent.preferences:
                    sections.append("PREFERENCES:\n- " + "\n- ".join(intent.preferences.split(",")))

        sections.append("NUTRITION GUIDELINES:\n" + ("\n".join("- " + c for c in constraint_text) if constraint_text else "- General healthy eating"))
        # sections.append("ALLERGIES (MUST AVOID):\n" + ("\n".join("- " + a for a in intent.allergies) if intent.allergies else "- None"))

        avoid = constraints.get("avoid", [])
        sections.append("FOODS TO AVOID:\n" + ("\n".join("- " + f for f in avoid) if avoid else "- None"))

        limit = constraints.get("limit", [])
        sections.append("FOODS TO LIMIT:\n" + ("\n".join("- " + f for f in limit) if limit else "- None"))

        if intent.ingredients_list:
            sections.append("DESIRED INGREDIENTS:\n- " + "\n- ".join(intent.ingredients_list))

        if intent.instructions:
            sections.append("INSTRUCTIONS:\n- " + "\n- ".join(intent.instructions.split(",")))

        sections.append("Based on the above, recommend specific safe and healthy food options.\nInclude a brief nutritional breakdown for each.")
        query = "\n\n".join(sections)
        print(f"  → Query built ({len(query)} chars)")
        return query

    def _step_get_recommendation(self, augmented_query: str) -> str:
        print("\n[Step 4] Nutrition RAG — getting LLM recommendation...")
        rec = self.nutrition_rag.ask(augmented_query)
        print("  → Recommendation received")
        return rec

    def _step_safety_filter(self, llm_rec: str, constraints: Dict, top_k: int) -> Dict:
        print("\n[Step 5] Safety Check — filtering unsafe foods...")
        search_terms = self.intent_parser.get_ingredients_list(llm_rec)
        search_query = " ".join(search_terms)
        candidates = self.nutrition_rag.get_retrieved_docs(search_query)
        print(f"  → Retrieved {len(candidates)} candidates")

        safe_docs = self.safety_filter.filter(
            candidates=candidates,
            # allergies=intent.allergies,
            constraints=constraints,
            avoid_foods=constraints.get("avoid", []),
        )

        print(f"  → {len(safe_docs)} safe foods after filtering")

        results = [
            {
                "name": doc.metadata.get("name"),
                "calories": doc.metadata.get("calories", 0),
                "protein_g": doc.metadata.get("protein_g", 0),
                "carbs_g": doc.metadata.get("carbs_g", 0),
                "fat_g": doc.metadata.get("fat_g", 0),
                "fiber_g": doc.metadata.get("fiber_g", 0),
                "sugar_g": doc.metadata.get("sugar_g", 0),
                "sodium_mg": doc.metadata.get("sodium_mg", 0),
            }
            for doc in safe_docs[:top_k]
        ]

        return {
            "candidates_count": len(candidates),
            "filtered_count": len(safe_docs),
            "results": results,
        }


if __name__ == "__main__":
    intent_parser = IntentParser(model_name=LLM_MODEL)
    medical_rag = MedicalRAG(folder_paths=[str(PDF_DIR)], model_name=LLM_MODEL, vectorstore_path=str(MEDICAL_VECTORSTORE_PATH), embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    medical_rag.initialize(force_rebuild=False)  # TODO: set back to False after first run

    nutrition_rag = RecipesNutritionRAG(data_folder=str(DATA_DIR), model_name=LLM_MODEL, vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH))
    nutrition_rag.initialize()

    safety_filter = SafetyFilter(debug=True)

    pipeline = RAGPipeline(
        intent_parser=intent_parser,
        medical_rag=medical_rag,
        nutrition_rag=nutrition_rag,
        safety_filter=safety_filter
    )

    print("\nRAG Pipeline ready!")

    query = "I have diabetes and high blood pressure. I want to eat something with pork belly, but I need to avoid too much sodium and sugar. What do you recommend for dinner?"
    result = pipeline.process(query)
    result.display()
