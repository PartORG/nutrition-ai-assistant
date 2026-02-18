"""
pipeline - Main RAG pipeline orchestrator with SQLite integration.

Coordinates the full recipe recommendation flow:

    Step 1: Intent parsing    — extract structured intent from user query
    Step 2: Medical RAG       — retrieve dietary constraints for health conditions
    Step 3: Query augmentation — build enriched query with all user context
    Step 4: Nutrition RAG     — retrieve and generate recipe recommendations
    Step 5: Safety check      — validate recipes against user constraints

Usage:
    from pipeline.pipeline import RAGPipeline, PipelineResult

    pipeline = RAGPipeline(intent_parser, medical_rag, nutrition_rag, safety_filter)
    result = pipeline.process("I have diabetes. Suggest a healthy dinner.")
    result.display()
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from settings import (
    PDF_DIR, DATA_DIR,
    MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH,
    LLM_MODEL,
)
from components.intent_retriever import UserIntent, IntentParser
from components.safety_filter import SafetyFilter, SafetyCheckResult
from rags.recipes_nutrition_rag import RecipesNutritionRAG
from rags.medical_rag import MedicalRAG
from database.db import UserDBHandler
from database.models import User, MedicalAdvice, UserProfileHistory


@dataclass
class PipelineResult:
    """Result container for a completed pipeline run.

    Holds the parsed intent, medical constraints, augmented query,
    raw LLM recommendation, and safety check results.
    """
    intent: UserIntent
    constraints: Dict[str, Any]
    augmented_query: str
    llm_recommendation: str
    safety_result: SafetyCheckResult = None

    def display(self):
        """Print a formatted summary of the pipeline result."""
        print("\n" + "=" * 60)
        print("PIPELINE RESULT")
        print("=" * 60)
        print(f"\n--- Intent ---\n{self.intent}")
        print(f"\n--- Constraints ---")
        for k, v in self.constraints.get("constraints", {}).items():
            print(f"  {k}: {v}")
        if self.safety_result:
            print(f"\n--- Safety Check ---")
            print(self.safety_result.summary)
            print(f"\n--- Safe Recipes ---\n{self.safety_result.safe_recipes_markdown}")
        else:
            print(f"\n--- LLM Recommendation ---\n{self.llm_recommendation}")


class RAGPipeline:
    """End-to-end nutrition recommendation pipeline.

    Orchestrates intent parsing, medical constraint retrieval, query
    augmentation, recipe generation, and safety filtering. Manages
    user persistence via SQLite through UserDBHandler.
    """

    def __init__(
        self,
        intent_parser: IntentParser,
        medical_rag: MedicalRAG,
        nutrition_rag: RecipesNutritionRAG,
        safety_filter: SafetyFilter,
    ):
        self.intent_parser = intent_parser
        self.medical_rag = medical_rag
        self.nutrition_rag = nutrition_rag
        self.safety_filter = safety_filter
        self.db_handler = UserDBHandler()
        self.db_handler.create_all_tables()

    def process(self, user_query: str, user_data: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Run the full pipeline for a user query.

        Args:
            user_query: The user's natural language request.
            user_data:  Optional dict with user identity fields
                        (name, surname, caretaker, health_condition, age, gender).
                        If None, prompts interactively via stdin.

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        print("\n" + "=" * 60)
        print("PROCESSING QUERY...")
        print("=" * 60)
        print(f"Query: {user_query}\n")

        # TODO: delete from here --> handle by login in App
        # Step 0: Handle user registration / lookup
        # user_id, is_new = self._step_handle_user(user_data) #TODO: delete this line after testing

        # Step 1: Parse intent from query
        intent = self._step_parse_intent(user_query)
        print(f"[INTENT] content: {intent}")
        # self._step_update_user(user_id, intent) TODO remove comment after testing
        is_new=True #TODO: delete this line after testing

        # Step 2: Get medical constraints (from RAG or DB cache)
        if not is_new:
            print("[PIPELINE] Bypassing Medical RAG. Fetching existing advice...")
            # TODO: update db function to get new columns too
            db_advice = self.db_handler.get_medical_advice_by_user(2) #TODO: change to user_id again

            if db_advice:
                # TODO: store constraints, avoid and limit (from Medical RAG) in the medical_advice table.
                latest_advice = db_advice[-1]
                constraints = {
                    "notes": latest_advice[2],
                    "constraints": {},
                    "avoid": [],
                    "limit": [],
                }
                print(f"[PIPELINE] Retrieved advice from DB: {constraints['notes']}")
            else:
                constraints = self._step_get_constraints(intent)
        else:
            constraints = self._step_get_constraints(intent)

        # Store medical advice in DB
        advice_text = constraints.get("notes", "")
        if advice_text:
            advice = MedicalAdvice(
                health_condition=intent.health_condition or "",
                medical_advice=advice_text,
                user_id=2, # TODO: change to user_id again
                dietary_limit=str(constraints.get("limit", {})),
                dietary_constraints=str(constraints.get("constraints", {})),
                avoid=str(constraints.get("avoid", [])),
            )
            self.db_handler.insert_medical_advice(advice)

        # Step 3 & 4: Build augmented query and get recommendation
        augmented_q = self._step_build_augmented_query(user_query, intent, constraints)
        llm_rec = self._step_get_recommendation(augmented_q)

        # Step 5: Safety filter
        safety_result = self._step_safety_check(llm_rec, constraints, intent)

        return PipelineResult(
            intent=intent,
            constraints=constraints,
            augmented_query=augmented_q,
            llm_recommendation=llm_rec,
            safety_result=safety_result,
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_handle_user(self, user_data: Optional[Dict[str, Any]] = None) -> tuple:
        """Create or retrieve a user from the DB.

        Args:
            user_data: Optional dict with keys: name, surname, caretaker,
                       health_condition, age, gender. If None, prompts via stdin.

        Returns:
            Tuple of (user_id, is_new) where is_new indicates first-time user.
        """
        if user_data:
            name = user_data.get("name", "")
            surname = user_data.get("surname", "")
            caretaker = user_data.get("caretaker", "")
            health_condition = user_data.get("health_condition", "")
            age = user_data.get("age", 0)
            gender = user_data.get("gender", "")
        else:
            csv_input = input("Enter user data (name,surname,caretaker,health_condition,age,gender): ")
            name, surname, caretaker, health_condition, age, gender = csv_input.strip().split(",")
            name, surname, caretaker, health_condition, gender = (
                name.strip(), surname.strip(), caretaker.strip(),
                health_condition.strip(), gender.strip(),
            )
            age = int(age)

        print(f"[DB] Handling user: {name} {surname}, caretaker: {caretaker}, health condition: {health_condition}, age: {age}")

        existing = self.db_handler.read_user(name, surname)
        print(f"[DB] Existing user query result: {existing}")

        if existing:
            user_id = existing[0]
            print(f"[DB] Found existing user with ID: {user_id}")
            return user_id, False
        else:
            user = User(
                name=name,
                surname=surname,
                user_name=f"{name}{surname}",
                caretaker=caretaker,
                age=int(age),
                gender=str(gender).capitalize(),
            )
            user_id = self.db_handler.insert_user(user)

            profile = UserProfileHistory(
                preferences="",
                user_id=user_id,
                health_condition=health_condition,
                restrictions="",
            )
            self.db_handler.insert_user_profile_history(profile)
            return user_id, True

    def _step_update_user(self, user_id: int, intent: UserIntent):
        """Save the user's current profile as a new UserProfileHistory snapshot."""
        print(f"[DB] Saving profile snapshot for user ID {user_id}...")
        print(f"  -> Preferences: {intent.ingredients_list}")
        print(f"  -> Restrictions: {intent.restrictions_list}")
        print(f"  -> Health conditions: {intent.health_condition}")

        health = intent.health_condition
        if isinstance(health, list):
            health = ", ".join(health)

        profile = UserProfileHistory(
            preferences=", ".join(intent.ingredients_list),
            user_id=user_id,
            health_condition=health,
            restrictions=", ".join(intent.restrictions_list),
        )
        self.db_handler.insert_user_profile_history(profile)

    def _step_parse_intent(self, user_query: str) -> UserIntent:
        """Extract structured intent from the user's natural language query."""
        print("[Step 1] Intent Parser — extracting structured data...")
        intent = self.intent_parser.parse(user_query)
        print(f"  -> Health conditions: {intent.health_condition or '(none)'}")
        print(f"  -> Dietary restrictions: {intent.restrictions or '(none)'}")
        print(f"  -> Preferences: {intent.preferences or '(none)'}")
        print(f"  -> Instructions: {intent.instructions or '(none)'}")
        return intent

    def _step_get_constraints(self, intent: UserIntent) -> Dict:
        """Retrieve dietary constraints from the Medical RAG based on health conditions."""
        print("\n[Step 2] Medical RAG — getting nutrition constraints...")
        print(f"  -> User medical conditions: {intent.health_condition or '(none)'}")
        constraints = self.medical_rag.get_constraints(intent.health_condition)
        print(f"  -> Constraints keys: {list(constraints.get('constraints', {}).keys())}")
        print(f"  -> Avoid foods: {constraints.get('avoid', [])}")
        print(f"  -> Limit foods: {constraints.get('limit', [])}")
        return constraints

    def _step_build_augmented_query(self, original_query: str, intent: UserIntent, constraints: Dict) -> str:
        """Build an enriched query combining user request, medical context, and constraints."""
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
        print(f"  -> Query built ({len(query)} chars)")
        return query

    def _step_get_recommendation(self, augmented_query: str) -> str:
        """Send the augmented query to Nutrition RAG and return the recipe markdown."""
        print("\n[Step 4] Nutrition RAG — getting LLM recommendation...")
        rec = self.nutrition_rag.ask(augmented_query)
        print("  -> Recommendation received")
        return rec

    def _step_safety_check(self, llm_rec: str, constraints: Dict, intent: UserIntent) -> SafetyCheckResult:
        """Validate recipe output against user constraints using the safety filter."""
        print("\n[Step 5] Safety Check — validating recipes against constraints...")
        safety_result = self.safety_filter.check(
            recipe_markdown=llm_rec,
            medical_constraints=constraints,
            user_intent=intent,
        )
        print(f"  -> {safety_result.safe_count}/{safety_result.total_count} recipes passed")
        return safety_result


if __name__ == "__main__":
    intent_parser = IntentParser(model_name=LLM_MODEL)
    medical_rag = MedicalRAG(
        folder_paths=[str(PDF_DIR)],
        model_name=LLM_MODEL,
        vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
        embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    )
    medical_rag.initialize(force_rebuild=False)

    nutrition_rag = RecipesNutritionRAG(
        data_folder=str(DATA_DIR),
        model_name=LLM_MODEL,
        vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH),
    )
    nutrition_rag.initialize()

    safety_filter = SafetyFilter(model_name=LLM_MODEL, debug=True)

    pipeline = RAGPipeline(
        intent_parser=intent_parser,
        medical_rag=medical_rag,
        nutrition_rag=nutrition_rag,
        safety_filter=safety_filter,
    )

    print("\nRAG Pipeline ready!")

    query = "I have diabetes and high blood pressure. I want to eat something with pork belly, but I need to avoid too much sodium and sugar. What do you recommend for dinner?"
    result = pipeline.process(query)
    result.display()
