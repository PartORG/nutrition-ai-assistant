from typing import List, Dict, Any
from dataclasses import dataclass, field
import sys
from pathlib import Path


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

@dataclass
class PipelineResult:
    """Result from the combined RAG pipeline."""
    intent: UserIntent
    constraints: Dict[str, Any]
    augmented_query: str
    llm_recommendation: str
    candidates_count: int
    filtered_count: int
    safe_foods: List[Dict[str, Any]]

    def display(self):
        """Pretty-print the pipeline result."""
        print("\n" + "=" * 60)
        print("PIPELINE RESULT")
        print("=" * 60)

        print(f"\n--- Intent ---\n{self.intent}")

        print(f"\n--- Constraints ---")
        for k, v in self.constraints.get("constraints", {}).items():
            print(f"  {k}: {v}")

        print(f"\n--- LLM Recommendation ---\n{self.llm_recommendation}")

        print(f"\n--- Safe Foods ({self.filtered_count}/{self.candidates_count} passed) ---")
        for i, food in enumerate(self.safe_foods, 1):
            print(f"  {i}. {food['name']}  |  {food['calories']} cal  |  "
                  f"P:{food['protein_g']}g  C:{food['carbs_g']}g  F:{food['fat_g']}g")


class RAGPipeline:
    """Combined pipeline for nutrition recommendations.
    
    Modular design — each step is a separate method that can be
    overridden, tested independently, or replaced with a different
    implementation (e.g., swap MedicalRAG for an API call).
    """
    
    def __init__(
        self,
        intent_parser: IntentParser,
        medical_rag: MedicalRAG,
        nutrition_rag: RecipesNutritionRAG,
        safety_filter: SafetyFilter
    ):
        self.intent_parser = intent_parser
        self.medical_rag = medical_rag
        self.nutrition_rag = nutrition_rag
        self.safety_filter = safety_filter
    
    # ── public entry point ──────────────────────────────────────
    
    def process(self, user_query: str, top_k: int = 10):
        """Run the full pipeline and return a PipelineResult."""
        print("\n" + "=" * 60)
        print("PROCESSING QUERY...")
        print("=" * 60)
        print(f"Query: {user_query}\n")
        
        # prompt user to enter his/hers/theys information (user_info = input('Enter your data: '))
                
        # parse user data --> name TEXT,
                # surname TEXT,
                # caretaker TEXT,
                # health condition TEXT,
                # created_at TEXT,
                # updated_at TEXT,
                # deleted_at TEXT,
                # age INTEGER,
                # gender TEXT
        
        # check if this user already exists in database read_user():
        # if exists - read user data and update user_query with user data (e.g. if user query doesn't contain health condition but it's stored in db - add it to the query)
        # TODO LATER: what to do ??? discuss and implement later.
        
        # if not exists - create new user in database with parsed data (insert_user(user_info)) and store user_id for future reference
        # store user data in the database (db_handler.add_user(user_info))
        # insert_user(user_info) ==> returns user_id
        # store user_id in variable user_id

        intent       = self._step_parse_intent(user_query)
        # print(intent)
        # update user data with preferences, restrictions, health conditions and allergies put to restrictions (db_handler.update_user(user_id, field, new_value))
        # update_user(user_id, field =  'preferences', new value = intent.preferences)

        constraints  = self._step_get_constraints(intent)

        # here take constraints and advice (that we dont have yet) and
        # update medical_advice table with health_condition, user_id, medical_advice (db_handler.update_medical_advice(user_id, constraints, advice))
        # health_condition check to take it from intent or from user prompt??

        augmented_q  = self._step_build_augmented_query(user_query, intent, constraints)
        llm_rec      = self._step_get_recommendation(augmented_q)
        safe_foods   = self._step_safety_filter(intent, constraints, top_k)
        
        return PipelineResult(
            intent=intent,
            constraints=constraints,
            augmented_query=augmented_q,
            llm_recommendation=llm_rec,
            candidates_count=safe_foods["candidates_count"],
            filtered_count=safe_foods["filtered_count"],
            safe_foods=safe_foods["results"],
        )
    
    # ── pipeline steps (each is independently testable) ─────────
    
    def _step_parse_intent(self, user_query: str) -> UserIntent:
        print("[Step 1] Intent Parser — extracting structured data...")
        intent = self.intent_parser.parse(user_query)
        print(f"  → Health conditions: {intent.health_condition or '(none)'}")
        print(f"  → Preferences: {intent.preferences or '(none)'}")
        print(f"  → Restrictions: {intent.restrictions or '(none)'}")
        print(f"  → Allergies: {intent.allergies or '(none)'}")
        return intent
    
    def _step_get_constraints(self, intent: UserIntent) -> Dict:
        print("\n[Step 2] Medical RAG — getting nutrition constraints...")
        constraints = self.medical_rag.get_constraints(intent.medical_conditions_list)
        print(f"  → Constraints: {list(constraints.get('constraints', {}).keys())}")
        print(f"  → Avoid: {constraints.get('avoid', [])}")
        print(f"  → Limit: {constraints.get('limit', [])}")
        # print(f"  → Medical Advice: {constraints.get('advice', '(none)')}")")
        return constraints
    
    def _step_build_augmented_query(
        self, original_query: str, intent: UserIntent, constraints: Dict
    ) -> str:
        print("\n[Step 3] Building augmented query...")
        augmented = self._build_augmented_query(original_query, intent, constraints)
        print(f"  → Query built ({len(augmented)} chars)")
        return augmented
    
    def _step_get_recommendation(self, augmented_query: str) -> str:
        print("\n[Step 4] Nutrition RAG — getting LLM recommendation...")
        rec = self.nutrition_rag.ask(augmented_query)
        print("  → Recommendation received")
        return rec
    
    def _step_safety_filter(
        self, intent: UserIntent, constraints: Dict, top_k: int
    ) -> Dict:
        print("\n[Step 5] Safety Check — filtering unsafe foods...")
        
        # Use ingredients from preferences for the search
        search_terms = intent.ingredients_list or ["healthy food"]
        search_query = " ".join(search_terms)
        
        candidates = self.nutrition_rag.get_retrieved_docs(search_query)
        print(f"  → Searched {len(candidates)} candidates")
        
        avoid_foods = constraints.get("avoid", [])
        safe_foods = self.safety_filter.filter(
            candidates=candidates,
            # allergies=intent.allergies,
            constraints=constraints,
            avoid_foods=avoid_foods,
        )
        print(f"  → {len(safe_foods)} safe foods after filtering")
        
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
            for doc in safe_foods[:top_k]
        ]
        
        return {
            "candidates_count": len(candidates),
            "filtered_count": len(safe_foods),
            "results": results,
        }
    
    # ── query builder ───────────────────────────────────────────
    
    def _build_augmented_query(
        self, original_query: str, intent: UserIntent, constraints: Dict
    ) -> str:
        """Build augmented query using User-model-aligned intent fields."""
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
        
        sections = [f"USER REQUEST: {original_query}"]
        
        if intent.health_condition:
            sections.append(f"MEDICAL CONDITIONS: {intent.health_condition}")
        
        if intent.restrictions:
            sections.append(f"DIETARY RESTRICTIONS: {intent.restrictions}")
        
        sections.append(
            "NUTRITION GUIDELINES (daily targets):\n"
            + ("\n".join("- " + c for c in constraint_text) if constraint_text else "- General healthy eating")
        )
        
        sections.append(
            "ALLERGIES (MUST AVOID):\n"
            + ("\n".join("- " + a for a in intent.allergies) if intent.allergies else "- None")
        )
        
        avoid = constraints.get("avoid", [])
        sections.append(
            "FOODS TO AVOID:\n"
            + ("\n".join("- " + f for f in avoid) if avoid else "- None")
        )
        
        limit = constraints.get("limit", [])
        sections.append(
            "FOODS TO LIMIT (reduce but okay in moderation):\n"
            + ("\n".join("- " + f for f in limit) if limit else "- None")
        )
        
        if intent.preferences:
            sections.append(f"DESIRED INGREDIENTS / PREFERENCES: {intent.preferences}")
        
        sections.append(
            "Based on the above, recommend specific food ingredients that are safe and healthy.\n"
            "Provide nutritional breakdown for each recommendation."
        )
        
        return "\n\n".join(sections)


if __name__ == "__main__":
    # Initialize Database (check if db file exists, else create new file with tables)

    # Initialize components
    intent_parser = IntentParser(model_name=LLM_MODEL)
    medical_rag = MedicalRAG(folder_paths=[str(PDF_DIR)], model_name=LLM_MODEL, vectorstore_path=str(MEDICAL_VECTORSTORE_PATH), embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    medical_rag.initialize(force_rebuild=False)  # TODO: set to False after first successful rebuild

    nutrition_rag = RecipesNutritionRAG(data_folder=str(DATA_DIR), model_name=LLM_MODEL, vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH))
    nutrition_rag.initialize()

    safety_filter = SafetyFilter(debug=True)

    # Create the pipeline
    pipeline = RAGPipeline(
        intent_parser=intent_parser,
        medical_rag=medical_rag,
        nutrition_rag=nutrition_rag,
        safety_filter=safety_filter
    )

    print("\nRAG Pipeline ready!")

    # Test the full pipeline end-to-end
    test_query = "I have parkinson. I want to make something with chicken, eggs and salad. Quick breakfast. I can't eat tomatoes or onions."

    result = pipeline.process(test_query)
    result.display()