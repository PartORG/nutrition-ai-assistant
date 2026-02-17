"""
intent_retriever - Extracts structured user intent from natural language queries.

This module provides two main exports:

    UserIntent    A dataclass representing the parsed fields from a user query
                  (name, preferences, restrictions, health conditions, etc.).
                  Aligned with the UserProfileHistory database model.

    IntentParser  Uses an Ollama LLM with JSON output parsing to convert a
                  free-text user query into a UserIntent instance. The LLM is
                  prompted with detailed examples and normalization rules so that
                  medical conditions, dietary restrictions, and meal-specific
                  instructions are correctly categorized.

Usage:
    from components.intent_retriever import IntentParser, UserIntent
    parser = IntentParser(model_name="llama3.2")
    intent = parser.parse("I have diabetes. Make me a low-sodium dinner.")
"""

from typing import List
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM


import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from settings import LLM_MODEL


@dataclass
class UserIntent:
    """Structured representation of parsed user intent.

    Each field corresponds to a category of information extracted from the
    user's natural language query. Fields are aligned with the
    UserProfileHistory database model for seamless storage.

    Fields not found in the query remain as empty strings.

    Attributes:
        name:             User's first name.
        surname:          User's last name.
        preferences:      General, long-term food preferences — favorite cuisines,
                          cooking styles, and preferred ingredients (comma-separated).
        restrictions:     Permanent dietary restrictions AND allergies — vegetarian,
                          vegan, gluten-free, nut allergy, etc. (comma-separated).
        health_condition: Medical conditions requiring dietary management — diabetes,
                          hypertension, kidney_disease, etc. (comma-separated, snake_case).
        caretaker:        Name of caretaker or helper, if mentioned.
        instructions:     Specific requests for THIS single meal — ingredients to use,
                          time constraints, cuisine to try today (comma-separated).
    """
    name: str = ""
    surname: str = ""
    preferences: str = ""
    restrictions: str = ""
    health_condition: str = ""
    caretaker: str = ""
    instructions: str = ""

    def __repr__(self):
        fields = {
            "name": self.name,
            "surname": self.surname,
            "preferences": self.preferences,
            "restrictions": self.restrictions,
            "health_condition": self.health_condition,
            "caretaker": self.caretaker,
            "instructions": self.instructions,
        }
        lines = [f"  {k}: {v}" for k, v in fields.items() if v]
        return "UserIntent:\n" + "\n".join(lines) if lines else "UserIntent: (empty)"

    @property
    def medical_conditions_list(self) -> List[str]:
        """Return health_condition as a list for pipeline processing.

        Splits the comma-separated health_condition string and strips whitespace.
        Example: "diabetes, hypertension" -> ["diabetes", "hypertension"]
        """
        return [c.strip() for c in self.health_condition.split(",") if c.strip()]

    @property
    def ingredients_list(self) -> List[str]:
        """Extract ingredient names from preferences string.

        Splits the comma-separated preferences string and strips whitespace.
        Example: "Italian food, quick cooking, tofu" -> ["Italian food", "quick cooking", "tofu"]
        """
        return [p.strip() for p in self.preferences.split(",") if p.strip()]

    @property
    def restrictions_list(self) -> List[str]:
        """Return restrictions as a list for pipeline processing.

        Splits the comma-separated restrictions string and strips whitespace.
        Example: "vegetarian, lactose-free" -> ["vegetarian", "lactose-free"]
        """
        return [r.strip() for r in self.restrictions.split(",") if r.strip()]


class IntentParser:
    """Parses user queries to extract structured intent using LLM.

    Uses an Ollama LLM with a carefully crafted system prompt to classify
    each piece of information in the user's query into the correct UserIntent
    field. The prompt includes normalization rules (e.g. "parkenson" ->
    "parkinsons_disease") and distinction rules (preferences vs restrictions
    vs instructions).

    The parser outputs JSON which is parsed into a UserIntent dataclass.
    """

    def __init__(self, model_name: str = "llama3.2"):
        # temperature=0 ensures deterministic, consistent extraction results.
        # format="json" forces the LLM to output valid JSON (Ollama feature).
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0,
            format="json"
        )
        self.parser = JsonOutputParser()
        self.chain = self._build_chain()

    def _build_chain(self):
        """Build the LangChain processing chain for intent extraction.

        Constructs a chain: prompt template -> LLM -> JSON parser.
        The system prompt contains detailed extraction rules, normalization
        mappings, and examples to guide the LLM.
        """
        system_instructions = """You are a medical nutrition data extractor for a specialized dietary assistant.
Your task: Extract ONLY explicitly stated information from user queries. Never infer, guess, or add information.

OUTPUT FORMAT: JSON object with these keys:

1. "name" (string): First name ONLY, or "" if not stated
2. "surname" (string): Last name ONLY, or "" if not stated
3. "preferences" (string): GENERAL, long-term food preferences
   - Favorite cuisines (Italian, Asian, Mediterranean)
   - Preferred cooking styles (quick meals, slow-cooked, grilled)
   - Ingredients the user generally loves/prefers
   - Comma-separated, or "" if not stated
4. "restrictions" (string): PERMANENT dietary restrictions AND allergies
   - Dietary lifestyles: vegetarian, vegan, pescatarian, keto, paleo, gluten-free, lactose-free
   - Food allergies and intolerances
   - Foods that must be avoided always
   - Comma-separated, or "" if not stated
5. "health_condition" (string): Medical conditions requiring dietary management
   - Examples: diabetes, hypertension, kidney_disease, parkinsons_disease, multiple_sclerosis, als, celiac_disease
   - Use snake_case format
   - Auto-correct common spelling errors
   - Comma-separated, or "" if not stated
6. "caretaker" (string): Name of caretaker/helper, or "" if not stated
7. "instructions" (string): SPECIFIC requests for THIS SINGLE meal
   - Ingredients to use for this meal
   - Special preparation for today
   - Time constraints for this meal
   - Cuisine to try today
   - Temporary dietary choice for this meal
   - Or "" if not stated

NORMALIZATION & ERROR CORRECTION:
Medical conditions (correct spelling automatically):
- "parkenson", "parkinson", "parkinsons" → "parkinsons_disease"
- "MS", "multiple sclerosis" → "multiple_sclerosis"
- "sugar", "blood sugar", "diabetic" → "diabetes"
- "high blood pressure", "blood pressure" → "hypertension"
- "kidney problems", "renal" → "kidney_disease"
- "ALS", "Lou Gehrig's" → "als"

Restrictions (standardize):
- "vegetarian", "veggie" → "vegetarian"
- "vegan", "plant-based" → "vegan"
- "keto", "ketogenic" → "keto"
- "gluten intolerant" → "gluten-free"

CRITICAL DISTINCTION RULES:
🔹 PREFERENCES (general, ongoing):
   - "I love Italian food" → preferences
   - "I prefer quick meals" → preferences
   - "I like chicken" → preferences

🔹 RESTRICTIONS (permanent, medical/ethical):
   - "I'm vegetarian" → restrictions
   - "I'm allergic to nuts" → restrictions
   - "I can't eat gluten" → restrictions

🔹 INSTRUCTIONS (specific, for this meal only):
   - "I want Italian food today" → instructions
   - "Make it vegetarian this time" → instructions
   - "Use chicken and rice" → instructions
   - "Quick breakfast" → instructions
   - "Use ingredients I have at home" → instructions
   - "I have onions, bell peppers" → instructions
   - "I have beef steaks at home" → instructions
   - "I have chicken wings in the fridge" → instructions

CRITICAL RULES:
1. ONLY extract what the user explicitly says. Never fill in a field the user did not mention.
2. "allergies" goes into "restrictions" field along with dietary restrictions.
3. "preferences" = general likes (ingredients, cuisines). "instructions" = specific requests for this meal.
4. Distinguish: "I'm vegetarian" = restriction. "I want vegetarian today" = instruction.

EXAMPLES:

Example 1 - Mixed Information:
Input: "My name is John Miller. I have diabetes and Parkinson's. I love Mediterranean food but today I want a quick vegan breakfast with oats."
Output: {{
  "name": "John",
  "surname": "Miller",
  "preferences": "Mediterranean food",
  "restrictions": "",
  "health_condition": "diabetes, parkinsons_disease",
  "caretaker": "",
  "instructions": "quick vegan breakfast, use oats"
}}

Example 2 - Allergies and Spelling Errors:
Input: "I have parkenson. Can't eat tomatoes or onions. Want chicken salad."
Output: {{
  "name": "",
  "surname": "",
  "preferences": "",
  "restrictions": "tomatoes, onions",
  "health_condition": "parkinsons_disease",
  "caretaker": "",
  "instructions": "chicken salad"
}}

Example 3 - Permanent vs Temporary:
Input: "I'm vegetarian and lactose intolerant. But today I want to try Italian cuisine with pasta."
Output: {{
  "name": "",
  "surname": "",
  "preferences": "",
  "restrictions": "vegetarian, lactose-free",
  "health_condition": "",
  "caretaker": "",
  "instructions": "Italian cuisine, pasta"
}}

Example 4 - Preferences Only:
Input: "I generally prefer Asian food and quick cooking. I like tofu and vegetables."
Output: {{
  "name": "",
  "surname": "",
  "preferences": "Asian food, quick cooking, tofu, vegetables",
  "restrictions": "",
  "health_condition": "",
  "caretaker": "",
  "instructions": ""
}}

Example 5 - Complex Medical:
Input: "Sarah here. Caretaker is Mike. I have MS and high blood pressure. Allergic to peanuts. Today make something with fish and rice, no salt."
Output: {{
  "name": "Sarah",
  "surname": "",
  "preferences": "",
  "restrictions": "peanuts",
  "health_condition": "multiple_sclerosis, hypertension",
  "caretaker": "Mike",
  "instructions": "fish, rice, no salt"
}}

Example 6 - Only Instructions:
Input: "Quick dinner. Use what I have: eggs, spinach, cheese. Make it low-carb."
Output: {{
  "name": "",
  "surname": "",
  "preferences": "",
  "restrictions": "",
  "health_condition": "",
  "caretaker": "",
  "instructions": "quick dinner, use eggs, spinach, cheese, low-carb"
}}

REMEMBER: Empty string "" for any field not explicitly mentioned!"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("user", "{query}")
        ])

        # LangChain Expression Language (LCEL) pipe:
        # prompt formats the input -> LLM generates JSON text -> parser converts to dict
        return prompt | self.llm | self.parser

    def parse(self, query: str) -> UserIntent:
        """Parse user query and return structured intent.

        Sends the query through the LLM chain, which extracts structured
        fields and returns them as a JSON dict. The dict is then mapped
        to a UserIntent dataclass. If parsing fails, returns an empty UserIntent.

        Args:
            query: The user's natural language input string.

        Returns:
            UserIntent with extracted fields, or empty UserIntent on error.
        """
        try:
            # Invoke the chain: prompt -> LLM -> JSON parser -> dict
            result = self.chain.invoke({"query": query})
            return UserIntent(
                name=result.get("name", ""),
                surname=result.get("surname", ""),
                preferences=result.get("preferences", ""),
                restrictions=result.get("restrictions", ""),
                health_condition=result.get("health_condition", ""),
                caretaker=result.get("caretaker", ""),
                instructions=result.get("instructions", ""),
            )
        except Exception as e:
            print(f"Error parsing intent: {e}")
            return UserIntent()


if __name__ == "__main__":
    # Initialize Intent Parser
    intent_parser = IntentParser(model_name=LLM_MODEL)
    print("Intent Parser initialized!")

    # Test the Intent Parser
    test_query = "I have parkinson. I want to make something with chicken, eggs and salad. Quick breakfast. I can't eat tomatoes or onions."

    print(f"Query: {test_query}\n")
    intent = intent_parser.parse(test_query)
    print(intent)
