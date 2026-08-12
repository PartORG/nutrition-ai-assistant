"""
infrastructure.llm.intent_parser - Multi-provider intent extraction.

Implements IntentParserPort using LangChain.
The LLM provider (openai / groq / ollama) is controlled by the
centralized LLM_PROVIDER setting.

Migrated from components/intent_retriever.py with these changes:
    - UserIntent dataclass moved to domain/models.py
    - Fields now use list[str] instead of comma-separated strings
    - parse() is async (wraps sync chain in run_in_executor)
    - Supports openai, groq, and ollama providers
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from domain.exceptions import IntentParsingError
from domain.models import UserIntent
from infrastructure.llm.llm_builder import build_llm

logger = logging.getLogger(__name__)

# System prompt — kept intact from the original (it's well-engineered)
_SYSTEM_INSTRUCTIONS = """You are a medical nutrition data extractor for a specialized dietary assistant.
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


class _IntentSchema(BaseModel):
    """Structured schema the LLM fills via `with_structured_output()`.

    Fields mirror the keys taught by the system prompt. The list-like fields
    stay comma-separated strings here (that's the format the prompt + examples
    teach); `parse()` splits them into lists for the UserIntent domain model.
    """

    name: str = Field(default="", description="First name only, or empty if not stated")
    surname: str = Field(default="", description="Last name only, or empty if not stated")
    preferences: str = Field(
        default="", description="General long-term food preferences, comma-separated, or empty"
    )
    restrictions: str = Field(
        default="",
        description="Permanent dietary restrictions and allergies, comma-separated, or empty",
    )
    health_condition: str = Field(
        default="", description="Medical conditions in snake_case, comma-separated, or empty"
    )
    caretaker: str = Field(default="", description="Caretaker/helper name, or empty")
    instructions: str = Field(
        default="", description="Specific requests for THIS meal, comma-separated, or empty"
    )


class IntentParser:
    """Implements IntentParserPort using any supported LLM provider.

    The LLM returns comma-separated strings which are split into list[str]
    to match the domain model.
    """

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434/",
        openai_api_key: str = "",
        groq_api_key: str = "",
    ):
        # No json_mode here — with_structured_output owns the output format.
        # chat_model=True is required: with_structured_output is a chat-model
        # feature. For ollama this selects ChatOllama (the text-completion
        # OllamaLLM raises NotImplementedError); openai/groq are chat models
        # regardless, so the flag is a harmless no-op there.
        self._llm = build_llm(
            provider=provider,
            model=model,
            temperature=0,
            json_mode=False,
            chat_model=True,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_INSTRUCTIONS),
            ("user", "{query}"),
        ])
        # with_structured_output binds the schema via the provider's native
        # structured-output / tool-calling API — more reliable than parsing a
        # free-form JSON string, and works across ollama / groq / openai.
        return prompt | self._llm.with_structured_output(_IntentSchema)

    async def parse(self, query: str) -> UserIntent:
        """Parse user query into structured UserIntent.

        Runs the sync LangChain chain in a thread pool to avoid blocking.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._chain.invoke, {"query": query},
            )
            # with_structured_output returns the pydantic model (or, on some
            # providers, a dict) — normalise to a dict either way.
            data = result if isinstance(result, dict) else result.model_dump()
            return UserIntent(
                name=data.get("name", ""),
                surname=data.get("surname", ""),
                preferences=_split_csv(data.get("preferences", "")),
                restrictions=_split_csv(data.get("restrictions", "")),
                health_conditions=_split_csv(data.get("health_condition", "")),
                instructions=_split_csv(data.get("instructions", "")),
                caretaker=data.get("caretaker", ""),
            )
        except Exception as e:
            logger.error("Intent parsing failed: %s", e)
            raise IntentParsingError(f"Failed to parse intent: {e}") from e


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated string into a cleaned list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# Backward-compatible alias (so old imports still work)
OllamaIntentParser = IntentParser
