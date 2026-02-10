from typing import List
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM

from pipeline.config import LLM_MODEL

@dataclass
class UserIntent:
    """Structured representation of parsed user intent.
    
    Aligned with the User database model fields.
    Fields not found in the query remain as empty strings.
    """
    name: str = ""
    surname: str = ""
    preferences: str = ""       # ingredients, cooking style, cuisine
    restrictions: str = ""      # dietary restrictions (keto, vegan, etc.)
    health_condition: str = ""  # medical conditions (diabetes, hypertension, etc.)
    caretaker: str = ""
    
    # Parsed lists used internally by the pipeline (not stored in DB)
    allergies: List[str] = field(default_factory=list)
    
    def __repr__(self):
        fields = {
            "name": self.name,
            "surname": self.surname,
            "preferences": self.preferences,
            "restrictions": self.restrictions,
            "health_condition": self.health_condition,
            "caretaker": self.caretaker,
            "allergies": self.allergies,
        }
        lines = [f"  {k}: {v}" for k, v in fields.items() if v]
        return "UserIntent:\n" + "\n".join(lines) if lines else "UserIntent: (empty)"
    
    @property
    def medical_conditions_list(self) -> List[str]:
        """Return health_condition as a list for pipeline processing."""
        return [c.strip() for c in self.health_condition.split(",") if c.strip()]
    
    @property
    def ingredients_list(self) -> List[str]:
        """Extract ingredient names from preferences string."""
        return [p.strip() for p in self.preferences.split(",") if p.strip()]


class IntentParser:
    """Parses user queries to extract structured intent using LLM."""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0,
            format="json"
        )
        self.parser = JsonOutputParser()
        self.chain = self._build_chain()
    
    def _build_chain(self):
        system_instructions = """You are a medical nutrition data extractor.
Your job: read the user query and extract ONLY information that is explicitly stated.
Do NOT add, guess, or infer anything that the user did not say.

Return a JSON object with these keys:
- "name" (string): user's first name, or "" if not stated
- "surname" (string): user's last name, or "" if not stated
- "preferences" (string): foods and cooking style the user wants, comma-separated, or "" if not stated
- "restrictions" (string): dietary restrictions the user follows, comma-separated, or "" if not stated
- "health_condition" (string): medical conditions the user has, comma-separated, or "" if not stated. Use snake_case. Normalize common phrases: "high blood pressure" -> "hypertension", "sugar problem" -> "diabetes"
- "caretaker" (string): caretaker name, or "" if not stated
- "allergies" (array of strings): foods the user cannot eat or is allergic to, or [] if not stated

CRITICAL RULES:
1. ONLY extract what the user explicitly says. Never fill in a field the user did not mention.
2. If the user does not mention a field, return "" for strings or [] for arrays.
3. Do not confuse allergies with preferences. "I can't eat X" = allergy. "I want X" = preference.

EXAMPLE:
Input: "I'm John. I have diabetes. I want chicken and rice. I'm allergic to peanuts."
Output: {{"name": "John", "surname": "", "preferences": "chicken, rice", "restrictions": "", "health_condition": "diabetes", "caretaker": "", "allergies": ["peanuts"]}}

EXAMPLE:
Input: "Quick vegan lunch with tofu"
Output: {{"name": "", "surname": "", "preferences": "tofu, quick lunch", "restrictions": "vegan", "health_condition": "", "caretaker": "", "allergies": []}}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("user", "{query}")
        ])
        
        return prompt | self.llm | self.parser
    
    def parse(self, query: str) -> UserIntent:
        """Parse user query and return structured intent."""
        try:
            result = self.chain.invoke({"query": query})
            return UserIntent(
                name=result.get("name", ""),
                surname=result.get("surname", ""),
                preferences=result.get("preferences", ""),
                restrictions=result.get("restrictions", ""),
                health_condition=result.get("health_condition", ""),
                caretaker=result.get("caretaker", ""),
                allergies=result.get("allergies", []),
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
