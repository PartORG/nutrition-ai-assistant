"""
Conversational Agent for Nutrition Assistant.

The agent orchestrates two main workflows:
1. Recipe Recommendations: Calls RAGPipeline and presents results
2. Recipe Selection: Saves chosen recipes to database
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# Import your existing components
from pipeline.pipeline import RAGPipeline, PipelineResult
from database.db import UserDBHandler
from database.models import RecipeHistory, NutritionHistory
from components.intent_retriever import IntentParser
from components.safety_filter import SafetyFilter
from rags.medical_rag import MedicalRAG
from rags.recipes_nutrition_rag import RecipesNutritionRAG

from settings import (
    LLM_MODEL,
    DATA_DIR,
    PDF_DIR,
    MEDICAL_VECTORSTORE_PATH,
    RECIPES_NUTRITION_VECTOR_PATH,
)

class AgentState:
    """Manages state across agent interactions.
    
    Stores the last RAG pipeline output to enable recipe selection
    without re-parsing chat history.
    """
    def __init__(self):
        self.last_recipes: List[Dict[str, Any]] = []
        self.last_raw_output: str = ""
        self.user_id: Optional[int] = None
        self.user_data: Dict[str, Any] = {}
    
    def clear_recipes(self):
        """Clear stored recipes after selection or timeout."""
        self.last_recipes = []
        self.last_raw_output = ""

# Global state instance (in production, use session management)
agent_state = AgentState()

# ----------------------------------------------------------------
# Global Pipeline Components (Singleton Pattern for Performance)
# ----------------------------------------------------------------
_pipeline_components = None

def _get_pipeline_components():
    """Lazy initialization of pipeline components (singleton pattern).
    
    Components are initialized only once on first call, then cached
    for all subsequent tool invocations. This improves performance by:
    - Avoiding repeated vectorstore loading
    - Reusing LLM connections
    - Keeping embeddings in memory
    
    Returns:
        Dict with initialized components:
        - intent_parser: IntentParser instance
        - medical_rag: MedicalRAG instance  
        - nutrition_rag: RecipesNutritionRAG instance
        - safety_filter: SafetyFilter instance
    """
    global _pipeline_components
    
    # Return cached components if already initialized
    if _pipeline_components is not None:
        print("✅ Using cached pipeline components")
        return _pipeline_components
    
    # First-time initialization
    print("\n" + "=" * 60)
    print("🔄 INITIALIZING PIPELINE COMPONENTS (ONE-TIME SETUP)")
    print("=" * 60)
    
    try:
        # ----------------------------------------------------------------
        # Step 1: Initialize Intent Parser
        # ----------------------------------------------------------------
        print("[1/4] Initializing Intent Parser...")
        intent_parser = IntentParser(model_name=LLM_MODEL)
        print("     ✓ Intent Parser ready")
        
        # ----------------------------------------------------------------
        # Step 2: Initialize Medical RAG with vectorstore
        # ----------------------------------------------------------------
        print("[2/4] Initializing Medical RAG...")
        medical_rag = MedicalRAG(
            folder_paths=[str(PDF_DIR)],
            model_name=LLM_MODEL,
            vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
            embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        )
        medical_rag.initialize(force_rebuild=False)
        print("     ✓ Medical RAG ready (vectorstore loaded)")
        
        # ----------------------------------------------------------------
        # Step 3: Initialize Nutrition RAG with vectorstore
        # ----------------------------------------------------------------
        print("[3/4] Initializing Recipes & Nutrition RAG...")
        nutrition_rag = RecipesNutritionRAG(
            data_folder=str(DATA_DIR),
            model_name=LLM_MODEL,
            vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH),
        )
        nutrition_rag.initialize()
        print("     ✓ Nutrition RAG ready (vectorstore loaded)")
        
        # ----------------------------------------------------------------
        # Step 4: Initialize Safety Filter
        # ----------------------------------------------------------------
        # print("[4/4] Initializing Safety Filter...")
        # safety_filter = SafetyFilter(model_name=LLM_MODEL, debug=True)
        # print("     ✓ Safety Filter ready")
        
        # Cache components for reuse
        _pipeline_components = {
            'intent_parser': intent_parser,
            'medical_rag': medical_rag,
            'nutrition_rag': nutrition_rag,
            'safety_filter': None, # TODO: add again after test
        }
        
        print("=" * 60)
        print("✅ ALL PIPELINE COMPONENTS INITIALIZED & CACHED")
        print("=" * 60 + "\n")
        
        return _pipeline_components
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n❌ FATAL ERROR during component initialization:\n{error_details}")
        raise RuntimeError(f"Failed to initialize pipeline components: {str(e)}")

# ----------------------------------------------------------------
# Recipe Parsing Functions (JSON-based)
# ----------------------------------------------------------------

def parse_recipes_from_json(json_text: str) -> List[Dict[str, Any]]:
    """Parse JSON recipes from RAG pipeline output.
    
    The pipeline returns recipes in JSON format. This function extracts
    and flattens the structure for use in database operations.
    
    Args:
        json_text: JSON string from pipeline (may contain surrounding text)
    
    Returns:
        List of recipe dictionaries with flattened structure:
        - recipe_number: int (1-based index)
        - recipe_name: str
        - servings: int
        - prep_time: str (e.g., "15 minutes")
        - cook_time: str
        - ingredients: str (newline-separated list)
        - cook_instructions: str (newline-separated steps)
        - calories: float
        - protein/carbs/fat/fiber/sugar/sodium: float
        - why_recommended: str
    """
    try:
        # Extract JSON if wrapped in code blocks or extra text
        json_match = re.search(r'\{[\s\S]*"recipes"[\s\S]*\}', json_text)
        if json_match:
            json_text = json_match.group(0)
        
        # Parse JSON
        data = json.loads(json_text)
        recipes = []
        
        for idx, recipe in enumerate(data.get("recipes", []), start=1):
            # Flatten ingredients array to string for DB storage
            ingredients_list = recipe.get("ingredients", [])
            if isinstance(ingredients_list, list):
                if ingredients_list and isinstance(ingredients_list[0], str):
                    # Format: ["200g chicken", "150ml oil"]
                    ingredients_str = "\n".join([f"- {ing}" for ing in ingredients_list])
                else:
                    # Format: [{"item": "chicken", "amount": 200, "unit": "g"}]
                    ingredients_str = "\n".join([
                        f"- {ing.get('amount', '')}{ing.get('unit', '')} {ing.get('item', '')}"
                        for ing in ingredients_list
                    ])
            else:
                ingredients_str = str(ingredients_list)
            
            # Get instructions (already string with \n separators from pipeline)
            instructions = recipe.get("cook_instructions", "")
            
            # Extract nutrition data
            nutrition = recipe.get("nutrition", {})
            
            # Build flattened recipe dictionary
            recipes.append({
                "recipe_number": idx,
                "recipe_name": recipe.get("name", f"Recipe {idx}"),
                "servings": recipe.get("servings", 2),
                "prep_time": recipe.get("prep_time", "Unknown"),
                "cook_time": recipe.get("cook_time", "Unknown"),
                "ingredients": ingredients_str,
                "cook_instructions": instructions,
                "calories": nutrition.get("calories", 0),
                "protein": nutrition.get("protein_g", 0),
                "carbs": nutrition.get("carbs_g", 0),
                "fat": nutrition.get("fat_g", 0),
                "fiber": nutrition.get("fiber_g", 0),
                "sugar": nutrition.get("sugar_g", 0),
                "sodium": nutrition.get("sodium_mg", 0),
                "why_recommended": recipe.get("why_recommended", ""),
            })
        
        return recipes
    
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON Parse Error: {e}")
        print(f"Raw text (first 500 chars): {json_text[:500]}")
        return []
    except Exception as e:
        print(f"⚠️ Recipe parsing failed: {e}")
        return []


def convert_recipes_to_markdown(recipes: List[Dict[str, Any]]) -> str:
    """Convert structured recipes to beautiful Markdown for user display.
    
    Takes the flattened recipe dictionaries from parse_recipes_from_json()
    and formats them into human-readable Markdown with clear sections.
    
    Args:
        recipes: List of recipe dictionaries with fields:
                 recipe_number, recipe_name, servings, prep_time, cook_time,
                 ingredients, cook_instructions, nutrition values, why_recommended
    
    Returns:
        Formatted Markdown string with all recipes, or message if empty
    """
    if not recipes:
        return "No recipes available."
    
    markdown_parts = []
    
    for recipe in recipes:
        # Build nutrition facts section
        nutrition_lines = [
            f"**📊 Nutrition Facts (per serving for {recipe['servings']} servings):**",
            f"- Calories: {recipe['calories']} kcal",
            f"- Protein: {recipe['protein']}g",
            f"- Carbs: {recipe['carbs']}g",
            f"- Fat: {recipe['fat']}g",
            f"- Fiber: {recipe['fiber']}g",
            f"- Sugar: {recipe['sugar']}g",
            f"- Sodium: {recipe['sodium']}mg",
        ]
        
        # Build time section (only if values available)
        time_parts = []
        if recipe.get('prep_time') and recipe['prep_time'] != "Unknown":
            time_parts.append(f"Prep: {recipe['prep_time']}")
        if recipe.get('cook_time') and recipe['cook_time'] != "Unknown":
            time_parts.append(f"Cook: {recipe['cook_time']}")
        time_str = " | ".join(time_parts) if time_parts else "Time not specified"
        
        # Why recommended section (optional field from pipeline)
        why_section = ""
        if recipe.get('why_recommended'):
            why_section = f"\n**💡 Why recommended:** {recipe['why_recommended']}\n"
        
        # Combine all sections into recipe markdown
        md = f"""### Recipe {recipe['recipe_number']}: {recipe['recipe_name']}
{why_section}
{chr(10).join(nutrition_lines)}

**🛒 Ingredients:**
{recipe['ingredients']}

**👨‍🍳 Instructions:**
{recipe['cook_instructions']}

**⏱️ Time:** {time_str}

---
"""
        markdown_parts.append(md)
    
    return "\n".join(markdown_parts)


def find_recipe_by_name(recipe_name: str, recipes: List[Dict[str, Any]]) -> Optional[int]:
    """Find recipe number by fuzzy name matching.
    
    Supports multiple matching strategies:
    1. Exact match (case-insensitive)
    2. Partial match (name contains search term)
    3. Fuzzy match (any word overlaps)
    
    Args:
        recipe_name: Name mentioned by user (e.g., "the salmon", "Grilled Salmon")
        recipes: List of recipes from agent_state.last_recipes
    
    Returns:
        Recipe number (1-based index) or None if not found
    
    Examples:
        User says "the Salmon" → Finds "Grilled Salmon with Lemon"
        User says "fish tacos" → Finds "Spicy Fish Tacos"
    """
    recipe_name_lower = recipe_name.lower().strip()
    
    # Strategy 1: Exact match (case-insensitive)
    for recipe in recipes:
        if recipe_name_lower == recipe["recipe_name"].lower():
            return recipe["recipe_number"]
    
    # Strategy 2: Partial match (contains)
    for recipe in recipes:
        if recipe_name_lower in recipe["recipe_name"].lower():
            return recipe["recipe_number"]
        if recipe["recipe_name"].lower() in recipe_name_lower:
            return recipe["recipe_number"]
    
    # Strategy 3: Fuzzy match (any word matches)
    search_words = set(recipe_name_lower.split())
    for recipe in recipes:
        recipe_words = set(recipe["recipe_name"].lower().split())
        # If any word overlaps, consider it a match
        if search_words & recipe_words:  # Set intersection
            return recipe["recipe_number"]
    
    # No match found
    return None


# ----------------------------------------------------------------
# Tool Input Schemas (for Structured Tools)
# ----------------------------------------------------------------

class RAGPipelineInput(BaseModel):
    """Input schema for RAG pipeline tool."""
    query: str = Field(description="User's recipe request or search query")

class DatabaseHandlingInput(BaseModel):
    """Input schema for database handling tool.
    
    Supports two selection methods:
    - By number: recipe_number=2
    - By name: recipe_name="Grilled Salmon"
    
    If both provided, recipe_number takes priority.
    """
    recipe_number: int = Field(
        default=None, 
        description="Recipe number from previous recommendations (1, 2, 3, etc.). Use this if user says 'recipe 2' or 'the second one'."
    )
    recipe_name: str = Field(
        default=None,
        description="Recipe name from previous recommendations. Use this if user says 'the Grilled Salmon' or 'I want the salmon'. Extract the recipe name from user's input."
    )
    rating: int = Field(
        default=None, 
        description="Optional rating from 1-5 stars"
    )


def rag_pipeline_tool_func(query: str) -> str:
    """Tool: Fetch recipe recommendations from RAG pipeline.
    
    This tool orchestrates the complete recipe recommendation workflow:
    1. Gets cached pipeline components (or initializes on first call)
    2. Creates RAG pipeline with cached components
    3. Processes user's unmodified query through pipeline
    4. Pipeline returns recipes in JSON format
    5. Parses JSON to structured data
    6. Stores recipes in agent_state for save_recipe tool
    7. Converts JSON to user-friendly Markdown
    8. Adds header, disclaimer, and action footer
    
    Args:
        query: User's recipe request (passed through UNCHANGED from agent)
    
    Returns:
        Formatted Markdown string with:
        - Header with recipe count
        - All recipes in readable format
        - Medical disclaimer
        - Action suggestions footer
    """
    try:
        print("\n" + "=" * 60)
        print("PROCESSING QUERY...")
        print("=" * 60)
        print(f"Query: {query}\n")
        
        # ----------------------------------------------------------------
        # Step 1: Get cached pipeline components (singleton pattern)
        # ----------------------------------------------------------------
        components = _get_pipeline_components()
        
        # ----------------------------------------------------------------
        # Step 2: Create RAG pipeline with cached components
        # ----------------------------------------------------------------
        print("Creating RAG Pipeline with cached components...")
        pipeline = RAGPipeline(
            intent_parser=components['intent_parser'],
            medical_rag=components['medical_rag'],
            nutrition_rag=components['nutrition_rag'],
            safety_filter=components['safety_filter'],
        )
        
        # ----------------------------------------------------------------
        # Step 3: Get user medical data from agent state (if logged in)
        # ----------------------------------------------------------------
        user_data = agent_state.user_data if agent_state.user_data else None
        
        # ----------------------------------------------------------------
        # Step 4: Process query through complete RAG pipeline
        # ----------------------------------------------------------------
        result: PipelineResult = pipeline.process(
            user_query=query,
            user_data=user_data
        )

        # ----------------------------------------------------------------
        # Step 5: Extract JSON output from pipeline result
        # ----------------------------------------------------------------
        # Priority: safety_result.safe_recipes_markdown > llm_recommendation
        # if result.safety_result and result.safety_result.safe_recipes_markdown:
        #     json_output = result.safety_result.safe_recipes_markdown
        # else:
        json_output = result.llm_recommendation
        
        # ----------------------------------------------------------------
        # Step 6: Parse JSON to structured recipe dictionaries
        # ----------------------------------------------------------------
        parsed_recipes = parse_recipes_from_json(json_output)
        
        # Store recipes in agent state for save_recipe tool access
        agent_state.last_recipes = parsed_recipes
        agent_state.last_raw_output = json_output
        
        # Handle case when no recipes were parsed successfully
        if not parsed_recipes:
            return """⚠️ **No recipes found.**

This could be because:
- Your requirements are too specific
- No recipes in our database match all constraints
- There was a parsing error

**Suggestions:**
- Try a more general query (e.g., "healthy dinner" instead of "gluten-free vegan keto dinner with 15 ingredients")
- Specify fewer restrictions
- Ask for a different type of meal

Would you like to try another search?"""
        
        # ----------------------------------------------------------------
        # Step 7: Convert parsed recipes to user-friendly Markdown format
        # ----------------------------------------------------------------
        markdown_output = convert_recipes_to_markdown(parsed_recipes)
        
        # Build complete user response with header, content, disclaimer, footer
        recipe_count = len(parsed_recipes)
        recipe_word = "recipe" if recipe_count == 1 else "recipes"
        
        header = f"🍽️ **Here are {recipe_count} {recipe_word} based on your request:**\n\n"
        
        disclaimer = f"""
---
⚠️ **Medical Disclaimer:** These are general recommendations based on available nutritional data. Always consult with healthcare providers or registered dietitians before making significant dietary changes, especially if you have medical conditions or specific health goals.
"""
        
        footer = f"""
💬 **What would you like to do?**
- **Cook one?** Say "I'll cook recipe 2" or "I want the {parsed_recipes[0]['recipe_name']}"
- **Need more options?** Ask for different criteria (e.g., "Show me vegetarian options")
- **Have questions?** Just ask me anything about nutrition or cooking!
"""
        
        # Return complete formatted response
        return header + markdown_output + disclaimer + footer
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ ERROR in rag_pipeline_tool_func:\n{error_details}")
        return f"⚠️ **Error processing your request:** {str(e)}\n\nPlease try rephrasing your query or contact support if the issue persists."


def database_handling_tool_func(recipe_number: int = None, recipe_name: str = None, rating: int = None) -> str:
    """Tool: Save selected recipe to database.
    
    Supports selection by:
    - Number: recipe_number=2 → Recipe 2 from last search
    - Name: recipe_name="Grilled Salmon" → Tool finds matching recipe automatically
    
    Args:
        recipe_number: Recipe number from previous recommendations (1-based index)
        recipe_name: Recipe name from previous recommendations (fuzzy matched)
        rating: Optional rating from 1-5 stars
    
    Returns:
        Confirmation message with recipe details or error message
    """
    try:
        # ----------------------------------------------------------------
        # Name-to-Number Conversion (if recipe_name provided)
        # ----------------------------------------------------------------
        if recipe_name and not recipe_number:
            # User selected by name, convert to number using fuzzy matching
            found_number = find_recipe_by_name(recipe_name, agent_state.last_recipes)
            
            if found_number:
                recipe_number = found_number
                print(f"🔍 Matched recipe name '{recipe_name}' to Recipe {recipe_number}")
            else:
                # Name not found in last recipes - provide helpful error
                if agent_state.last_recipes:
                    available_names = [r['recipe_name'] for r in agent_state.last_recipes]
                    available_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(available_names)])
                    return f"""⚠️ **Recipe name '{recipe_name}' not found.**

Available recipes from your last search:
{available_list}

Please select by number (e.g., "I'll cook recipe 2") or use the exact recipe name."""
                else:
                    return """⚠️ **No recipes available.**

Please search for recipes first!"""
        
        # ----------------------------------------------------------------
        # Validate that at least one selection method was used
        # ----------------------------------------------------------------
        if not recipe_number and not recipe_name:
            return """⚠️ **No recipe selected.**

Please specify which recipe you want:
- By number: "I'll cook recipe 2"
- By name: "I want the Grilled Salmon"
"""
        
        # ----------------------------------------------------------------
        # Check if we have recipes stored from previous search
        # ----------------------------------------------------------------
        if not agent_state.last_recipes:
            return """⚠️ **No recipes available to save.**

Please search for recipes first by asking for recommendations (e.g., "I need dinner ideas with chicken").

Once you see recipe options, you can select one to save!"""
        
        # ----------------------------------------------------------------
        # Validate recipe_number range
        # ----------------------------------------------------------------
        if recipe_number and (recipe_number < 1 or recipe_number > len(agent_state.last_recipes)):
            available = ", ".join(str(i) for i in range(1, len(agent_state.last_recipes) + 1))
            return f"""⚠️ **Invalid recipe number.**

Available recipes from your last search: {available}

Please select one of these numbers (e.g., "I'll cook recipe 2")."""
        
        # ----------------------------------------------------------------
        # Validate user_id (require login in production)
        # ----------------------------------------------------------------
        if not agent_state.user_id:
            return """⚠️ **No user logged in.**

Cannot save recipe without user authentication. Please log in first!"""
        
        # ----------------------------------------------------------------
        # Get selected recipe (convert 1-based to 0-based index)
        # ----------------------------------------------------------------
        recipe = agent_state.last_recipes[recipe_number - 1]
        
        # ----------------------------------------------------------------
        # Initialize database handler
        # ----------------------------------------------------------------
        db = UserDBHandler()
        
        # ----------------------------------------------------------------
        # Create RecipeHistory entry with new JSON fields
        # ----------------------------------------------------------------
        recipe_history = RecipeHistory(
            recipe_name=recipe["recipe_name"],
            servings=recipe["servings"],
            prep_time=recipe["prep_time"],
            ingredients=recipe["ingredients"],
            cook_instructions=recipe["cook_instructions"],
            rating=rating,
            user_id=agent_state.user_id,
        )
        
        # ----------------------------------------------------------------
        # Create NutritionHistory entry with all nutrition fields
        # ----------------------------------------------------------------
        nutrition_history = NutritionHistory(
            calories=recipe["calories"],
            protein=recipe["protein"],
            carbs=recipe["carbs"],
            fat=recipe["fat"],
            fiber=recipe["fiber"],
            sugar=recipe["sugar"],
            sodium=recipe["sodium"],
            user_id=agent_state.user_id,
        )
        
        # ----------------------------------------------------------------
        # Save to database (two-step process: recipe first, then nutrition)
        # ----------------------------------------------------------------
        recipe_id = db.create_recipe_history(recipe_history)
        db.create_nutrition_history(nutrition_history, recipe_id)
        
        # ----------------------------------------------------------------
        # Build success confirmation message
        # ----------------------------------------------------------------
        rating_text = f" with {rating}⭐ rating" if rating else ""
        
        # Extract key info for confirmation
        cook_time_info = f" | Cook: {recipe['cook_time']}" if recipe.get('cook_time') and recipe['cook_time'] != "Unknown" else ""
        
        confirmation = f"""✅ **Recipe saved successfully{rating_text}!**

**📋 Saved Recipe Details:**
- **Name:** {recipe['recipe_name']}
- **Servings:** {recipe['servings']}
- **Time:** Prep: {recipe['prep_time']}{cook_time_info}
- **Calories:** {recipe['calories']} kcal per serving

**🍳 What's next?**
- Ready to cook? Check your saved recipes anytime in your cooking history!
- Want more recipes? Just ask! (e.g., "Show me vegetarian options")
- Need cooking tips? I'm here to help!

Enjoy your meal! 🍽️"""
        
        # Clear recipes from state after successful save
        agent_state.clear_recipes()
        
        return confirmation
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ DATABASE ERROR in database_handling_tool_func:\n{error_details}")
        return f"""⚠️ **Error saving recipe:** {str(e)}

This might be a temporary issue. Please try again, or contact support if the problem persists."""


def create_nutrition_agent(user_id: Optional[int] = None, user_data: Optional[Dict] = None) -> AgentExecutor:
    """Initialize the nutrition assistant agent.
    
    Args:
        user_id: Optional user ID for database operations
        user_data: Optional user context (preferences, health_condition, etc.)
    
    Returns:
        Configured AgentExecutor ready for chat
    """
    if user_id:
        agent_state.user_id = user_id
    if user_data:
        agent_state.user_data = user_data
    
    llm = ChatOllama(
        model="llama3:8b",
        temperature=0,
        ollama_base_url="http://localhost:11434",
    )

    tools = [
        StructuredTool.from_function(
            func=rag_pipeline_tool_func,
            name="search_recipes",
            description="Search for recipe recommendations based on user query. Use this when user asks for recipes or meal suggestions.",
            args_schema=RAGPipelineInput,
        ),
        StructuredTool.from_function(
            func=database_handling_tool_func,
            name="save_recipe",
            description="Save a selected recipe to user's cooking history. Use when user chooses a recipe by number.",
            args_schema=DatabaseHandlingInput,
        ),
    ]
    
    system_message = """You are a helpful nutrition assistant that helps users find and save recipes.

You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

CRITICAL WORKFLOW:
1. When user asks for recipes or meal suggestions → ALWAYS use 'search_recipes' tool
2. PASS THE USER'S COMPLETE ORIGINAL QUERY to search_recipes WITHOUT MODIFICATION
3. DO NOT rephrase, summarize, or interpret the query - use it EXACTLY as user wrote it
4. AFTER tool returns Observation → IMMEDIATELY use "Final Answer" with the COMPLETE tool output (DO NOT modify or summarize!)
5. After showing recipes, ask if user wants to cook one or needs more suggestions
6. When user selects a recipe by NUMBER (e.g., "I'll cook recipe 2", "recipe 3", "the second one") → use 'save_recipe' tool, then "Final Answer" with confirmation
7. For general questions not about recipes → answer directly with "Final Answer"

RESPONSE FORMAT (follow exactly):

Step 1 - Call Tool with ORIGINAL query:
```json
{{
  "action": "search_recipes",
  "action_input": {{"query": "[EXACT USER QUERY - DO NOT CHANGE]"}}
}}
```

Step 2 - After Observation, IMMEDIATELY use Final Answer:
```json
{{
  "action": "Final Answer",
  "action_input": "[PASTE COMPLETE OBSERVATION TEXT HERE WITHOUT ANY CHANGES]"
}}
```

WORKFLOW EXAMPLES:

Example 1 - Recipe Search:
User: "I have diabetes and need dinner ideas with chicken"
Thought: User wants recipes, I must use search_recipes tool with their EXACT query
Action:
```json
{{
  "action": "search_recipes",
  "action_input": {{"query": "I have diabetes and need dinner ideas with chicken"}}
}}
```
Observation: [Tool returns full recipes with markdown formatting]
Thought: I received the recipes, now I must show them to user with Final Answer
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "[COMPLETE OBSERVATION TEXT] Would you like to cook one of these, or see more options?"
}}
```

Example 2 - Recipe Selection:
User: "I'll cook recipe 2"
Thought: User selected recipe number 2, I must save it
Action:
```json
{{
  "action": "save_recipe",
  "action_input": {{"recipe_number": 2}}
}}
```
Observation: Recipe saved successfully...
Thought: Recipe was saved, confirm to user
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "✅ Recipe saved! Enjoy cooking!"
}}
```
Example 3 - Another Search (EXACT WORDING):
User: "show me low-carb breakfast with eggs"
Thought: User wants recipes, use their EXACT words
Action:
```json
{{
  "action": "search_recipes",
  "action_input": {{"query": "show me low-carb breakfast with eggs"}}
}}
```

Example 3a - Recipe Selection by Name (NEW METHOD):
User: "I want to cook the Grilled Salmon"
Thought: User selected recipe by name, I should extract the name and pass it to save_recipe tool
Action:
```json
{{
  "action": "save_recipe",
  "action_input": {{"recipe_name": "Grilled Salmon"}}
}}
```
Note: Tool will automatically find matching recipe number from previous search using fuzzy matching
Observation: 🔍 Matched recipe name 'Grilled Salmon' to Recipe 2
✅ Recipe saved successfully!...
Thought: Recipe was saved, confirm to user
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "✅ Grilled Salmon saved! Enjoy cooking!"
}}
```

Example 3b - Partial Name Match:
User: "I'll cook the salmon"
Thought: User mentioned "salmon", I'll pass this partial name to the tool
Action:
```json
{{
  "action": "save_recipe",
  "action_input": {{"recipe_name": "salmon"}}
}}
```
Note: Tool uses fuzzy matching to find recipes containing "salmon" in their name
Observation: 🔍 Matched recipe name 'salmon' to Recipe 2
✅ Recipe saved successfully!...
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "✅ Grilled Salmon with Lemon saved! Enjoy cooking!"
}}
```

Example 4 - General Question:
User: "What's the difference between protein and carbs?"
Thought: This is a general nutrition question, no tool needed
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "Protein is essential for building muscles and repairing tissue, found in meat, fish, eggs, and legumes. Carbohydrates provide energy, found in grains, fruits, and vegetables. Both are important macronutrients with different roles."
}}
```

CRITICAL RULES:
- NEVER modify the user's query when calling search_recipes
- NEVER rephrase "I have diabetes and want fish" to "diabetic fish recipes"
- USE the user's EXACT words - they contain important context!
- Your Intent Parser and Medical RAG will handle the interpretation internally
- NEVER repeat the same tool call twice - if you already called it, move to Final Answer
- NEVER summarize or modify the Observation from search_recipes - pass it through COMPLETELY
- Use "Final Answer" after EVERY tool call (search_recipes AND save_recipe)
- Parse recipe numbers from natural language (e.g., "second one" = recipe 2, "the first recipe" = recipe 1)
- Parse recipe names from natural language and pass to save_recipe tool:
  * "I want the Grilled Salmon" → {{"recipe_name": "Grilled Salmon"}}
  * "cook the salmon" → {{"recipe_name": "salmon"}}
  * Tool will handle fuzzy matching automatically - do NOT try to find the number yourself
- PREFER recipe_name over recipe_number when user mentions a dish name (even partial names are OK)
- Always show the COMPLETE recipe details including why_recommended if available
- Be conversational and friendly in your responses
- If user asks for recipe by name but you cannot find it in last_recipes, politely ask them to specify the recipe number instead"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "{agent_scratchpad}"),
    ])
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )
    
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False,
    )
    
    return agent_executor


def chat_loop():
    """Simple CLI chat interface for testing the agent."""
    print("=" * 60)
    print("NUTRITION ASSISTANT AGENT")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    # Initialize agent (for testing without user login)
    test_user_data = {
        "name": "Test",
        "surname": "User",
        "health_condition": "diabetes",
        "age": 35,
        "gender": "male",
    }
    
    agent = create_nutrition_agent(user_id=1, user_data=test_user_data)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    chat_loop()