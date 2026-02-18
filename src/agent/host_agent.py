"""
Conversational Agent for Nutrition Assistant.

The agent orchestrates two main workflows:
1. Recipe Recommendations: Calls RAGPipeline and presents results
2. Recipe Selection: Saves chosen recipes to database
"""

import re
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

# from dotenv import load_dotenv
# from langchain_groq import ChatGroq

# Import your existing components
from pipeline.pipeline import RAGPipeline, PipelineResult
from database.db import UserDBHandler
from database.models import RecipeHistory, NutritionHistory
from settings import LLM_MODEL

# load_dotenv()

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

def parse_recipes_from_markdown(markdown_text: str) -> List[Dict[str, Any]]:
    """Extract structured recipe data from RAG pipeline markdown output."""
    recipes = []
    
    # ✅ NEUES PATTERN (angepasst an tatsächlichen Output)
    pattern = r"\*\*Recipe (\d+): (.+?)\*\*\s+Servings: (\d+)\s+Prep Time: (.+?)\s+(?:Cook Time: .+?\s+)?(?:Total Time: .+?\s+)?Ingredients:\s+(.+?)\s+Instructions:\s+(.+?)\s+Nutrition Facts \(per serving\):\s+- Calories: (\d+)\s+- Protein: ([\d.]+)g\s+- Fat: ([\d.]+)g\s+- Saturated Fat: ([\d.]+)g\s+- Cholesterol: ([\d.]+)mg\s+- Sodium: ([\d.]+)mg\s+- Carbohydrates: ([\d.]+)g\s+- Fiber: ([\d.]+)g\s+- Sugars: ([\d.]+)g"
    
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    
    if not matches:
        print(f"⚠️ WARNING: Could not parse recipes. Trying fallback parser...")
        # FALLBACK: Simple split by "**Recipe"
        return parse_recipes_fallback(markdown_text)
    
    for match in matches:
        (recipe_num, name, servings, prep_time, ingredients, 
         instructions, calories, protein, fat, sat_fat, cholesterol, 
         sodium, carbs, fiber, sugars) = match
        
        recipe = {
            "recipe_number": int(recipe_num),
            "recipe_name": name.strip(),
            "servings": int(servings),
            "prep_time": prep_time.strip(),
            "ingredients": ingredients.strip(),
            "cook_instructions": instructions.strip(),
            "calories": float(calories),
            "protein": float(protein),
            "fat": float(fat),
            "carbs": float(carbs),
            "fiber": float(fiber),
            "sugar": float(sugars),
            "sodium": float(sodium),
        }
        recipes.append(recipe)
    
    return recipes


def parse_recipes_fallback(markdown_text: str) -> List[Dict[str, Any]]:
    """Fallback parser for when regex fails."""
    recipes = []
    
    # Split by "**Recipe X:"
    recipe_blocks = re.split(r'\*\*Recipe \d+:', markdown_text)[1:]  # Skip first empty part
    
    for idx, block in enumerate(recipe_blocks, start=1):
        # Extract basic info with simpler patterns
        name_match = re.search(r'^(.+?)\*\*', block)
        servings_match = re.search(r'Servings: (\d+)', block)
        prep_time_match = re.search(r'Prep Time: (.+?)(?:\n|$)', block)
        
        # Extract ingredients block
        ingredients_match = re.search(r'Ingredients:\s*\n(.+?)\n\nInstructions:', block, re.DOTALL)
        
        # Extract instructions block
        instructions_match = re.search(r'Instructions:\s*\n(.+?)\n\nNutrition Facts', block, re.DOTALL)
        
        # Extract nutrition
        calories_match = re.search(r'Calories: (\d+)', block)
        protein_match = re.search(r'Protein: ([\d.]+)g', block)
        carbs_match = re.search(r'Carbohydrates: ([\d.]+)g', block)
        fat_match = re.search(r'Fat: ([\d.]+)g', block)
        
        if name_match and servings_match:
            recipe = {
                "recipe_number": idx,
                "recipe_name": name_match.group(1).strip(),
                "servings": int(servings_match.group(1)),
                "prep_time": prep_time_match.group(1).strip() if prep_time_match else "Unknown",
                "ingredients": ingredients_match.group(1).strip() if ingredients_match else "",
                "cook_instructions": instructions_match.group(1).strip() if instructions_match else "",
                "calories": float(calories_match.group(1)) if calories_match else 0,
                "protein": float(protein_match.group(1)) if protein_match else 0,
                "carbs": float(carbs_match.group(1)) if carbs_match else 0,
                "fat": float(fat_match.group(1)) if fat_match else 0,
                "fiber": 0,
                "sugar": 0,
                "sodium": 0,
            }
            recipes.append(recipe)
    
    return recipes

# ----------------------------------------------------------------
# Tool Input Schemas (for Structured Tools)
# ----------------------------------------------------------------

class RAGPipelineInput(BaseModel):
    """Input schema for RAG pipeline tool."""
    query: str = Field(description="User's recipe request or search query")

class DatabaseHandlingInput(BaseModel):
    """Input schema for database handling tool."""
    recipe_number: int = Field(description="Recipe number from previous recommendations (1, 2, 3, etc.)")
    rating: int = Field(default=None, description="Optional rating from 1-5 stars")


def rag_pipeline_tool_func(query: str) -> str:
    """Tool: Fetch recipe recommendations from RAG pipeline.
    
    This tool should ONLY be called when:
    - User asks for NEW recipe suggestions
    - User wants MORE recommendations after seeing previous results
    
    Args:
        query: User's recipe request (preferences will be added automatically)
    
    Returns:
        Formatted markdown text with recipe recommendations
    """
    try:
        # Get pipeline instance (singleton pattern recommended)
        from pipeline.pipeline import RAGPipeline
        from components.intent_retriever import IntentParser
        from components.safety_filter import SafetyFilter
        from rags.recipes_nutrition_rag import RecipesNutritionRAG
        from rags.medical_rag import MedicalRAG
        from settings import (
            PDF_DIR, DATA_DIR,
            MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH,
        )
        
        # Initialize components (in production, do this once at startup)
        intent_parser = IntentParser(model_name=LLM_MODEL)
        medical_rag = MedicalRAG(
            folder_paths=[str(PDF_DIR)],
            model_name=LLM_MODEL,
            vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
        )
        medical_rag.initialize(force_rebuild=False)
        
        nutrition_rag = RecipesNutritionRAG(
            data_folder=str(DATA_DIR),
            model_name=LLM_MODEL,
            vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH),
        )
        nutrition_rag.initialize()
        
        safety_filter = SafetyFilter(model_name=LLM_MODEL, debug=False)
        
        pipeline = RAGPipeline(
            intent_parser=intent_parser,
            medical_rag=medical_rag,
            nutrition_rag=nutrition_rag,
            safety_filter=safety_filter,
        )
        
        # Run pipeline with user data from state
        result: PipelineResult = pipeline.process(
            user_query=query,
            user_data=agent_state.user_data if agent_state.user_data else None
        )
        
        # Extract formatted output
        if result.safety_result and result.safety_result.safe_recipes_markdown:
            output_text = result.safety_result.safe_recipes_markdown
        else:
            output_text = result.llm_recommendation
        
        # Parse recipes and store in state
        parsed_recipes = parse_recipes_from_markdown(output_text)
        agent_state.last_recipes = parsed_recipes
        agent_state.last_raw_output = output_text
        
        # Output footer:
        if len(parsed_recipes) > 0:
            footer = f"\n\n---\n✨ **Found {len(parsed_recipes)} recipes above!**\n\n💬 **What would you like to do?**\n- Cook one? (e.g., 'I'll cook recipe 2')\n- Need more options? (e.g., 'Show me more vegetarian recipes')\n- Have questions? Just ask!"
        else:
            footer = "\n\n---\n⚠️ No recipes found. Try rephrasing your request (e.g., 'low-carb dinner with fish')."
        
        return output_text + footer
        
    except Exception as e:
        return f"Error: {str(e)}"
    

def database_handling_tool_func(recipe_number: int, rating: int = None) -> str:
    """Tool: Save selected recipe to database.
    
    This tool should ONLY be called when:
    - User explicitly selects a recipe by number (e.g., "I'll cook recipe 2")
    - User wants to rate a recipe they cooked
    
    Args:
    recipe_number: Recipe number from previous recommendations (1, 2, 3, etc.)
    rating: Optional rating from 1-5 stars (default: None)
    
    Returns:
        Confirmation message or error
    """
    try:
        # Validate
        if recipe_number <= 0:
            return "Please specify a valid recipe number (e.g., 'recipe_number:2')."
        
        if not agent_state.last_recipes:
            return "No recipes available. Please search for recipes first using the RAG tool."
        
        if recipe_number > len(agent_state.last_recipes):
            return f"Invalid recipe number. Choose between 1 and {len(agent_state.last_recipes)}."
        
        # Get recipe data
        recipe = agent_state.last_recipes[recipe_number - 1]
        
        # Initialize DB handler
        db_handler = UserDBHandler()
        
        # Generate unique recipe_id (hash-based)
        import hashlib
        recipe_id_str = f"{recipe['recipe_name']}_{recipe['ingredients']}"
        recipe_id = int(hashlib.md5(recipe_id_str.encode()).hexdigest()[:8], 16)
        
        # Save to RecipeHistory
        recipe_history = RecipeHistory(
            user_id=agent_state.user_id or 1,  # Fallback for testing TODO: delete 'or 1' when testing is done
            recipe_id=recipe_id,
            recipe_name=recipe["recipe_name"],
            servings=recipe["servings"],
            ingredients=recipe["ingredients"],
            cook_instructions=recipe["cook_instructions"],
            prep_time=recipe["prep_time"],
        )
        history_id = db_handler.insert_recipe_history(recipe_history)
        
        # Save to NutritionHistory
        nutrition_history = NutritionHistory(
            user_id=agent_state.user_id or 1, # TODO: delete 'or 1' after testing
            recipe_id=recipe_id,
            calories=recipe.get("calories", 0),
            protein=recipe.get("protein", 0),
            fat=recipe.get("fat", 0),
            carbohydrates=recipe.get("carbs", 0),
            fiber=recipe.get("fiber", 0),
            sugar=recipe.get("sugar", 0),
            sodium=recipe.get("sodium", 0),
        )
        db_handler.insert_nutrition_history(nutrition_history)
        
        # Clear state after successful save
        agent_state.clear_recipes()
        
        response = f"✅ **Recipe saved successfully!**\n\n📋 **Details:**\n- Recipe: {recipe['recipe_name']}\n- ID: {history_id}\n- Servings: {recipe['servings']}\n- Prep Time: {recipe['prep_time']}"
        
        if rating:
            response += f"\n- Your Rating: {rating}/5 ⭐"
        
        response += "\n\n🍳 **Enjoy cooking!** Need more recipes? Just ask!"
        
        return response
        
    except Exception as e:
        return f"Error saving recipe: {str(e)}"


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

    # llm = ChatGroq(
    #     model="llama-3.3-70b-versatile",
    #     temperature=0,
    #     max_tokens=512,
    # )

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
2. AFTER tool returns Observation → IMMEDIATELY use "Final Answer" with the COMPLETE tool output (DO NOT modify or summarize!)
3. After showing recipes, ask if user wants to cook one or needs more suggestions
4. When user selects a recipe by NUMBER (e.g., "I'll cook recipe 2", "recipe 3", "the second one") → use 'save_recipe' tool, then "Final Answer" with confirmation
5. For general questions not about recipes → answer directly with "Final Answer"

RESPONSE FORMAT (follow exactly):

Step 1 - Call Tool:
```json
{{
  "action": "search_recipes",
  "action_input": {{"query": "chicken dinner"}}
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
User: "I need dinner ideas with chicken"
Thought: User wants recipes, I must use search_recipes tool
Action:
```json
{{
  "action": "search_recipes",
  "action_input": {{"query": "dinner ideas with chicken"}}
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

Example 3 - General Question:
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
- NEVER repeat the same tool call twice - if you already called it, move to Final Answer
- NEVER summarize or modify the Observation from search_recipes - pass it through COMPLETELY
- Use "Final Answer" after EVERY tool call (search_recipes AND save_recipe)
- Parse recipe numbers from natural language (e.g., "second one" = recipe 2, "the first recipe" = recipe 1)
- Always show the COMPLETE recipe details including ingredients, instructions, and nutrition facts
- Be conversational and friendly in your responses"""

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