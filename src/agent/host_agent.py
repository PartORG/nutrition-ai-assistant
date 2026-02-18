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

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import your existing components
from pipeline.pipeline import RAGPipeline, PipelineResult
from database.db import UserDBHandler
from database.models import RecipeHistory, NutritionHistory
from settings import LLM_MODEL

load_dotenv()

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
    """Extract structured recipe data from RAG pipeline markdown output.
    
    Expected format:
    ### Recipe 1: Spaghetti Carbonara
    - **Servings:** 4
    - **Prep Time:** 30 minutes
    - **Ingredients:** pasta, eggs, bacon, ...
    - **Instructions:** 1. Boil pasta... 2. Cook bacon...
    - **Nutrition:** Calories: 450, Protein: 25g, Carbs: 50g, Fat: 18g
    
    Args:
        markdown_text: Raw markdown output from pipeline
        
    Returns:
        List of recipe dictionaries with parsed fields
    """
    recipes = []
    
    # Regex pattern to match recipe blocks
    pattern = r"### Recipe (\d+): (.+?)\n- \*\*Servings:\*\* (\d+)\n- \*\*Prep Time:\*\* (.+?)\n- \*\*Ingredients:\*\* (.+?)\n- \*\*Instructions:\*\* (.+?)\n- \*\*Nutrition:\*\* (.+?)(?=\n\n###|\Z)"
    
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    # FALLBACK if Regex fails:
    if not matches:
        print(f"⚠️ WARNING: Could not parse recipes from markdown. Raw output:\n{markdown_text[:500]}...")
        # Optional: Simple fallback parsing
        return []
    
    for match in matches:
        recipe_num, name, servings, prep_time, ingredients, cook_instructions, nutrition = match
        
        # Parse nutrition string: "Calories: 450, Protein: 25g, ..."
        nutrition_dict = {}
        for item in nutrition.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip().lower()
                value = value.strip().rstrip("g").strip()
                try:
                    nutrition_dict[key] = float(value)
                except ValueError:
                    nutrition_dict[key] = value
        
        recipe = {
            "recipe_number": int(recipe_num),
            "recipe_name": name.strip(),
            "servings": int(servings),
            "prep_time": prep_time.strip(),
            "ingredients": ingredients.strip(),
            "cook_instructions": cook_instructions.strip(),
            "calories": nutrition_dict.get("calories", 0),
            "protein": nutrition_dict.get("protein", 0),
            "carbs": nutrition_dict.get("carbs", 0),
            "fat": nutrition_dict.get("fat", 0),
            "fiber": nutrition_dict.get("fiber", 0),
            "sugar": nutrition_dict.get("sugar", 0),
            "sodium": nutrition_dict.get("sodium", 0),
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
    
    # llm = ChatOllama(
    #     model="llama3:8b",
    #     temperature=0,
    #     ollama_base_url="http://localhost:11434",
    # )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
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

IMPORTANT RULES:
1. When user asks for recipes or meal suggestions → ALWAYS use 'search_recipes' tool
2. Return the COMPLETE tool output to the user WITHOUT modification
3. After showing recipes, ask if user wants to cook one or needs more suggestions
4. When user selects a recipe by NUMBER (e.g., "I'll cook recipe 2", "recipe 3", "the second one") → use 'save_recipe' tool
5. For general questions not about recipes → answer directly with "Final Answer"

WORKFLOW EXAMPLES:

Example 1 - Recipe Search:
User: "I need dinner ideas with chicken"
Thought: User wants recipes, I should use search_recipes tool
Action:
WORKFLOW EXAMPLES:

Example 1 - Recipe Search:
User: "I need dinner ideas with chicken"
Action: Use 'search_recipes' with query "dinner ideas with chicken"
Response: [Show FULL tool output] + "Would you like to cook one of these, or see more options?"

Example 2 - Recipe Selection:
User: "I'll cook recipe 2"
Action: Use 'save_recipe' with recipe_number=2
Response: "Recipe saved! Enjoy cooking!"

Example 3 - General Question:
User: "What's the difference between protein and carbs?"
Action: No tool needed
Response: [Direct answer]

CRITICAL: 
- Always show the COMPLETE recipe details from search_recipes
- Parse recipe numbers from natural language (e.g., "second one" = recipe 2)"""

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
        max_iterations=2,
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