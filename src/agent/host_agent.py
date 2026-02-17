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

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# Import your existing components
from pipeline.pipeline import RAGPipeline, PipelineResult
from database.db import UserDBHandler
from database.models import RecipeHistory, NutritionHistory
from settings import LLM_MODEL


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
        
        # Add helper text for user
        footer = f"\n\n---\n**Found {len(parsed_recipes)} recipes.** Reply with the recipe number you'd like to cook (e.g., 'I'll cook recipe 2'), or ask for more suggestions."
        
        return output_text + footer
        
    except Exception as e:
        return f"Error fetching recipes: {str(e)}. Please try rephrasing your request."
    

def database_handling_tool_func(input_str: str) -> str:
    """Tool: Save selected recipe to database.
    
    This tool should ONLY be called when:
    - User explicitly selects a recipe by number (e.g., "I'll cook recipe 2")
    - User wants to rate a recipe they cooked
    
    Args:
        input_str: Expected format "recipe_number:2,rating:5" or just "recipe_number:2"
    
    Returns:
        Confirmation message or error
    """
    try:
        # Parse input (format: "recipe_number:2,rating:4")
        params = {}
        for item in input_str.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                params[key.strip()] = value.strip()
        
        recipe_number = int(params.get("recipe_number", 0))
        rating = int(params.get("rating", 0)) if "rating" in params else None
        
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
            user_id=agent_state.user_id or 1,  # Fallback for testing
            recipe_id=recipe_id,
            recipe_name=recipe["recipe_name"],
            servings=recipe["servings"],
            ingredients=recipe["ingredients"],
            instructions=recipe["instructions"],
            prep_time=recipe["prep_time"],
        )
        history_id = db_handler.insert_recipe_history(recipe_history)
        
        # Save to NutritionHistory
        nutrition_history = NutritionHistory(
            user_id=agent_state.user_id or 1,
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
        
        response = f"✅ Recipe '{recipe['recipe_name']}' saved successfully (ID: {history_id})!"
        if rating:
            response += f" Rating: {rating}/5 ⭐"
        
        return response
        
    except Exception as e:
        return f"Error saving recipe: {str(e)}"


AGENT_PROMPT_TEMPLATE = """You are a helpful nutrition assistant chatbot.

Your job is to help users find recipes and save them to their cooking history.

IMPORTANT WORKFLOW:
1. When user asks for recipe recommendations → Use 'rag_pipeline' tool
2. After showing recipes → Ask user if they want to cook one or need more suggestions
3. When user selects a recipe (e.g., "I'll cook recipe 2") → Use 'database_handling' tool with format: "recipe_number:2"
4. For general questions → Answer directly without tools

TOOLS AVAILABLE:
{tools}

TOOL NAMES: {tool_names}

CONVERSATION HISTORY:
{chat_history}

USER INPUT: {input}

REASONING PROCESS (think step by step):
{agent_scratchpad}

Remember:
- Always parse recipe numbers from user messages (e.g., "recipe 2", "#2", "the second one")
- Confirm before saving to database
- Be friendly and conversational
"""

agent_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    template=AGENT_PROMPT_TEMPLATE,
)


def create_nutrition_agent(user_id: Optional[int] = None, user_data: Optional[Dict] = None) -> AgentExecutor:
    """Initialize the nutrition assistant agent.
    
    Args:
        user_id: Optional user ID for database operations
        user_data: Optional user context (preferences, health_condition, etc.)
    
    Returns:
        Configured AgentExecutor ready for chat
    """
    # Set user context in state
    if user_id:
        agent_state.user_id = user_id
    if user_data:
        agent_state.user_data = user_data
    
    # Initialize Ollama LLM
    llm = ChatOllama(
        model="llama3:8b",
        temperature=0.1,  # Low temp for consistent tool selection
    )
    
    # Define tools
    tools = [
        Tool(
            name="rag_pipeline",
            func=rag_pipeline_tool_func,
            description=(
                "Use this tool to search for recipe recommendations. "
                "Input should be the user's recipe request or query. "
                "ONLY use this when user explicitly asks for recipes or more suggestions."
            ),
        ),
        Tool(
            name="database_handling",
            func=database_handling_tool_func,
            description=(
                "Use this tool to save a selected recipe to the user's cooking history. "
                "Input format: 'recipe_number:X,rating:Y' (rating is optional). "
                "ONLY use this when user selects a specific recipe number."
            ),
        ),
    ]
    
    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",  # Important for AgentExecutor
    )
    
    # Create ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt,
    )
    
    # Wrap in executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # For debugging
        handle_parsing_errors=True,
        max_iterations=1,  # Prevent infinite loops
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