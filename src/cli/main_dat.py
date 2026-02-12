# Typer + Rich app
# update requirements-dev.txt with typer, rich 
# --help , initialize (PipelineRagDummy.initialize()),
# chat (PipelineRagDummy.ask()),
# dummy class PipelineRAGDummy: 
#    - method initialize(return string "initializing RAGs..." timer.sleep(3) "All RAGS are operational.")
#    - method ask(question) --> question: user_question, Answer: 'Dummy answer'
# 
# CLI chat: while loop that checks if user typed 'exit'/ 'quit'
# user input - stores into question variable and calls PipelineRagDummy.ask(question);
# 
# src/cli/main.py :
# 
# if __name__ == "__main__":
#     cli_app()
# 
# (venv) src/cli/ python 


import typer
import sys
from pathlib import Path
from rich.console import Console

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from pipeline.test_pipeline import RAGPipeline
from pipeline.medical_rag import MedicalRAG
from pipeline.recipes_nutrition_rag import RecipesNutritionRAG
from pipeline.intent_retriever import IntentParser
from pipeline.safety_filter import SafetyFilter
from pipeline.config import (
    PDF_DIR, DATA_DIR,
    MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH,
    LLM_MODEL,
)


app = typer.Typer()
console = Console()


@app.command()
def initialize():
    """Initialize the RAG pipeline"""
    intent_parser = IntentParser(model_name=LLM_MODEL)
    medical_rag = MedicalRAG(folder_paths=[str(PDF_DIR)], model_name=LLM_MODEL, vectorstore_path=str(MEDICAL_VECTORSTORE_PATH), embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    medical_rag.initialize(force_rebuild=False)  # TODO: set back to False after first run

    nutrition_rag = RecipesNutritionRAG(data_folder=str(DATA_DIR), model_name=LLM_MODEL, vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH))
    nutrition_rag.initialize()


@app.command()
def chat():
    """Start an interactive chat session"""
    console.print("Starting chat session. Type 'exit' or 'quit' to end.", style="blue")
    
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

    while True:
        question = console.input("[bold green]You:[/bold green] ")
        
        if question.lower() in ['exit', 'quit']:
            console.print("Goodbye!", style="yellow")
            break
        
        answer = pipeline.process(question)
        answer.display()


if __name__ == "__main__":
    app()