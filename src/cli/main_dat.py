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

from pipeline.PipelineRAGDummyDat import PipelineRagDummy

app = typer.Typer()
console = Console()


@app.command()
def initialize():
    """Initialize the RAG pipeline"""
    PipelineRagDummy.initialize()


@app.command()
def chat():
    """Start an interactive chat session"""
    console.print("Starting chat session. Type 'exit' or 'quit' to end.", style="blue")
    
    while True:
        question = console.input("[bold green]You:[/bold green] ")
        
        if question.lower() in ['exit', 'quit']:
            console.print("Goodbye!", style="yellow")
            break
        
        answer = PipelineRagDummy.ask(question)
        console.print(f"[bold blue]Assistant:[/bold blue] {answer}")


if __name__ == "__main__":
    app()