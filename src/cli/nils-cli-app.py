import typer
import time

# ============================================================================
# PART 1: Dummy RAG Pipeline Class
# ============================================================================

class PipelineRagDummy:
    """Dummy RAG pipeline for testing CLI functionality."""
    
    def initialize(self) -> str:
        """Initialize RAGs with a 3-second delay."""
        typer.echo("Initializing RAGs...")
        time.sleep(3)
        return "All RAGS are operational."
    
    def ask(self, question: str) -> str:
        """Answer user question with dummy response."""
        return "Just have a Banh Mi!"


# ============================================================================
# PART 2: Typer CLI App Setup
# ============================================================================

cli_app = typer.Typer()
pipeline = PipelineRagDummy()


@cli_app.command()
def initialize() -> None:
    """Initialize the RAG system."""
    result = pipeline.initialize()
    typer.echo(result)


@cli_app.command()
def chat() -> None:
    """Start chat mode with the RAG pipeline."""
    typer.echo("Chat mode started. Type 'exit' or 'quit' to leave.")
    typer.echo("What do you want to eat?")
    
    while True:
        # Get user input
        user_input = typer.prompt("You")
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            typer.echo("Exiting...")
            break
        
        # Call ask method and display response
        answer = pipeline.ask(user_input)
        typer.echo(f"Assistant: {answer}")


# ============================================================================
# PART 3: Main Entry Point
# ============================================================================

if __name__ == "__main__":
    cli_app()