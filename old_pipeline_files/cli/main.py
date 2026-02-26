"""
CLI for the Nutrition AI Assistant.

Provides three commands:
    initialize  — Load and build RAG vectorstores.
    ask         — Send a single query and display recipe recommendations.
    chat        — Interactive session with continuous queries.

Usage:
    python src/cli/main.py initialize
    python src/cli/main.py ask "I have diabetes. Suggest a healthy dinner."
    python src/cli/main.py chat --name John --surname Doe
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

# -- Path setup so imports work from src/ ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from settings import (
    PDF_DIR, DATA_DIR,
    MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH,
    LLM_MODEL,
)
from components.intent_retriever import IntentParser
from components.safety_filter import SafetyFilter, SafetyCheckResult
from rags.recipes_nutrition_rag import RecipesNutritionRAG
from rags.medical_rag import MedicalRAG
from pipeline.pipeline import RAGPipeline, PipelineResult

__version__ = "0.1.0"

app = typer.Typer(help="Nutrition AI Assistant CLI", add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_pipeline(rebuild: bool = False) -> RAGPipeline:
    """Create and initialize the full RAG pipeline with a Rich spinner."""
    with console.status("[bold cyan]Loading Intent Parser...", spinner="dots"):
        intent_parser = IntentParser(model_name=LLM_MODEL)
    console.print("  [green]Intent Parser loaded[/green]")

    with console.status("[bold cyan]Loading Medical RAG...", spinner="dots"):
        medical_rag = MedicalRAG(
            folder_paths=[str(PDF_DIR)],
            model_name=LLM_MODEL,
            vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
            embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        )
        medical_rag.initialize(force_rebuild=rebuild)
    console.print("  [green]Medical RAG loaded[/green]")

    with console.status("[bold cyan]Loading Nutrition RAG...", spinner="dots"):
        nutrition_rag = RecipesNutritionRAG(
            data_folder=str(DATA_DIR),
            model_name=LLM_MODEL,
            vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH),
        )
        nutrition_rag.initialize(force_rebuild=rebuild)
    console.print("  [green]Nutrition RAG loaded[/green]")

    safety_filter = SafetyFilter(model_name=LLM_MODEL, debug=False)
    console.print("  [green]Safety Filter loaded[/green]")

    pipeline = RAGPipeline(
        intent_parser=intent_parser,
        medical_rag=medical_rag,
        nutrition_rag=nutrition_rag,
        safety_filter=safety_filter,
    )
    return pipeline


def _prompt_user_data() -> dict:
    """Interactively prompt for user data using Rich prompts."""
    console.print(Panel("[bold]User Registration[/bold]", border_style="blue"))
    name = Prompt.ask("[bold]First name[/bold]", default="Guest")
    surname = Prompt.ask("[bold]Last name[/bold]", default="User")
    age = int(Prompt.ask("[bold]Age[/bold]", default="30"))
    gender = Prompt.ask("[bold]Gender[/bold]", default="Other")
    caretaker = Prompt.ask("[bold]Caretaker name[/bold] (leave empty if none)", default="")
    health_condition = Prompt.ask("[bold]Health conditions[/bold] (comma-separated, leave empty if none)", default="")
    console.print()
    return {
        "name": name,
        "surname": surname,
        "age": age,
        "gender": gender,
        "caretaker": caretaker,
        "health_condition": health_condition,
    }


def _render_result(result: PipelineResult) -> None:
    """Display a PipelineResult using Rich panels, tables, and markdown."""

    # -- Intent panel --
    intent = result.intent
    intent_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    intent_table.add_column("Field", style="bold")
    intent_table.add_column("Value")

    fields = [
        ("Name", f"{intent.name} {intent.surname}".strip()),
        ("Health conditions", intent.health_condition),
        ("Restrictions", intent.restrictions),
        ("Preferences", intent.preferences),
        ("Instructions", intent.instructions),
        ("Caretaker", intent.caretaker),
    ]
    for label, value in fields:
        if value:
            intent_table.add_row(label, value)

    console.print(Panel(intent_table, title="Parsed Intent", border_style="blue"))

    # -- Constraints panel --
    constraints = result.constraints
    constraint_lines = []

    avoid = constraints.get("avoid", [])
    if avoid:
        constraint_lines.append(f"[bold red]Avoid:[/bold red] {', '.join(avoid)}")

    limit = constraints.get("limit", [])
    if limit:
        constraint_lines.append(f"[bold yellow]Limit:[/bold yellow] {', '.join(limit)}")

    rules = constraints.get("constraints", {})
    for nutrient, rule in rules.items():
        parts = []
        if rule.get("max") is not None:
            parts.append(f"max {rule['max']}")
        if rule.get("min") is not None:
            parts.append(f"min {rule['min']}")
        if parts:
            constraint_lines.append(f"  {nutrient}: {', '.join(parts)}")

    notes = constraints.get("notes", "")
    if notes:
        constraint_lines.append(f"\n[dim]{notes}[/dim]")

    if constraint_lines:
        console.print(Panel(
            "\n".join(constraint_lines),
            title="Medical Constraints",
            border_style="yellow",
        ))

    # -- Augmented query panel --
    if result.augmented_query:
        console.print(Panel(
            result.augmented_query,
            title="Augmented Query",
            border_style="magenta",
        ))

    # -- Safety check panel --
    safety = result.safety_result
    if safety and safety.total_count > 0:
        safety_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        safety_table.add_column("Status", width=3)
        safety_table.add_column("Recipe")
        safety_table.add_column("Verdict")
        safety_table.add_column("Issues")

        for rv in safety.recipe_verdicts:
            if rv.verdict.value == "safe":
                icon, style = "[green]OK[/green]", "green"
            elif rv.verdict.value == "warning":
                icon, style = "[yellow]!![/yellow]", "yellow"
            else:
                icon, style = "[red]XX[/red]", "red"

            issues_text = "; ".join(iss.description for iss in rv.issues[:2]) if rv.issues else ""
            safety_table.add_row(
                icon, rv.recipe_name, f"[{style}]{rv.verdict.value.upper()}[/{style}]", issues_text,
            )

        header = f"{safety.safe_count}/{safety.total_count} recipes passed safety check"
        console.print(Panel(safety_table, title=f"Safety Check ({header})", border_style="cyan"))

    # -- Recipes panel --
    if safety and safety.total_count == 0:
        console.print(Panel(
            "[bold yellow]The model could not generate valid recipes for your query.[/bold yellow]\n"
            "This can happen when constraints conflict with requested ingredients.\n"
            "Try rephrasing your query or relaxing some constraints.\n\n"
            f"[dim]Raw model response:[/dim]\n{result.llm_recommendation[:500]}",
            title="No Recipes Found",
            border_style="yellow",
        ))
    elif safety and safety.safe_recipes_markdown:
        console.print(Panel(
            Markdown(safety.safe_recipes_markdown),
            title="Recommended Recipes",
            border_style="green",
        ))
    elif result.llm_recommendation:
        console.print(Panel(
            Markdown(result.llm_recommendation),
            title="Recommended Recipes",
            border_style="green",
        ))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def version_callback(value: bool):
    if value:
        console.print(f"nutrition-ai v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Nutrition AI Assistant CLI"""


@app.command()
def initialize(
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Force rebuild vectorstores from scratch."),
):
    """Initialize the RAG pipeline and build/load vectorstores."""
    console.print("\n[bold]Initializing Nutrition AI pipeline...[/bold]\n")
    try:
        _init_pipeline(rebuild=rebuild)
        console.print(Panel(
            "[bold green]Pipeline initialized successfully![/bold green]\n"
            "Vectorstores are loaded and ready.\n"
            "Run [bold]ask[/bold] or [bold]chat[/bold] to start querying.",
            title="Ready",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Initialization failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your nutrition/recipe question."),
):
    """Ask a single nutrition/recipe question."""
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    user_data = _prompt_user_data()
    pipeline = _init_pipeline()

    with console.status("[bold cyan]Processing your query...", spinner="dots"):
        result = pipeline.process(query, user_data=user_data)

    console.print()
    _render_result(result)


@app.command()
def chat():
    """Start an interactive chat session."""
    console.print(Panel(
        "[bold]Nutrition AI Chat[/bold]\n"
        "Type your questions below. Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
        border_style="cyan",
    ))

    user_data = _prompt_user_data()
    pipeline = _init_pipeline()

    console.print()
    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if query.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not query.strip():
            continue

        with console.status("[bold cyan]Thinking...", spinner="dots"):
            result = pipeline.process(query, user_data=user_data)

        console.print()
        _render_result(result)
        console.print()


if __name__ == "__main__":
    app()
