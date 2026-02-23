"""
CLI for the Nutrition AI Assistant.

Commands:
    register    — Create a new account and log in.
    login       — Authenticate and save your session locally.
    logout      — Clear the saved session.
    whoami      — Show the currently logged-in user.
    history     — List your recent conversations.
    initialize  — Pre-build RAG vectorstores (auto-runs on first use).
    ask         — Single-shot question (recommendation pipeline, no history saved).
    chat        — Interactive chat session with full history persistence.

Usage:
    python cli/main.py register
    python cli/main.py login
    python cli/main.py chat
    python cli/main.py chat --conversation <id>   # resume existing conversation
    python cli/main.py ask "I have diabetes. Suggest a healthy dinner."
    python cli/main.py history
    python cli/main.py logout
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

# -- Path setup so imports from src/ work --
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from infrastructure.config import Settings
from factory import ServiceFactory
from application.context import SessionContext
from application.dto import RecommendationResult, RegisterRequest, LoginRequest

__version__ = "0.3.0"

app = typer.Typer(help="Nutrition AI Assistant CLI", add_completion=False)
console = Console()

# Stores active CLI session: user_id, login, optional full name.
_SESSION_FILE = project_root / ".cli-session.json"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _load_session() -> dict | None:
    if _SESSION_FILE.exists():
        try:
            return json.loads(_SESSION_FILE.read_text())
        except Exception:
            return None
    return None


def _save_session(user_id: int, login: str, name: str = "") -> None:
    _SESSION_FILE.write_text(
        json.dumps({"user_id": user_id, "login": login, "name": name}, indent=2)
    )


def _clear_session() -> None:
    if _SESSION_FILE.exists():
        _SESSION_FILE.unlink()


def _require_session() -> dict:
    session = _load_session()
    if not session:
        console.print(
            "[red]Not logged in.[/red]  "
            "Run [bold]python cli/main.py login[/bold] first."
        )
        raise typer.Exit(code=1)
    return session


# ---------------------------------------------------------------------------
# Factory helper — everything runs inside a single asyncio.run()
# ---------------------------------------------------------------------------

async def _make_factory() -> ServiceFactory:
    config = Settings.from_env(project_root=project_root)
    factory = ServiceFactory(config)
    with console.status("[bold cyan]Initializing pipeline...", spinner="dots"):
        await factory.initialize()
    console.print("  [green]Pipeline ready[/green]\n")
    return factory


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

def _render_recommendation(result: RecommendationResult) -> None:
    safety = result.safety_check
    if safety and safety.total_count > 0:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("", width=3)
        t.add_column("Recipe")
        t.add_column("Verdict")
        t.add_column("Issues")
        for rv in safety.recipe_verdicts:
            if rv.verdict.value == "safe":
                icon, style = "[green]OK[/green]", "green"
            elif rv.verdict.value == "warning":
                icon, style = "[yellow]!![/yellow]", "yellow"
            else:
                icon, style = "[red]XX[/red]", "red"
            issues = "; ".join(i.description for i in rv.issues[:2]) if rv.issues else ""
            t.add_row(icon, rv.recipe_name, f"[{style}]{rv.verdict.value.upper()}[/{style}]", issues)
        header = f"{safety.safe_count}/{safety.total_count} passed"
        console.print(Panel(t, title=f"Safety Check ({header})", border_style="cyan"))

    text = result.summary or result.raw_response
    if text:
        console.print(Panel(Markdown(text), title="Recommended Recipes", border_style="green"))
    else:
        console.print(Panel(
            "[bold yellow]No recipes found. Try rephrasing your query.[/bold yellow]",
            title="No Results", border_style="yellow",
        ))


# ---------------------------------------------------------------------------
# Commands — version
# ---------------------------------------------------------------------------

def _version_cb(value: bool):
    if value:
        console.print(f"nutrition-ai v{__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-v",
        help="Show version and exit.",
        callback=_version_cb,
        is_eager=True,
    ),
):
    """Nutrition AI Assistant CLI"""


# ---------------------------------------------------------------------------
# Commands — auth
# ---------------------------------------------------------------------------

@app.command()
def register():
    """Create a new account and log in."""
    console.print(Panel("[bold]Create Account[/bold]", border_style="blue"))
    name = Prompt.ask("First name")
    surname = Prompt.ask("Last name")
    username = Prompt.ask("Username")
    age_str = Prompt.ask("Age", default="0")
    gender = Prompt.ask("Gender", default="other")
    password = typer.prompt("Password", hide_input=True)
    password2 = typer.prompt("Confirm password", hide_input=True)

    if password != password2:
        console.print("[red]Passwords do not match.[/red]")
        raise typer.Exit(code=1)

    async def _do():
        factory = await _make_factory()
        svc = factory.create_authentication_service()
        return await svc.register(RegisterRequest(
            login=username,
            password=password,
            name=name,
            surname=surname,
            age=int(age_str) if age_str.isdigit() else 0,
            gender=gender,
        ))

    try:
        token = asyncio.run(_do())
        _save_session(user_id=token.user_id, login=username, name=f"{name} {surname}")
        console.print(Panel(
            f"[bold green]Account created![/bold green]\n"
            f"Logged in as [bold]{username}[/bold] (user_id={token.user_id})",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Registration failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def login():
    """Authenticate and save your session."""
    username = Prompt.ask("Username")
    password = typer.prompt("Password", hide_input=True)

    async def _do():
        factory = await _make_factory()
        svc = factory.create_authentication_service()
        return await svc.login(LoginRequest(login=username, password=password))

    try:
        token = asyncio.run(_do())
        _save_session(user_id=token.user_id, login=username)
        console.print(Panel(
            f"[bold green]Logged in![/bold green]\n"
            f"Welcome back, [bold]{username}[/bold] (user_id={token.user_id})",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Login failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def logout():
    """Clear the saved session."""
    _clear_session()
    console.print("[green]Logged out.[/green]")


@app.command()
def whoami():
    """Show the currently logged-in user."""
    session = _load_session()
    if not session:
        console.print("[yellow]Not logged in.[/yellow]")
        return
    name = session.get("name", "")
    label = f"{name} ({session['login']})" if name else session["login"]
    console.print(f"Logged in as [bold]{label}[/bold]  (user_id={session['user_id']})")


# ---------------------------------------------------------------------------
# Commands — pipeline / history
# ---------------------------------------------------------------------------

@app.command()
def initialize(
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Force rebuild vectorstores."),
):
    """Pre-build RAG vectorstores (runs automatically on first ask/chat)."""
    console.print("\n[bold]Initializing Nutrition AI pipeline...[/bold]\n")
    try:
        asyncio.run(_make_factory())
        console.print(Panel(
            "[bold green]Pipeline initialized successfully![/bold green]\n"
            "Run [bold]ask[/bold] or [bold]chat[/bold] to start.",
            title="Ready", border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Initialization failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def history():
    """List your recent conversations (requires login)."""
    session = _require_session()

    async def _do():
        factory = await _make_factory()
        svc = factory.create_chat_history_service()
        return await svc.list_conversations(session["user_id"])

    try:
        convos = asyncio.run(_do())
    except Exception as e:
        console.print(f"[bold red]Failed to load history:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not convos:
        console.print("[yellow]No conversations yet.  Start one with [bold]chat[/bold].[/yellow]")
        return

    t = Table(title="Recent Conversations", box=box.ROUNDED)
    t.add_column("#", width=3, style="dim")
    t.add_column("Title")
    t.add_column("Conversation ID", style="dim")
    t.add_column("Last Activity")
    for i, c in enumerate(convos[:15], 1):
        title = c.title or "[dim](untitled)[/dim]"
        last = c.last_message_at.strftime("%Y-%m-%d %H:%M") if c.last_message_at else "—"
        t.add_row(str(i), title, c.conversation_id, last)
    console.print(t)


# ---------------------------------------------------------------------------
# Commands — ask / chat
# ---------------------------------------------------------------------------

@app.command()
def ask(
    query: str = typer.Argument(..., help="Your nutrition/recipe question."),
):
    """Single-shot question — recommendation pipeline, no history saved."""
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    session = _load_session()

    async def _do():
        factory = await _make_factory()
        user_id = session["user_id"] if session else 0
        if session:
            user_data = await factory.create_profile_service().load_user_context(user_id)
        else:
            console.print("[dim]Running as guest — log in for personalised results.[/dim]\n")
            user_data = _prompt_guest_data()

        ctx = SessionContext(
            user_id=user_id,
            conversation_id=f"cli-ask-{uuid.uuid4().hex[:8]}",
            user_data=user_data,
        )
        with console.status("[bold cyan]Processing...", spinner="dots"):
            return await factory.create_recommendation_service().get_recommendations(ctx, query)

    try:
        result = asyncio.run(_do())
        console.print()
        _render_recommendation(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def chat(
    conversation_id: str = typer.Option(
        None, "--conversation", "-c",
        help="Resume an existing conversation ID (use 'history' to find IDs).",
    ),
):
    """Interactive chat session with full history persistence."""
    session = _load_session()

    async def _run():
        factory = await _make_factory()

        if session:
            user_id = session["user_id"]
            display_name = session.get("name") or session["login"]
            user_data = await factory.create_profile_service().load_user_context(user_id)
            conv_id = conversation_id or str(uuid.uuid4())
        else:
            console.print(
                "[dim]Running as guest — history will NOT be saved.\n"
                "Log in with [bold]login[/bold] for persistence and personalisation.[/dim]\n"
            )
            display_name = "Guest"
            user_data = _prompt_guest_data()
            conv_id = f"guest-{uuid.uuid4().hex[:8]}"
            user_id = 0

        ctx = SessionContext(user_id=user_id, conversation_id=conv_id, user_data=user_data)
        agent = factory.create_agent(ctx)

        subtitle_parts = [f"conversation: {conv_id}"]
        if session:
            subtitle_parts.insert(0, f"user: {display_name}")
        console.print(Panel(
            "[bold]Nutrition AI Chat[/bold]\n"
            "Ask anything about nutrition, recipes, or your diet.\n"
            "Type [bold]exit[/bold] to stop.",
            subtitle=f"[dim]{'  |  '.join(subtitle_parts)}[/dim]",
            border_style="cyan",
        ))

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
                response = await agent.run(ctx, query)

            console.print()
            console.print(Panel(Markdown(response), title="Assistant", border_style="green"))
            console.print()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Guest data helper
# ---------------------------------------------------------------------------

def _prompt_guest_data() -> dict:
    console.print(Panel("[bold]Quick Profile (guest)[/bold]", border_style="blue"))
    age_str = Prompt.ask("Age", default="30")
    gender = Prompt.ask("Gender", default="other")
    health = Prompt.ask("Health conditions (comma-separated, or enter to skip)", default="")
    console.print()
    return {
        "age": int(age_str) if age_str.isdigit() else 30,
        "gender": gender,
        "health_conditions": [h.strip() for h in health.split(",") if h.strip()],
    }


if __name__ == "__main__":
    app()
