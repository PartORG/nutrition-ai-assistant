"""
adapters.cli.main - CLI adapter for the Nutrition AI Assistant.

Mirrors src/adapters/rest/ but for terminal use.  Uses the same
ServiceFactory, AuthenticationService, and AgentExecutor as the REST API
so all behaviour (auth, recommendations, agent) is identical.

Commands
--------
  register   Create a new account
  login      Sign in and save credentials locally (~/.nutrition-ai/session.json)
  logout     Clear stored credentials
  whoami     Show the currently logged-in user
  profile    Display your health profile and medical advice
  ask        One-shot recipe/nutrition query  (requires login, loads full pipeline)
  chat       Interactive chat session          (requires login, loads full pipeline)
  init       Build/rebuild RAG vectorstores   (admin / first-time setup)

Usage
-----
  python src/adapters/cli/main.py login
  python src/adapters/cli/main.py ask "healthy dinner with chicken"
  python src/adapters/cli/main.py chat
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ── Ensure src/ is on the path (same pattern as the old src/cli/main.py) ──
_SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SRC))

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from adapters.cli.session import Session, clear_session, load_session, save_session
from application.context import SessionContext
from application.dto import LoginRequest, RegisterRequest
from domain.exceptions import AuthenticationError, DuplicateLoginError
from factory import ServiceFactory
from infrastructure.config import Settings
from infrastructure.persistence.migrations import run_migrations

__version__ = "1.0.0"

console = Console()
app = typer.Typer(
    help="Nutrition AI Assistant CLI",
    add_completion=False,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _require_session() -> Session:
    """Return the stored session or exit with a user-friendly error."""
    session = load_session()
    if session is None:
        console.print(
            "[bold red]Not logged in.[/bold red] "
            "Run [bold]login[/bold] (or [bold]register[/bold]) first."
        )
        raise typer.Exit(code=1)
    return session


async def _make_factory(*, full_init: bool) -> ServiceFactory:
    """Create a ServiceFactory at the required initialisation level.

    full_init=False  — runs DB migrations only (fast).  Sufficient for auth,
                       profile, and whoami commands that don't need LLMs.
    full_init=True   — runs the full pipeline initialisation including Medical
                       RAG and Recipe RAG.  Required for ask and chat.
    """
    config = Settings.from_env()
    factory = ServiceFactory(config)
    if full_init:
        with console.status(
            "[bold cyan]Loading AI pipeline (first run may take a minute)…",
            spinner="dots",
        ):
            await factory.initialize()
        console.print("  [green]Pipeline ready.[/green]")
    else:
        # Light-weight: just make sure the DB schema is up-to-date.
        await run_migrations(factory._connection)
    return factory


async def _build_ctx(session: Session, factory: ServiceFactory) -> SessionContext:
    """Build a SessionContext pre-populated with the user's profile data."""
    profile_svc = factory.create_profile_service()
    user_data = await profile_svc.load_user_context(session.user_id)
    return SessionContext(
        user_id=session.user_id,
        conversation_id="cli",
        user_data=user_data,
    )


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nutrition-ai v{__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands: Auth
# ---------------------------------------------------------------------------

@app.command()
def register() -> None:
    """Create a new account."""
    console.print(Panel("[bold]Create Account[/bold]", border_style="blue"))

    login_name = Prompt.ask("[bold]Username[/bold]  (min 3 chars)")
    password   = Prompt.ask("[bold]Password[/bold]   (min 6 chars)", password=True)
    name       = Prompt.ask("[bold]First name[/bold]", default="")
    surname    = Prompt.ask("[bold]Last name[/bold]",  default="")
    age_str    = Prompt.ask("[bold]Age[/bold]",         default="0")
    gender     = Prompt.ask("[bold]Gender[/bold]",      default="")
    caretaker  = Prompt.ask("[bold]Caretaker[/bold] (leave empty if none)", default="")
    health     = Prompt.ask(
        "[bold]Health conditions[/bold] (comma-separated, or leave empty)", default=""
    )

    async def _run() -> None:
        factory  = await _make_factory(full_init=False)
        auth_svc = factory.create_authentication_service()
        try:
            token = await auth_svc.register(RegisterRequest(
                login=login_name,
                password=password,
                name=name,
                surname=surname,
                age=int(age_str) if age_str.isdigit() else 0,
                gender=gender,
                caretaker=caretaker,
                health_condition=health,
            ))
        except DuplicateLoginError:
            console.print(
                f"[bold red]Username '{login_name}' is already taken.[/bold red]"
            )
            raise typer.Exit(code=1)

        if health:
            profile_svc = factory.create_profile_service()
            await profile_svc.save_initial_profile(token.user_id, health)

        save_session(Session(
            user_id=token.user_id,
            access_token=token.access_token,
            login=login_name,
        ))
        console.print(Panel(
            f"[bold green]Account created and logged in![/bold green]\n"
            f"Welcome, [bold]{login_name}[/bold] (user_id={token.user_id}).\n"
            "Run [bold]chat[/bold] or [bold]ask[/bold] to get started.",
            border_style="green",
        ))

    asyncio.run(_run())


@app.command()
def login() -> None:
    """Sign in to your account."""
    login_name = Prompt.ask("[bold]Username[/bold]")
    password   = Prompt.ask("[bold]Password[/bold]", password=True)

    async def _run() -> None:
        factory  = await _make_factory(full_init=False)
        auth_svc = factory.create_authentication_service()
        try:
            token = await auth_svc.login(LoginRequest(
                login=login_name, password=password
            ))
        except AuthenticationError:
            console.print(
                "[bold red]Login failed.[/bold red] "
                "Check your username and password."
            )
            raise typer.Exit(code=1)

        save_session(Session(
            user_id=token.user_id,
            access_token=token.access_token,
            login=login_name,
        ))
        console.print(Panel(
            f"[bold green]Logged in![/bold green] "
            f"Welcome back, [bold]{login_name}[/bold].\n"
            "Run [bold]chat[/bold] or [bold]ask[/bold] to continue.",
            border_style="green",
        ))

    asyncio.run(_run())


@app.command()
def logout() -> None:
    """Sign out and clear stored credentials."""
    session = load_session()
    if session is None:
        console.print("[dim]Not currently logged in.[/dim]")
        return
    label = session.login or f"user #{session.user_id}"
    if Confirm.ask(f"Sign out [bold]{label}[/bold]?"):
        clear_session()
        console.print("[green]Logged out.[/green]")


@app.command()
def whoami() -> None:
    """Show the currently logged-in user."""
    session = load_session()
    if session is None:
        console.print("[dim]Not logged in.[/dim]")
        return
    console.print(
        f"Logged in as [bold]{session.login or '?'}[/bold] "
        f"(user_id={session.user_id})"
    )


# ---------------------------------------------------------------------------
# Commands: Profile (requires login, DB-only — no RAG needed)
# ---------------------------------------------------------------------------

@app.command()
def profile() -> None:
    """Show your health profile, medical advice, and dietary preferences."""
    session = _require_session()

    async def _run() -> None:
        factory     = await _make_factory(full_init=False)
        profile_svc = factory.create_profile_service()
        user_repo   = factory.create_user_repository()
        ctx         = await _build_ctx(session, factory)

        user_entity = await user_repo.get_by_id(session.user_id)
        profiles    = await profile_svc.get_profile_history(ctx)
        medical     = await profile_svc.get_medical_advice(ctx)

        # ── User info ──────────────────────────────────────────────────────
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("Field", style="bold")
        t.add_column("Value")
        if user_entity:
            full_name = f"{user_entity.name} {user_entity.surname}".strip()
            if full_name:
                t.add_row("Name",      full_name)
            t.add_row("Username",  user_entity.user_name or "[dim]—[/dim]")
            if user_entity.age:
                t.add_row("Age",   str(user_entity.age))
            if user_entity.gender:
                t.add_row("Gender", user_entity.gender)
            if user_entity.caretaker:
                t.add_row("Caretaker", user_entity.caretaker)
        console.print(Panel(t, title="Your Profile", border_style="blue"))

        # ── Health & dietary ───────────────────────────────────────────────
        if profiles:
            p  = profiles[0]
            t2 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            t2.add_column("Field", style="bold")
            t2.add_column("Value")
            t2.add_row("Health conditions", p.health_condition or "[dim]none[/dim]")
            t2.add_row("Preferences",       p.preferences      or "[dim]none[/dim]")
            t2.add_row("Restrictions",      p.restrictions     or "[dim]none[/dim]")
            console.print(Panel(t2, title="Health & Dietary Profile", border_style="yellow"))

        # ── Medical advice ─────────────────────────────────────────────────
        if medical:
            m     = medical[0]
            lines = []
            if m.medical_advice:
                lines.append(f"[bold]Advice:[/bold] {m.medical_advice}")
            if m.avoid:
                lines.append(f"[bold red]Avoid:[/bold red] {m.avoid}")
            if m.dietary_limit:
                lines.append(f"[bold yellow]Limit:[/bold yellow] {m.dietary_limit}")
            if m.dietary_constraints:
                lines.append(f"[bold]Constraints:[/bold] {m.dietary_constraints}")
            if lines:
                console.print(Panel(
                    "\n".join(lines),
                    title="Medical Advice",
                    border_style="red",
                ))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Commands: Recipes / Nutrition (requires login + full pipeline)
# ---------------------------------------------------------------------------

@app.command()
def ask(
    query: str = typer.Argument(..., help="Your nutrition or recipe question."),
) -> None:
    """Ask a one-shot recipe or nutrition question (requires login)."""
    session = _require_session()

    async def _run() -> None:
        factory = await _make_factory(full_init=True)
        ctx     = await _build_ctx(session, factory)
        rec_svc = factory.create_recommendation_service()

        with console.status("[bold cyan]Thinking…", spinner="dots"):
            result = await rec_svc.get_recommendations(ctx, query)

        safe_recipes = result.safe_recipes
        if safe_recipes:
            console.print(Panel(
                Markdown(result.safety_result.safe_recipes_markdown),
                title=f"Recipes ({len(safe_recipes)} found)",
                border_style="green",
            ))
        else:
            console.print(Panel(
                "[bold yellow]No recipes found.[/bold yellow]\n"
                "Try rephrasing your request or relaxing some constraints.",
                border_style="yellow",
            ))

    asyncio.run(_run())


@app.command()
def chat() -> None:
    """Start an interactive AI chat session (requires login)."""
    session = _require_session()

    async def _run() -> None:
        factory = await _make_factory(full_init=True)
        ctx     = await _build_ctx(session, factory)
        agent   = factory.create_agent(ctx)

        label = session.login or f"user #{session.user_id}"
        console.print(Panel(
            f"[bold]Nutrition AI Chat[/bold]\n"
            f"Logged in as [bold]{label}[/bold]\n"
            "Type your question, or [bold]exit[/bold] / [bold]quit[/bold] to stop.",
            border_style="cyan",
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if user_input.strip().lower() in ("exit", "quit", "q", "bye"):
                console.print("[dim]Goodbye![/dim]")
                break

            if not user_input.strip():
                continue

            with console.status("[bold cyan]Thinking…", spinner="dots"):
                response = await agent.run(ctx, user_input)

            console.print()
            console.print(Panel(Markdown(response), title="NutriAI", border_style="green"))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Commands: Init (admin / first-time setup, no login required)
# ---------------------------------------------------------------------------

@app.command()
def init(
    rebuild: bool = typer.Option(
        False, "--rebuild", "-r",
        help="Force rebuild vectorstores from scratch.",
    ),
) -> None:
    """Build or rebuild the RAG vectorstores (admin / first-time setup).

    Does not require login.  Run this once before using ask or chat.
    """
    async def _run() -> None:
        config  = Settings.from_env()
        factory = ServiceFactory(config)
        with console.status(
            "[bold cyan]Initialising pipeline — this may take several minutes…",
            spinner="dots",
        ):
            await factory.initialize()
        console.print(Panel(
            "[bold green]Pipeline initialised![/bold green]\n"
            "Vectorstores are built and ready.\n"
            "Run [bold]login[/bold] and then [bold]chat[/bold] or [bold]ask[/bold] to start.",
            border_style="green",
        ))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Global version option
# ---------------------------------------------------------------------------

@app.callback()
def _callback(
    version: bool = typer.Option(
        False, "--version", "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Nutrition AI Assistant CLI"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
