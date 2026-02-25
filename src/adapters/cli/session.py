"""
adapters.cli.session - Local session credential storage.

Credentials (user_id + JWT access_token) are stored in
~/.nutrition-ai/session.json so the user stays logged in between CLI
invocations without re-entering their password every time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

_SESSION_DIR  = Path.home() / ".nutrition-ai"
_SESSION_FILE = _SESSION_DIR / "session.json"


@dataclass
class Session:
    user_id: int
    access_token: str
    login: str = ""


def load_session() -> Session | None:
    """Return the stored session, or None if the user is not logged in."""
    if not _SESSION_FILE.exists():
        return None
    try:
        data = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
        return Session(**data)
    except Exception:
        return None


def save_session(session: Session) -> None:
    """Persist session credentials to disk."""
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(
        json.dumps(asdict(session), indent=2), encoding="utf-8"
    )


def clear_session() -> None:
    """Delete stored credentials (logout)."""
    if _SESSION_FILE.exists():
        _SESSION_FILE.unlink()
