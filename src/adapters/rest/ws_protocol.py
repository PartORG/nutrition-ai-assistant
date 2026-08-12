"""
adapters.rest.ws_protocol - Typed frames for the chat WebSocket.

The socket used to carry bare text in both directions, which meant an error and
an answer arrived identically: `send_text("Sorry, an unexpected error…")` looked
exactly like a recipe recommendation to the client, so the Flutter app rendered
failures as if the assistant had said them.

Every server frame is now a JSON object with a `type`:

    {"type": "status", "stage": "thinking", "detail": "Searching recipes…"}
    {"type": "answer", "content": "...", "conversation_id": "..."}
    {"type": "error",  "message": "...", "recoverable": true}

`status` frames exist because a single turn can take tens of seconds — the LLM
runs locally and a recipe search calls retrieval, generation and a safety pass.
Without them the UI is blank and the user cannot tell working from hung.

Clients that predate this protocol receive JSON where they expected prose. That
is a breaking change to the socket, made deliberately: the previous behaviour
could not distinguish success from failure at all.
"""

from __future__ import annotations

import json
from typing import Any, Literal

FrameType = Literal["status", "answer", "error"]

# Stages a client may want to render differently. Anything else is still valid;
# a client should fall back to showing `detail`.
STAGE_THINKING = "thinking"
STAGE_TOOL = "tool"
STAGE_FINALIZING = "finalizing"


def status_frame(stage: str, detail: str = "") -> str:
    return json.dumps({"type": "status", "stage": stage, "detail": detail})


def answer_frame(content: str, conversation_id: str = "") -> str:
    return json.dumps(
        {"type": "answer", "content": content, "conversation_id": conversation_id}
    )


def error_frame(message: str, recoverable: bool = True) -> str:
    """An error the client must render as a failure, not as assistant speech."""
    return json.dumps(
        {"type": "error", "message": message, "recoverable": recoverable}
    )


def parse_frame(raw: str) -> dict[str, Any]:
    """Parse a frame. Used by tests and any Python client."""
    frame = json.loads(raw)
    if not isinstance(frame, dict) or "type" not in frame:
        raise ValueError(f"Malformed frame: {raw[:120]}")
    return frame
