"""WebSocket endpoint for real-time conversational agent."""

from datetime import datetime, timedelta
from uuid import uuid4

import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query

from adapters.rest.dependencies import (
    get_factory, get_current_user, build_session_ctx, CurrentUser,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Reuse a conversation that had activity within this window
_CONVERSATION_REUSE_HOURS = 48


@router.websocket("/ws/chat")
async def websocket_chat(
    ws: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket chat endpoint.

    Authentication via query parameter: /ws/chat?token=<JWT>
    (WebSocket handshake does not support Authorization headers in browsers.)

    Conversation persistence:
      - If the user has a conversation active within the last 48 h, reuse it
        so the agent remembers prior context and the Flutter app can display
        previous messages.
      - Otherwise a fresh conversation is started.

    Protocol:
      - Client sends: plain text message
      - Server sends: plain text agent response
      - On auth failure: close with code 4001
      - On unhandled error: send plain-text error then close 1011
    """
    factory = get_factory()
    auth_service = factory.create_authentication_service()

    try:
        payload = auth_service.verify_token(token)
    except Exception:
        await ws.close(code=4001, reason="Invalid or expired token")
        return

    await ws.accept()

    user_id = payload["user_id"]
    conversation_id = await _resolve_conversation_id(user_id, factory)
    await _cleanup_old_data(user_id, factory)

    ctx = await build_session_ctx(user_id, conversation_id, factory)
    agent = factory.create_agent(ctx)

    try:
        while True:
            message = await ws.receive_text()
            logger.info("WS user=%d conv=%s | %s", user_id, conversation_id, message[:200])
            ctx.new_request()
            response = await agent.run(ctx, message)
            await ws.send_text(response)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Unhandled error in WS handler for user %d", user_id)
        try:
            await ws.send_text(
                f"Sorry, an unexpected error occurred: {exc}\n\nPlease reconnect and try again."
            )
        except Exception:
            pass
        await ws.close(code=1011)


@router.get("/chat/history")
async def get_chat_history(
    hours: int = Query(default=24, ge=1, le=168),
    user: CurrentUser = Depends(get_current_user),
    factory=Depends(get_factory),
):
    """Return recent chat messages for the current user's latest conversation.

    Fetches messages from the last `hours` hours (default 24, max 168 = 7 days).
    Used by the Flutter app to pre-populate the chat screen on reconnect.
    """
    chat_service = factory.create_chat_history_service()
    conversations = await chat_service.list_conversations(user.user_id)
    if not conversations:
        return {"messages": [], "conversation_id": None}

    latest = conversations[0]  # newest first
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

    if latest.last_message_at < cutoff:
        return {"messages": [], "conversation_id": latest.conversation_id}

    all_messages = await chat_service.load_history(latest.conversation_id)
    recent = [m for m in all_messages if m.created_at >= cutoff]

    return {
        "conversation_id": latest.conversation_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at}
            for m in recent
        ],
    }


async def _cleanup_old_data(user_id: int, factory) -> None:
    """Soft-delete messages and conversations older than the reuse window.

    Called on every WebSocket connect so the DB doesn't grow indefinitely.
    Failures are swallowed — the session is never interrupted by cleanup.
    """
    try:
        cutoff = (datetime.now() - timedelta(hours=_CONVERSATION_REUSE_HOURS)).isoformat()
        chat_service = factory.create_chat_history_service()
        await chat_service.purge_old_data(user_id, cutoff)
    except Exception:
        logger.exception("Old-data cleanup failed for user %d — continuing", user_id)


async def _resolve_conversation_id(user_id: int, factory) -> str:
    """Return the latest conversation_id if active within 48 h, else a new one."""
    try:
        chat_service = factory.create_chat_history_service()
        conversations = await chat_service.list_conversations(user_id)
        if conversations:
            latest = conversations[0]
            cutoff = (datetime.now() - timedelta(hours=_CONVERSATION_REUSE_HOURS)).isoformat()
            if latest.last_message_at >= cutoff:
                logger.info(
                    "Reusing conversation %s for user %d",
                    latest.conversation_id, user_id,
                )
                return latest.conversation_id
    except Exception:
        logger.exception("Failed to look up latest conversation — starting fresh")
    return uuid4().hex
