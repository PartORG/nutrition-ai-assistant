"""WebSocket endpoint for real-time conversational agent."""

from uuid import uuid4

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from adapters.rest.dependencies import get_factory, build_session_ctx

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/chat")
async def websocket_chat(
    ws: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket chat endpoint.

    Authentication via query parameter: /ws/chat?token=<JWT>
    (WebSocket handshake does not support Authorization headers in browsers.)

    Protocol:
      - Client sends: plain text message
      - Server sends: plain text agent response
      - On auth failure: close with code 4001
      - On unhandled error: send JSON {"error": "..."} then close with code 1011
    """
    factory = get_factory()
    auth_service = factory.create_authentication_service()

    # Validate token before accepting the connection
    try:
        payload = auth_service.verify_token(token)
    except Exception:
        await ws.close(code=4001, reason="Invalid or expired token")
        return

    await ws.accept()

    user_id = payload["user_id"]
    conversation_id = uuid4().hex

    ctx = await build_session_ctx(user_id, conversation_id, factory)
    agent = factory.create_agent(ctx)

    try:
        while True:
            message = await ws.receive_text()
            logger.info("WS user=%d | %s", user_id, message[:200])
            ctx.new_request()
            response = await agent.run(ctx, message)
            await ws.send_text(response)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Unhandled error in WS handler for user %d", user_id)
        try:
            # Send as plain text so the Flutter chat renders it as a readable
            # message rather than displaying raw JSON in the bubble.
            await ws.send_text(
                f"Sorry, an unexpected error occurred: {exc}\n\nPlease reconnect and try again."
            )
        except Exception:
            pass
        await ws.close(code=1011)
