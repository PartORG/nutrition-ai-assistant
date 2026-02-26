"""Protected conversation history endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from factory import ServiceFactory
from adapters.rest.dependencies import get_factory, get_current_user, CurrentUser
from adapters.rest.schemas import ConversationOut, MessageOut

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    service = factory.create_chat_history_service()
    conversations = await service.list_conversations(user.user_id)
    return [
        ConversationOut(
            conversation_id=c.conversation_id,
            title=c.title,
            last_message_at=c.last_message_at,
            created_at=c.created_at,
        )
        for c in conversations
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
)
async def get_messages(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    service = factory.create_chat_history_service()
    messages = await service.load_history(conversation_id)
    # Ensure user owns this conversation
    if messages and messages[0].user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Not your conversation.")
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
        )
        for m in messages
    ]
