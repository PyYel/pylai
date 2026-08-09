# chat.py
import json
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    Query,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Response,
)
from pydantic import BaseModel
from datetime import datetime, timezone
from uuid import uuid4

from pylcloud.database import DatabaseRelationalPostgreSQL
from pylcloud.gpt import GPTAzure
from constants import (
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
    POSTGRES_SCHEMA,
    POSTGRES_PORT,
    SLM_MODEL_NAME,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
)

from ..models.table import Conversations, Messages
from ..utils import verify_session_token

router = APIRouter()

api_db = DatabaseRelationalPostgreSQL(
    host=POSTGRES_HOST,
    database=POSTGRES_DATABASE,
    schema=POSTGRES_SCHEMA,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    port=POSTGRES_PORT,
)

api_gpt = GPTAzure()


class SendMessageRequest(BaseModel):
    content: str


@router.get("/chat/conversations", response_model=list[Conversations])
async def get_conversations(user_id: str = Depends(verify_session_token)):
    """Returns the list of existing user conversations."""

    conversations = api_db.query_data(
        SELECT="id, name, assistant_slug, user_id, created_at, updated_at",
        FROM="conversations",
        WHERE=["user_id", "assistant_slug"],
        VALUES=[user_id, "ngsra"],  # We don't want to mix with genai4soc conversations
    )

    try:
        return [Conversations(**conversation) for conversation in conversations]

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/conversation", response_model=Conversations)
async def create_conversation(user_id: str = Depends(verify_session_token)):
    """Creates a new empty conversation."""

    try:

        created_at = datetime.now(tz=timezone.utc)
        updated_at = datetime.now(tz=timezone.utc)
        id = str(uuid4())

        api_db.send_data(
            table_name="conversations",
            id=id,
            user_id=user_id,
            assistant_slug="ngsra",
            name="New Chat",
            created_at=created_at,
            updated_at=updated_at,
        )

        return Conversations(
            id=id,
            name="New conversation",
            assistant_slug="ngsra",
            user_id=user_id,
            created_at=created_at,
            updated_at=updated_at,
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/edit", response_model=Conversations)
async def edit_conversation(
    conversation_id: str = Query(..., alias="conversation_id"),
    conversation_name: str = Query(..., alias="conversation_name"),
    user_id: str = Depends(verify_session_token),
):
    """Edits conversation metadata."""

    try:

        conversations = api_db.query_data(
            SELECT="id, name, assistant_slug, user_id, created_at, updated_at",
            FROM="conversations",
            WHERE=["user_id", "id"],
            VALUES=[user_id, conversation_id],
        )
        if conversations == []:
            raise HTTPException(404, detail=("Conversation not found"))
        else:
            conversation = Conversations(**conversations[0])

        conversation_name = conversation_name[:32]  # Cuts if too long

        api_db.update_data(
            table_name="conversations",
            WHERE=["user_id", "id"],
            VALUES=[user_id, conversation_id],
            name=conversation_name,
        )
        conversation.name = conversation_name

        return conversation

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/conversation")
async def delete_conversation(
    conversation_id: str = Query(..., alias="conversation_id"),
    user_id: str = Depends(verify_session_token),
):
    """Creates a new empty conversation."""

    try:

        api_db.delete_data(
            FROM="conversations",
            WHERE=["user_id", "id"],
            VALUES=[user_id, conversation_id],
        )
        return Response(status_code=208)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/messages", response_model=list[Messages])
async def get_messages(
    conversation_id: str = Query(..., alias="conversation_id"),
    user_id: str = Depends(verify_session_token),
):
    """Returns all messages of a conversation."""
    # TODO: Fetch messages where conversation_id = conversation_id

    try:

        # Checks if user has access to conversation first
        conversations = api_db.query_data(
            SELECT="id",
            FROM="conversations",
            WHERE=["user_id", "id"],
            VALUES=[user_id, conversation_id],
        )
        if conversations == []:
            raise HTTPException(403, detail=("Forbidden"))

        messages = api_db.query_data(
            SELECT="id, conversation_id, role, content, created_at, updated_at",
            FROM="messages",
            WHERE="conversation_id",
            VALUES=conversation_id,
        )

        return [Messages(**message) for message in messages]

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/message", response_model=Messages)
async def send_message(
    payload: SendMessageRequest,
    conversation_id: str = Query(..., alias="conversation_id"),
    user_id: str = Depends(verify_session_token),
):

    try:

        # Checks if user has access to conversation first
        conversations = api_db.query_data(
            SELECT="id",
            FROM="conversations",
            WHERE=["user_id", "id"],
            VALUES=[user_id, conversation_id],
        )
        if conversations == []:
            raise HTTPException(403, detail=("Forbidden"))

        # Gets conversation history (10 last messages) TODO: history/memory management
        messages = api_db.query_data(
            SELECT="content, role",
            FROM="messages",
            WHERE="conversation_id",
            VALUES=conversation_id,
        )[-10:]

        if messages == []:
            api_db.update_data(
                table_name="conversations",
                WHERE=["user_id", "id"],
                VALUES=[user_id, conversation_id],
                name=payload.content[:32],
            )

        # Inserts the user message to the db
        api_db.send_data(
            table_name="messages",
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=payload.content,
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
        )

        extra_body = {
            "user": "test_user",
            "metadata": {
                "session_id": "conversation_id",
                "tags": ["production", "tas"],
                "environment": "dev",
            },
        }

        response = api_gpt.return_generation(
            model_name=LLM_MODEL_NAME,
            user_prompt=payload.content,
            messages=messages,
            extra_body=extra_body,
        )

        created_at = datetime.now(tz=timezone.utc)
        updated_at = datetime.now(tz=timezone.utc)
        id = str(uuid4())

        if response:
            # Inserts LLM response
            api_db.send_data(
                table_name="messages",
                id=str(uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=response["text"],
                created_at=datetime.now(tz=timezone.utc),
                updated_at=datetime.now(tz=timezone.utc),
            )

            return Messages(
                id=id,
                conversation_id=conversation_id,
                content=response["text"],
                role="assistant",
                created_at=created_at,
                updated_at=updated_at,
            )

        else:
            raise HTTPException(
                status_code=500, detail="An error occured when generating a response"
            )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/chat/stream")
async def chat_stream(
    websocket: WebSocket,
    conversation_id: str = Query(...),
    message_id: str = Query(...),
):
    """
    Streams assistant response tokens for a given message_id.
    Authenticates once via the first JSON message received over the socket.
    """
    await websocket.accept()
    user_id = None

    try:
        # 1. Wait for Auth Frame from frontend
        auth_data_raw = await websocket.receive_text()
        auth_data = json.loads(auth_data_raw)

        if auth_data.get("type") == "auth" and "token" in auth_data:
            # Check the DB once right here upon opening the stream
            user_id = await verify_session_token(auth_data["token"])

            if not user_id:
                await websocket.close(
                    code=4003, reason="Invalid or expired session token"
                )
                return
        else:
            await websocket.close(code=4003, reason="Authentication required")
            return

        # ─── Connection is now fully authenticated ───────────────────────────
        print(f"[WS Stream] Handshake complete. User {user_id} authorized.")

        # 2. Stream tokens (Your Generation Logic)
        tokens = ["This ", "is ", "a ", "streamed ", "response."]

        for token in tokens:
            await websocket.send_json({"token": token, "done": False})

        # Send final frame completing the stream
        await websocket.send_json({"token": "", "done": True})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected cleanly for user: {user_id}")
    except Exception as e:
        print(f"Error in stream: {e}")
        # Only try to close if the connection hasn't been closed yet
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except RuntimeError:
            pass  # Socket might already be abandoned
