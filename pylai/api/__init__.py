from fastapi import FastAPI, Response, Depends, HTTPException
import json

from .v1.routes import chat
from .v1.routes import users

from .v1.utils import verify_session_token


def include_routers_v1(app: FastAPI):
    """
    Adds the backend Interface routes to the FastAPI. This is for Interface version 1.
    """
    app.include_router(chat.router, prefix="/api/v1", tags=["chatbot"])
    app.include_router(users.router, prefix="/api/v1", tags=["users"])

    @app.get("/health")
    async def health():
        return Response(status_code=200, content=json.dumps({"status": "ok"}))

    @app.get("/auth")
    async def auth(user_id: str = Depends(verify_session_token)):
        try:
            return Response(status_code=200, content=json.dumps({"user_id": user_id}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return None


def include_routers_v2(app: FastAPI):
    """
    Adds the backend Interface routes to the FastAPI. This is for Interface version 2.
    """

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return None
