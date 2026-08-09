from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from uuid import UUID


# DB Tables
# Union[str, UUID] for input (as UUID) / output (as str) verification compatibility
class User(BaseModel):
    id: Union[str, UUID]
    name: str
    email: str
    username: str
    created_at: datetime
    updated_at: datetime


class Conversations(BaseModel):
    id: Union[str, UUID]
    name: str
    user_id: Union[str, UUID]
    assistant_slug: str
    created_at: datetime
    updated_at: datetime


class Messages(BaseModel):
    id: Union[str, UUID]
    conversation_id: Union[str, UUID]
    content: str
    role: Literal["user", "assistant", "agent"]
    created_at: datetime
    updated_at: datetime


class ApiKey(BaseModel):
    id: Union[str, UUID]
    name: str
    key: str
    user_id: Union[str, UUID]
    created_at: datetime
    updated_at: datetime
