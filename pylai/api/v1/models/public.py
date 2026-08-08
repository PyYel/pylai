from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import UUID

from ..models import table


# User public data that can be shared to anyone
class User(BaseModel):
    user_id: Union[str, UUID]
    user_name: str
    user_fullname: str
    created_at: datetime


# File metadata without the backend-specific fields
class File(BaseModel):
    file_id: Union[str, UUID]
    file_name: str
    added_at: datetime
    added_by: Union[str, UUID]


class WorkspaceUser(User):
    """
    Extension of the public.User table with workspace metadata
    """

    administrate: Optional[bool] = None
    moderate: Optional[bool] = None
    contribute: Optional[bool] = None


class Workspace(table.Workspaces):
    """
    Extension of the Workspaces table with assigned user metadata.
    """

    workspace_users: List[WorkspaceUser] = []
    workspace_files: List[table.Files] = []


class Conversation(table.Conversations):
    """
    Allows for smaller metadata only response.
    """

    conversation_messages: List[dict[str, str]] = []
    conversation_costs: dict[str, dict[str, float]] = {}


class GroupUser(table.Groups):
    """
    Public user information, may be aggregated across multiple tables.
    """

    user_id: Union[str, UUID]
    added_at: datetime
    added_by: Union[str, UUID]
    administrate: bool
    moderate: bool
