from pydantic import BaseModel, create_model
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import UUID

from ..models import table


# Everything but the encrypted password
class User(BaseModel):
    user_id: Union[str, UUID]
    user_name: str
    user_jti: Union[str, UUID]
    user_email: str
    user_fullname: str
    user_subscription_type: Optional[str]
    user_subscription_start: Optional[datetime]
    user_subscription_end: Optional[datetime]
    created_at: datetime
    created_by: Union[str, UUID]
    crud_workspaces: bool
    crud_groups: bool
    share_workspaces: bool
