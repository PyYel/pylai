# user.py
from fastapi import APIRouter, Depends
from fastapi import APIRouter, HTTPException, Query, status, Response
from datetime import datetime, timezone

from pylcloud.database import DatabaseRelationalPostgreSQL
from constants import (
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
    POSTGRES_SCHEMA,
    POSTGRES_PORT,
)

from ..models.table import User, ApiKey
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


@router.get("/users/info", response_model=User)
async def get_user_info(user_id: str = Depends(verify_session_token)):
    """Returns user info from db."""
    # TODO: Fetch user profile data using user_id

    try:

        user_info = api_db.query_data(
            SELECT="id, name, email, username, created_at, updated_at",
            FROM="user",
            WHERE="id",
            VALUES=user_id,
        )

        return User(**user_info[0])

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/apikey", response_model=ApiKey)
async def get_user_api_key(user_id: str = Depends(verify_session_token)):
    """Returns user api key info."""
    # TODO: Fetch user api key metadata (masked key, creation date, etc.)

    try:

        return ApiKey(
            id="id",
            name="key",
            user_id="test_user",
            key="keysecret",
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
