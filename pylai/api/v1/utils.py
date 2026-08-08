from datetime import datetime, timezone
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pycloud.database import DatabaseRelationalPostgreSQL
from constants import POSTGRES_DATABASE, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_USER, POSTGRES_SCHEMA, POSTGRES_PORT

api_db = DatabaseRelationalPostgreSQL(
    host=POSTGRES_HOST,
    database=POSTGRES_DATABASE,
    schema=POSTGRES_SCHEMA,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    port=POSTGRES_PORT,
)

security = HTTPBearer()

async def verify_session_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """
    Retreives the user_id from an auth token (better auth session token) by
    reading the session table. If a user_id is returned, then it means this session token is valid.
    """

    try:
            
        token = credentials.credentials
        if not token:
            raise HTTPException(status_code=401, detail="Missing token")
        
        # Checks the better auth managed session table
        session_record = api_db.query_data(
            SELECT="user_id", # TODO: upgrade with expire_at check as well ?
            FROM="session",
            WHERE="token",
            VALUES=token
        )

        if not session_record:
            raise HTTPException(status_code=401, detail="Invalid session token")
        else:
            return session_record[0]["user_id"]

    except Exception as e:
        print(f"verify_session_token error: {e}")
        raise HTTPException(status_code=500, detail="Auth failed")

