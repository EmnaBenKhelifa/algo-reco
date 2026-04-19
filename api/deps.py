from fastapi import Header, HTTPException

from api.config import get_settings


async def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    key = get_settings()["api_key"]
    if not key:
        raise HTTPException(
            status_code=503,
            detail="PREDICTIONS_API_KEY is not configured on the server",
        )
    if not x_api_key or x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
