"""
API Key Authentication
======================
Simple header-based API key guard.  Reads the key from the ``API_KEY``
environment variable.  If the variable is **not set**, authentication is
disabled so local development works without extra config.

Protected endpoints receive the dependency ``require_api_key``.  Public
endpoints (health, docs, headshots) are excluded automatically by the
middleware so they never require a key.
"""

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"

_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

# Read key from environment — ``None`` means auth is disabled.
_CONFIGURED_KEY: Optional[str] = os.getenv("API_KEY")


def _is_auth_enabled() -> bool:
    return _CONFIGURED_KEY is not None and len(_CONFIGURED_KEY) > 0


async def require_api_key(
    key: Optional[str] = Security(_api_key_scheme),
) -> Optional[str]:
    """FastAPI dependency — raises 401 if the key is wrong.
    When ``API_KEY`` env-var is unset the check is skipped entirely."""
    if not _is_auth_enabled():
        return None  # auth disabled
    if not key or not secrets.compare_digest(key, _CONFIGURED_KEY):  # type: ignore[arg-type]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    return key
