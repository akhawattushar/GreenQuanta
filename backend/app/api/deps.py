"""Shared FastAPI dependencies: database handle and JWT-authenticated user."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import TokenError, decode_access_token
from app.db.database import Database, UserRepository

_db: Database | None = None

bearer_scheme = HTTPBearer(auto_error=False, description="JWT issued by /auth/login")


def init_database(db: Database) -> None:
    global _db
    _db = db


def get_db() -> Database:
    if _db is None:  # pragma: no cover - set during app startup
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not initialised.",
        )
    return _db


DbDep = Annotated[Database, Depends(get_db)]
CredsDep = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]


def get_current_user(credentials: CredsDep, db: DbDep) -> dict:
    """Reject every request without a valid, unexpired bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = UserRepository(db).get(payload.sub)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account no longer exists.")
    if user.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is not active.")
    return user


CurrentUser = Annotated[dict, Depends(get_current_user)]


def require_roles(*roles: str):
    """Dependency factory for role-based access control."""

    def _guard(user: CurrentUser) -> dict:
        if roles and user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires one of: {', '.join(roles)}.",
            )
        return user

    return _guard


AdminUser = Annotated[dict, Depends(require_roles("admin"))]
#: Roles allowed to trigger compute-heavy runs. Regulators/viewers are read-only.
WriteUser = Annotated[dict, Depends(require_roles("operator", "admin", "researcher"))]
