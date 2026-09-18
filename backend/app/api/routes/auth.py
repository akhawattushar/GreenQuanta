"""Signup, login and the authenticated identity endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbDep
from app.core.config import get_settings
from app.core.security import create_access_token, hash_password, verify_password
from app.db.database import AuditRepository, DatabaseError, UserRepository
from app.schemas.models import LoginRequest, SignupRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(user: dict) -> TokenResponse:
    settings = get_settings()
    token = create_access_token(subject=user["id"], email=user["email"], role=user["role"])
    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserOut(**{k: user[k] for k in ("id", "name", "email", "role", "status", "created_at")}),
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account and return a JWT",
)
def register(payload: SignupRequest, db: DbDep) -> TokenResponse:
    settings = get_settings()
    users = UserRepository(db)
    email = payload.email.lower()

    if users.get_by_email(email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with that email already exists.")

    role = payload.role
    # The very first account bootstraps an administrator so the admin panel is
    # reachable; afterwards only addresses listed in ADMIN_EMAILS may self-assign.
    if role == "admin" and users.count() > 0 and email not in settings.admin_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The admin role must be granted by an existing administrator.",
        )
    if users.count() == 0:
        role = "admin"

    try:
        user = users.create(
            name=payload.name.strip(),
            email=email,
            password_hash=hash_password(payload.password),
            role=role,
        )
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    AuditRepository(db).log(user_id=user["id"], action="auth.register", detail=email)
    return _token_response(user)


@router.post("/login", response_model=TokenResponse, summary="Exchange credentials for a JWT")
def login(payload: LoginRequest, db: DbDep) -> TokenResponse:
    users = UserRepository(db)
    user = users.get_by_email(payload.email)
    # Same message either way so the endpoint does not confirm which emails exist.
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    AuditRepository(db).log(user_id=user["id"], action="auth.login", detail=user["email"])
    return _token_response(user)


@router.get("/me", response_model=UserOut, summary="Current authenticated user")
def me(user: CurrentUser) -> UserOut:
    return UserOut(**{k: user[k] for k in ("id", "name", "email", "role", "status", "created_at")})
