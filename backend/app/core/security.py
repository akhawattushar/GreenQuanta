"""Password hashing and HS256 JWT issuance/verification.

Implemented on the standard library (hashlib / hmac) so the whole auth path can
be unit-tested without installing native crypto wheels. The token format is a
standard compact JWS, so `PyJWT` or `python-jose` can decode it unchanged if you
later swap this module out.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from app.core.config import get_settings

PBKDF2_ROUNDS = 240_000
ALGORITHM = "HS256"


class TokenError(Exception):
    """Raised when a JWT is malformed, tampered with, or expired."""


# --------------------------------------------------------------------------
# password hashing
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Return `pbkdf2_sha256$rounds$salt$hash`."""
    if not password:
        raise ValueError("Password must not be empty.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return "$".join(
        ["pbkdf2_sha256", str(PBKDF2_ROUNDS), base64.b64encode(salt).decode(), base64.b64encode(digest).decode()]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, rounds, salt_b64, digest_b64 = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(expected, actual)


# --------------------------------------------------------------------------
# JWT
# --------------------------------------------------------------------------
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _sign(message: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()


@dataclass(frozen=True)
class TokenPayload:
    sub: str
    email: str
    role: str
    exp: int
    iat: int


def create_access_token(*, subject: str, email: str, role: str, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    now = int(time.time())
    ttl = expires_minutes if expires_minutes is not None else settings.access_token_expire_minutes
    payload = {
        "sub": str(subject),
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + ttl * 60,
        "iss": "quantafleet-api",
    }
    header = {"alg": ALGORITHM, "typ": "JWT"}
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode()),
        _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()),
    ]
    signing_input = ".".join(segments).encode("ascii")
    segments.append(_b64url_encode(_sign(signing_input, settings.secret_key)))
    return ".".join(segments)


def decode_access_token(token: str) -> TokenPayload:
    settings = get_settings()
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise TokenError("Malformed token.") from exc

    try:
        header = json.loads(_b64url_decode(header_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenError("Malformed token header.") from exc

    if header.get("alg") != ALGORITHM:
        raise TokenError("Unsupported token algorithm.")

    expected_sig = _sign(f"{header_b64}.{payload_b64}".encode("ascii"), settings.secret_key)
    try:
        provided_sig = _b64url_decode(signature_b64)
    except ValueError as exc:
        raise TokenError("Malformed token signature.") from exc
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise TokenError("Invalid token signature.")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenError("Malformed token payload.") from exc

    if int(payload.get("exp", 0)) < int(time.time()):
        raise TokenError("Token has expired.")

    for claim in ("sub", "email", "role"):
        if claim not in payload:
            raise TokenError(f"Token is missing the {claim!r} claim.")

    return TokenPayload(
        sub=str(payload["sub"]),
        email=str(payload["email"]),
        role=str(payload["role"]),
        exp=int(payload["exp"]),
        iat=int(payload.get("iat", 0)),
    )
