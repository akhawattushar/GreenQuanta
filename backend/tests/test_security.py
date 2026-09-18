"""Password hashing and JWT handling."""

import time

import pytest

from app.core.security import (
    TokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_round_trip():
    stored = hash_password("correct horse battery")
    assert stored != "correct horse battery"
    assert verify_password("correct horse battery", stored)
    assert not verify_password("wrong password", stored)


def test_password_hash_is_salted():
    assert hash_password("same") != hash_password("same")


def test_verify_rejects_garbage():
    assert not verify_password("x", "not-a-valid-hash")


def test_token_round_trip():
    token = create_access_token(subject="usr-1", email="a@b.com", role="admin")
    payload = decode_access_token(token)
    assert payload.sub == "usr-1"
    assert payload.email == "a@b.com"
    assert payload.role == "admin"
    assert payload.exp > time.time()


def test_tampered_signature_is_rejected():
    token = create_access_token(subject="usr-1", email="a@b.com", role="operator")
    head, payload, _ = token.split(".")
    with pytest.raises(TokenError):
        decode_access_token(f"{head}.{payload}.AAAA")


def test_expired_token_is_rejected():
    token = create_access_token(subject="u", email="a@b.com", role="operator", expires_minutes=-1)
    with pytest.raises(TokenError):
        decode_access_token(token)


def test_malformed_token_is_rejected():
    with pytest.raises(TokenError):
        decode_access_token("not-a-token")
