"""Test-only Locust login: explicit DB allowlist, separate signer, employee role."""

from __future__ import annotations

import re
import secrets
import time
from typing import Any

import jwt
import structlog
from fastapi import Header, HTTPException
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.config import settings
from llm_wiki.storage.metadata import AccessDecision, AllowedUser, access_for_email

logger = structlog.get_logger(__name__)
TOKEN_KID = "aqyl-load-test-v1"
TOKEN_ISSUER = "urn:aqyl:test:load-test"
TOKEN_AUDIENCE = "aqyl-test-api"
ACCOUNT_PATTERN = re.compile(r"loadtest-([0-9]{4})@aqyl\.test\.invalid\Z")


def account_email(number: int) -> str:
    if not 1 <= number <= 1000:
        raise ValueError("Load-test account number must be between 1 and 1000")
    return f"loadtest-{number:04d}@aqyl.test.invalid"


def is_test_email(email: str) -> bool:
    match = ACCOUNT_PATTERN.fullmatch(email)
    return bool(match and 1 <= int(match.group(1)) <= 1000)


def require_load_test_login(
    x_load_test_secret: str | None = Header(default=None),
) -> None:
    if not settings.load_test_auth_active:
        raise HTTPException(404, "Not found")
    supplied = (x_load_test_secret or "").encode()
    expected = settings.load_test_login_secret.encode()
    if not secrets.compare_digest(supplied, expected):
        logger.info("load_test_login_rejected", reason="invalid_credentials")
        raise HTTPException(401, "Invalid load-test credentials")


async def load_test_access(session: AsyncSession, email: str) -> AccessDecision:
    if not settings.load_test_auth_active or not is_test_email(email):
        return AccessDecision(False, False, "load_test_disabled")
    row = await session.get(AllowedUser, email, populate_existing=True)
    if row is None or not row.load_test_enabled or row.blocked:
        return AccessDecision(False, False, "load_test_not_allowed")
    # Even an accidentally elevated whitelist row cannot make a test token admin.
    return AccessDecision(True, False, "ok")


async def access_for_claims(session: AsyncSession, claims: dict[str, Any]) -> AccessDecision:
    from llm_wiki.api.auth import claims_email

    email = claims_email(claims)
    if claims.get("auth_source") == "load_test":
        return await load_test_access(session, email)
    return await access_for_email(session, email)


def issue_load_test_token(email: str) -> str:
    if not settings.load_test_auth_active or not is_test_email(email):
        raise HTTPException(404, "Not found")
    now = int(time.time())
    return jwt.encode(
        {
            "iss": TOKEN_ISSUER, "aud": TOKEN_AUDIENCE,
            "sub": email, "email": email, "name": f"Load test {email[9:13]}",
            "auth_source": "load_test", "iat": now, "nbf": now,
            "exp": now + settings.load_test_token_ttl_s, "jti": secrets.token_hex(16),
        },
        settings.load_test_signing_secret, algorithm="HS256", headers={"kid": TOKEN_KID},
    )


def verify_load_test_token(token: str) -> dict[str, Any]:
    if not settings.load_test_auth_active:
        raise HTTPException(401, "Invalid or expired token")
    try:
        claims = jwt.decode(
            token, settings.load_test_signing_secret, algorithms=["HS256"],
            issuer=TOKEN_ISSUER, audience=TOKEN_AUDIENCE,
            options={"require": ["iss", "aud", "sub", "email", "auth_source",
                                 "iat", "nbf", "exp", "jti"]},
        )
        email = claims["email"]
        if (
            not isinstance(email, str) or not is_test_email(email)
            or claims["sub"] != email or claims["auth_source"] != "load_test"
            or claims["exp"] - claims["iat"] > settings.load_test_token_ttl_s
        ):
            raise ValueError("Invalid test claims")
        return claims
    except Exception as exc:
        logger.info("load_test_token_rejected", error_type=type(exc).__name__)
        raise HTTPException(401, "Invalid or expired token") from exc


async def seed_load_test_users(session: AsyncSession) -> int:
    """Idempotent startup seed; never unblocks/re-enables an existing account."""
    if not settings.load_test_auth_active:
        return 0
    emails = [account_email(n) for n in range(1, settings.load_test_user_count + 1)]
    result = await session.execute(
        insert(AllowedUser).values([
            {"email": email, "is_admin": False, "blocked": False,
             "load_test_enabled": True, "note": "Dedicated Locust test identity"}
            for email in emails
        ]).on_conflict_do_nothing(index_elements=[AllowedUser.email]).returning(AllowedUser.email)
    )
    count = len(result.scalars().all())
    await session.commit()
    return count
