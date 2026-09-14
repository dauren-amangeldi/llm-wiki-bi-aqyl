import json
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llm_wiki.api.v1 import auth
from llm_wiki.storage.metadata import AccessDecision, User


async def test_verified_profile_upsert_keeps_role_and_updates_name(db_session, monkeypatch):
    claims = {"email": "alice@bi.group", "name": "Alice Example", "given_name": "Alice"}
    monkeypatch.setattr(auth, "verify_access_token", lambda _: claims)
    monkeypatch.setattr(auth, "access_for_email", AsyncMock(return_value=AccessDecision(True, False, "ok")))
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer test-token")]})
    first = await auth.auth_me(request, db_session)
    assert json.loads(first.body)["role"] == "employee"
    profile = await db_session.get(User, "alice@bi.group")
    assert profile.name == "Alice Example"
    claims["name"] = "Alice Updated"
    second = await auth.auth_me(request, db_session)
    await db_session.refresh(profile)
    assert profile.name == "Alice Updated"
    assert json.loads(second.body)["name"] == "Alice Updated"


async def test_denied_identity_cannot_create_profile(db_session, monkeypatch):
    monkeypatch.setattr(auth, "verify_access_token", lambda _: {"email": "blocked@bi.group"})
    monkeypatch.setattr(auth, "access_for_email", AsyncMock(return_value=AccessDecision(False, False, "blocked")))
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer test-token")]})
    with pytest.raises(HTTPException) as exc:
        await auth.auth_me(request, db_session)
    assert exc.value.status_code == 403
    assert await db_session.get(User, "blocked@bi.group") is None
