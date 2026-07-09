"""Manage the Keycloak access whitelist (``allowed_users``).

Only matters when AUTH_ENABLED is on: a verified Keycloak identity must have a
matching, non-blocked row here to use the API, and ``is_admin`` grants the
admin role. Blocking is the "deny" direction — it keeps the row (audit trail)
but refuses access.

Run inside the container:
    docker compose exec api uv run python scripts/manage_access.py list
    docker compose exec api uv run python scripts/manage_access.py add alice@bi.group --admin
    docker compose exec api uv run python scripts/manage_access.py add bob@bi.group --note "contractor"
    docker compose exec api uv run python scripts/manage_access.py set-admin alice@bi.group --off
    docker compose exec api uv run python scripts/manage_access.py block bob@bi.group
    docker compose exec api uv run python scripts/manage_access.py unblock bob@bi.group
    docker compose exec api uv run python scripts/manage_access.py remove bob@bi.group
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from llm_wiki.api.deps import _SessionLocal
from llm_wiki.storage.metadata import AllowedUser, _norm_email


async def _list() -> None:
    async with _SessionLocal() as session:
        rows = (
            (await session.execute(select(AllowedUser).order_by(AllowedUser.email)))
            .scalars()
            .all()
        )
    if not rows:
        print("(whitelist is empty — with AUTH_ENABLED nobody can log in)")
        return
    print(f"{'email':<40} {'admin':<6} {'blocked':<8} note")
    print("-" * 72)
    for r in rows:
        print(
            f"{r.email:<40} {'yes' if r.is_admin else 'no':<6} "
            f"{'YES' if r.blocked else 'no':<8} {r.note or ''}"
        )


async def _add(email: str, admin: bool, note: str | None) -> None:
    email = _norm_email(email)
    async with _SessionLocal() as session:
        existing = await session.get(AllowedUser, email)
        if existing is not None:
            existing.is_admin = admin
            if note is not None:
                existing.note = note
            existing.blocked = False
            action = "updated (unblocked)"
        else:
            session.add(AllowedUser(email=email, is_admin=admin, note=note))
            action = "added"
        await session.commit()
    print(f"{action}: {email} (admin={'yes' if admin else 'no'})")


async def _set_flag(email: str, *, blocked: bool | None = None, admin: bool | None = None) -> None:
    email = _norm_email(email)
    async with _SessionLocal() as session:
        row = await session.get(AllowedUser, email)
        if row is None:
            print(f"not found: {email}")
            return
        if blocked is not None:
            row.blocked = blocked
        if admin is not None:
            row.is_admin = admin
        await session.commit()
        print(
            f"{email}: admin={'yes' if row.is_admin else 'no'} "
            f"blocked={'YES' if row.blocked else 'no'}"
        )


async def _remove(email: str) -> None:
    email = _norm_email(email)
    async with _SessionLocal() as session:
        row = await session.get(AllowedUser, email)
        if row is None:
            print(f"not found: {email}")
            return
        await session.delete(row)
        await session.commit()
    print(f"removed: {email}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the Keycloak access whitelist")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all whitelist rows")

    p_add = sub.add_parser("add", help="Add or update a user (also unblocks)")
    p_add.add_argument("email")
    p_add.add_argument("--admin", action="store_true", help="Grant admin role")
    p_add.add_argument("--note", default=None, help="Free-text note")

    p_admin = sub.add_parser("set-admin", help="Toggle the admin role")
    p_admin.add_argument("email")
    p_admin.add_argument("--off", action="store_true", help="Revoke admin (default grants)")

    p_block = sub.add_parser("block", help="Deny access (keeps the row)")
    p_block.add_argument("email")

    p_unblock = sub.add_parser("unblock", help="Re-allow a blocked user")
    p_unblock.add_argument("email")

    p_rm = sub.add_parser("remove", help="Delete a user from the whitelist")
    p_rm.add_argument("email")

    args = parser.parse_args()

    if args.cmd == "list":
        asyncio.run(_list())
    elif args.cmd == "add":
        asyncio.run(_add(args.email, args.admin, args.note))
    elif args.cmd == "set-admin":
        asyncio.run(_set_flag(args.email, admin=not args.off))
    elif args.cmd == "block":
        asyncio.run(_set_flag(args.email, blocked=True))
    elif args.cmd == "unblock":
        asyncio.run(_set_flag(args.email, blocked=False))
    elif args.cmd == "remove":
        asyncio.run(_remove(args.email))


if __name__ == "__main__":
    main()
