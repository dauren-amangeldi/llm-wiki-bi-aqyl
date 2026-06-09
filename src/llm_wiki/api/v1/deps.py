"""Dependency helpers for API v1 — thin auth stub (no real validation for MVP)."""

from fastapi import Request


def get_current_user(request: Request) -> dict:  # type: ignore[return]
    """Return a demo user derived from request headers (no auth enforcement)."""
    return {
        "email": request.headers.get("X-User-Email", "demo@bi.group"),
        "role": request.headers.get("X-User-Role", "admin"),
        "business_unit": request.headers.get("X-Business-Unit", "HQ"),
        "geo": request.headers.get("X-User-Geo", "KZ"),
        "position": request.headers.get("X-User-Position", "employee"),
    }
