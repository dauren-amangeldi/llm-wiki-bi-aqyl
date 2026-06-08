"""FastAPI dependencies for the v1 API adapter.

Phase 0: authentication is bypassed — get_current_user always succeeds,
returning headers if present or a hardcoded demo identity if absent.
Phase 7 will replace this with a real JWT/session check (401 on missing auth).
"""

from fastapi import Request

_DEFAULTS = {
    "email": "demo@bi.group",
    "role": "employee",
    "business_unit": "HQ",
    "geo": "KZ",
    "position": "employee",
}


def get_current_user(request: Request) -> dict:  # type: ignore[type-arg]
    """Extract user context from request headers.

    Reads X-User-Email, X-User-Role, X-Business-Unit, X-User-Geo,
    X-User-Position.  Missing headers fall back to demo defaults so the
    frontend can call any endpoint without an auth token during development.

    Returns:
        Dict with keys: email, role, business_unit, geo, position.
    """
    return {
        "email": request.headers.get("X-User-Email", _DEFAULTS["email"]),
        "role": request.headers.get("X-User-Role", _DEFAULTS["role"]),
        "business_unit": request.headers.get("X-Business-Unit", _DEFAULTS["business_unit"]),
        "geo": request.headers.get("X-User-Geo", _DEFAULTS["geo"]),
        "position": request.headers.get("X-User-Position", _DEFAULTS["position"]),
    }
