"""Bearer-token authentication (docs/03 §1, docs/06 S1).

The token is compared with :func:`hmac.compare_digest` on bytes. A missing, malformed or
wrong header is ``401 unauthorized``; the API exception handler adds ``WWW-Authenticate``.
"""

from __future__ import annotations

import hmac

from fastapi import Request

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode

_SCHEME = "Bearer"


def _unauthorized(message: str = "invalid or missing bearer token") -> CaracalError:
    return CaracalError(ErrorCode.UNAUTHORIZED, message)


def _settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise CaracalError(ErrorCode.INTERNAL, "application settings are not configured")
    return settings


def require_auth(request: Request) -> None:
    """FastAPI dependency: validate the bearer token and expose it on ``request.state``."""
    header = request.headers.get("authorization")
    if header is None:
        raise _unauthorized("missing authorization header")
    scheme, separator, token = header.partition(" ")
    if not separator or scheme != _SCHEME or not token:
        raise _unauthorized()
    expected = _settings(request).CARACAL_API_TOKEN.get_secret_value()
    if not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise _unauthorized()
    request.state.api_token = token


def bearer_token(request: Request) -> str:
    """Return the already-validated token (used to build the rate-limit identity)."""
    token = getattr(request.state, "api_token", None)
    if not isinstance(token, str):
        raise _unauthorized()
    return token
