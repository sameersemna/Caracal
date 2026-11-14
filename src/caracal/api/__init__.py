"""REST adapter: authentication, schemas and the ``/v1`` routes."""

from caracal.api.auth import require_auth

__all__ = ["require_auth"]
