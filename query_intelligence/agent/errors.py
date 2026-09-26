"""Exceptions shared by the agent service and the API (kept free of heavy imports)."""

from __future__ import annotations


class SessionAccessError(LookupError):
    """The session belongs to another caller (reported as 404 so ids cannot be probed)."""
