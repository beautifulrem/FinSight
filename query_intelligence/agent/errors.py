"""Exceptions shared by the agent service and the API (kept free of heavy imports)."""

from __future__ import annotations


class SessionAccessError(LookupError):
    """The session belongs to another caller (reported as 404 so ids cannot be probed)."""


class NoPendingClarificationError(ValueError):
    """``/agent/resume`` on a session that is not waiting for a clarification (reported as 409).

    ``code`` is machine-readable: ``no_pending_clarification`` when the session never asked, or when a
    different reply arrives after the clarification was already answered.
    """

    code = "no_pending_clarification"

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} has no pending clarification")
        self.session_id = session_id
