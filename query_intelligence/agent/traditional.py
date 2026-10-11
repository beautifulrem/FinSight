"""Re-export of ``query_intelligence.traditional`` (the converter moved there in round 17 so the NLU can use it)."""

from __future__ import annotations

from ..traditional import has_traditional, to_simplified

__all__ = ["has_traditional", "to_simplified"]
