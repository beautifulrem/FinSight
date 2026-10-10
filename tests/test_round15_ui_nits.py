"""Round 15 (review-11 K17): a negative per-share gain is bracketed in the cost-basis product ("200 × (−90.5)", not
"200 × -90.5"), in both languages, and the sentence still verifies."""

from __future__ import annotations

import pytest

from query_intelligence.agent.composer import cost_basis_sentences
from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.verifier import verify_answer

_TOOL_LOG = [
    {
        "tool": "get_price_history",
        "ok": True,
        "data": {
            "symbol": "600519.SH",
            "name": "贵州茅台",
            "close": 1409.5,
            "as_of": "2026-04-22",
            "evidence_id": "price_600519.SH",
        },
    }
]


def _store() -> EvidenceStore:
    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_600519.SH",
            kind="structured",
            source_type="market",
            payload={"close": 1409.5, "symbol": "600519.SH"},
        )
    )
    return store


@pytest.mark.parametrize(
    ("question", "zh", "product"),
    [
        ("我200股茅台成本1500，现在盈亏多少", True, "200 × (−90.5) = -18100 元"),
        ("I hold 200 shares of Moutai at a cost of 1500; what's my P&L?", False, "200 × (−90.5) = CNY -18100"),
    ],
)
def test_a_negative_per_share_gap_is_bracketed_and_verifies(question, zh, product):
    sentence = cost_basis_sentences(_TOOL_LOG, question, zh)[0]
    assert product in sentence, sentence
    assert "× -" not in sentence
    report = verify_answer({"answer": sentence}, _store(), query=question, allow_derived=True)
    assert report.passed, report


def test_a_positive_per_share_gap_is_not_bracketed():
    sentence = cost_basis_sentences(_TOOL_LOG, "我200股茅台成本1400，现在盈亏多少", True)[0]
    assert "200 × 9.5 = 1900 元" in sentence, sentence


def _python_templates() -> set[str]:
    """Every string literal and f-string of the server modules whose fixed texts the web UI re-localises, with the
    interpolated names written as {a} / {b} the way ``frontend/src/lib/localize.ts`` writes them."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    names = {"names[0]": "{a}", "names[1]": "{b}", "name": "{a}"}
    found: set[str] = set()
    for module in (
        "query_intelligence/agent/followups.py",
        "query_intelligence/answer_guards.py",
        "query_intelligence/chat/language.py",
    ):
        for node in ast.walk(ast.parse((root / module).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
            elif isinstance(node, ast.JoinedStr):
                parts = []
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        parts.append(str(value.value))
                    else:
                        parts.append(names.get(ast.unparse(value.value), "{?}"))
                found.add("".join(parts))
    return found


def test_the_ui_relocalised_templates_match_the_server_templates():
    """The web UI rebuilds next-question chips and the default disclaimer in the UI language from copies of the server
    templates (K17); a server wording change without the same change in localize.ts would silently stop the switch."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "localize.ts").read_text(
        encoding="utf-8"
    )
    copies = re.findall(r'(?:zh|en): "((?:[^"\\]|\\.)*)"', source)
    assert len(copies) >= 20
    server = _python_templates()
    missing = [copy for copy in copies if copy not in server]
    assert missing == []
