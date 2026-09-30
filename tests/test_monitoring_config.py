"""Sanity checks for the committed Grafana dashboard and Prometheus rules.

Structure: valid JSON, unique panel ids, non-overlapping grid positions, every query on the Prometheus
datasource. Metrics: every ``finsight_*`` series referenced by the dashboard or the alert rules is exported
by the app (trace-fed and audit metrics after one sample trace, or the scrape-time ops collector).
``promtool check rules`` / ``promtool test rules`` run the rule semantics (see docs/a2a-and-observability.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from query_intelligence.agent.audit import AuditTraceSink
from query_intelligence.agent.telemetry import PrometheusTraceSink

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "monitoring" / "grafana" / "finsight-dashboard.json"
ALERTS = ROOT / "monitoring" / "prometheus" / "alerts.yml"
OPS_COLLECTOR = ROOT / "query_intelligence" / "integrations" / "ops_metrics.py"
_METRIC = re.compile(r"\bfinsight_[a-z_]+")
_SUFFIXES = ("_bucket", "_count", "_sum", "_created")


def _panels() -> list[dict]:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))["panels"]


def test_dashboard_structure():
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    panels = dashboard["panels"]
    assert dashboard["uid"] == "finsight-ops" and len(panels) == 30
    assert len({panel["id"] for panel in panels}) == len(panels)

    cells: dict[tuple[int, int], int] = {}
    for panel in panels:
        grid = panel["gridPos"]
        assert 0 <= grid["x"] and grid["x"] + grid["w"] <= 24
        for x in range(grid["x"], grid["x"] + grid["w"]):
            for y in range(grid["y"], grid["y"] + grid["h"]):
                assert (x, y) not in cells, f"panel {panel['id']} overlaps panel {cells[(x, y)]}"
                cells[(x, y)] = panel["id"]
        if panel["type"] == "row":
            continue
        assert panel["datasource"]["uid"] == "prometheus"
        for target in panel["targets"]:
            assert target["expr"].strip() and target["datasource"]["uid"] == "prometheus"


def test_quality_row_has_prompt_version_and_feedback_panels():
    titles = {panel["title"]: panel for panel in _panels()}
    repair = titles["Repair rate by prompt version"]["targets"][0]["expr"]
    assert 'outcome="repaired"' in repair and "by (prompt_version)" in repair
    feedback = titles["User feedback: thumbs-up ratio by prompt version"]["targets"][0]["expr"]
    assert 'rating="up"' in feedback and "by (prompt_version)" in feedback
    assert (
        "finsight_audit_events_total"
        in titles["Audit events per hour (refusals, compliance edits)"]["targets"][0]["expr"]
    )


def test_every_referenced_metric_is_exported():
    from prometheus_client import generate_latest

    sink = PrometheusTraceSink()
    AuditTraceSink(registry=sink.registry, path="off")
    sample = {
        "route": "agent",
        "answer_source": "llm_agent",
        "duration_ms": 10,
        "tools": [{"tool": "t", "ok": True, "latency_ms": 1}],
        "llm_calls": [{"model": "m", "prompt": "agent_system@v3#x", "prompt_tokens": 1}],
        "cost": 0.1,
        "currency": "CNY",
        "verification_passed": False,
        "degraded": ["x", "instruction_like_text_removed_from_evidence"],
        "route_reasons": ["input_guard:instruction_like_text_removed"],
        "compliance_notes": ["attributed_document_claim"],
    }
    sink.emit(sample)
    sink.record_feedback("up", "v3")
    exported = generate_latest(sink.registry).decode()
    exported += OPS_COLLECTOR.read_text(encoding="utf-8")  # scrape-time metric names are defined there

    referenced = set(_METRIC.findall(DASHBOARD.read_text(encoding="utf-8") + ALERTS.read_text(encoding="utf-8")))
    missing = []
    for name in sorted(referenced):
        base = next((name.removesuffix(suffix) for suffix in _SUFFIXES if name.endswith(suffix)), name)
        if base not in exported and base.removesuffix("_total") not in exported:
            missing.append(name)
    assert missing == []


def test_injection_redaction_panels_use_the_redaction_counter():
    titles = {panel["title"]: panel for panel in _panels()}
    series = titles["Injection-filter redactions per hour (source, answered / refused)"]["targets"][0]["expr"]
    answered = titles["Answered after input-guard redaction (24 h)"]["targets"][0]["expr"]

    assert "finsight_injection_redactions_total" in series and "by (source, outcome)" in series
    assert 'source="user_message"' in answered and 'outcome="answered"' in answered


def test_output_safety_panel_counts_edits_by_kind():
    from prometheus_client import generate_latest

    titles = {panel["title"]: panel for panel in _panels()}
    expr = titles["Output-safety edits per hour by kind"]["targets"][0]["expr"]
    assert "finsight_output_safety_edits_total" in expr and "by (kind)" in expr

    sink = PrometheusTraceSink()
    notes = [
        "attributed_document_claim",
        "attributed_document_claim",  # one run, one increment per kind
        "omitted_document_promotion",
        "removed_prohibited_promotion",  # the compliance guard's own note: not an output-layer edit
        "omitted_conflicting_document_figure",
    ]
    sink.emit({"route": "workflow", "answer_source": "llm_compose", "compliance_notes": notes})
    sink.emit({"route": "agent", "answer_source": "llm_agent", "compliance_notes": ["omitted_document_trading_call"]})
    sink.emit({"route": "workflow", "answer_source": "template", "compliance_notes": []})
    exported = generate_latest(sink.registry).decode()
    for kind in ("attribution", "promotion_or_contact", "conflicting_figure", "trading_call"):
        assert f'finsight_output_safety_edits_total{{kind="{kind}"}} 1.0' in exported, kind
    assert exported.count("finsight_output_safety_edits_total{") == 4
