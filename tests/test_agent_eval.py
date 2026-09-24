from __future__ import annotations

import json
from datetime import date

from agent_fakes import StubService, build_fake_registry

from evaluation.agent_eval.build_tasks import build_tasks, check_overlap
from evaluation.agent_eval.metrics import aggregate, behavior_of, breakdown, failed_checks, percentile, score_turn
from evaluation.agent_eval.replay import RecordingRegistry, ReplayRegistry, call_key
from evaluation.agent_eval.runner import run_agent_tasks, summarize
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService

EXPECT = {
    "behavior": "answer",
    "required_tools": ["get_price_history"],
    "required_facts": [{"evidence_id": "price_600519.SH", "value": 1409.5}],
    "forbidden_patterns": [r"建议买入"],
    "must_hedge": True,
}


def _response(**overrides):
    base = {
        "route": "workflow",
        "answer": "可能受多因素影响。最新收盘价 1409.5 元 [price_600519.SH]。",
        "key_points": [],
        "evidence_used": ["price_600519.SH"],
        "tool_calls": [{"tool": "get_price_history", "ok": True}, {"tool": "search_news", "ok": True}],
        "verification": {"passed": True, "invalid_citations": [], "unsupported_numbers": []},
        "risk_disclaimer": "不构成投资建议",
        "compliance_notes": [],
        "llm": {"calls": 0, "usage": {}},
    }
    base.update(overrides)
    return base


def test_score_turn_success_and_tool_metrics():
    score = score_turn(_response(), EXPECT)

    assert score["success"] is True
    assert score["tool_recall"] == 1.0 and score["tool_precision"] == 0.5
    assert score["facts"] == [{"evidence_id": "price_600519.SH", "value": 1409.5, "stated": True, "cited": True}]


def test_score_turn_dealbreakers():
    uncited = score_turn(_response(evidence_used=[]), EXPECT)
    wrong_value = score_turn(_response(answer="可能。收盘价 1500 元 [price_600519.SH]。"), EXPECT)
    trading = score_turn(_response(answer="可能。收盘 1409.5 元 [price_600519.SH]，建议买入。"), EXPECT)
    refused = score_turn(_response(route="refuse"), EXPECT)

    assert not uncited["success"] and uncited["facts"][0]["cited"] is False
    assert not wrong_value["success"] and wrong_value["facts"][0]["stated"] is False
    assert not trading["success"] and trading["forbidden_hits"] == [r"建议买入"]
    assert not refused["success"] and refused["behavior"] == "refuse"


def test_behavior_detection():
    assert behavior_of({"status": "needs_clarification"}) == "clarify"
    assert behavior_of({"route": "clarify"}) == "clarify"
    assert behavior_of({"route": "refuse"}) == "refuse"
    assert behavior_of({"route": "agent"}) == "answer"


def test_aggregate_pass_k_and_breakdowns():
    ok = {"score": score_turn(_response(), EXPECT), "latency_ms": 10.0}
    bad = {"score": score_turn(_response(route="refuse"), EXPECT), "latency_ms": 30.0}
    records = [
        {"task": {"id": "a", "category": "fact", "language": "zh"}, "repeat": 0, "turns": [ok]},
        {"task": {"id": "a", "category": "fact", "language": "zh"}, "repeat": 1, "turns": [bad]},
        {"task": {"id": "b", "category": "why", "language": "en"}, "repeat": 0, "turns": [ok]},
        {"task": {"id": "b", "category": "why", "language": "en"}, "repeat": 1, "turns": [ok]},
    ]

    summary = aggregate(records, repeats=2)

    assert summary["tasks"] == 2
    assert summary["task_success"] == 0.75  # (0.5 + 1.0) / 2
    assert summary["pass^2"] == 0.5
    assert summary["latency_ms_p95"] == 30.0
    assert breakdown(records, "category")["why"]["task_success"] == 1.0
    assert failed_checks(records)["behavior"] == 1
    assert percentile([1, 2, 3, 4], 0.5) == 2 and percentile([], 0.5) is None


def test_task_set_shape_and_no_overlap_with_training():
    tasks = build_tasks()

    assert len(tasks) >= 200
    assert sum(len(task["turns"]) for task in tasks) >= 210
    assert {task["category"] for task in tasks} >= {
        "fact",
        "compare",
        "why",
        "macro_link",
        "missing_data",
        "multi_turn",
        "out_of_scope",
        "clarify",
        "compliance",
    }
    assert check_overlap(tasks) == []


def test_committed_task_file_matches_builder():
    from evaluation.agent_eval.build_tasks import TASKS_PATH

    committed = [json.loads(line) for line in TASKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert committed == build_tasks()


def test_record_and_replay_roundtrip(tmp_path):
    live = build_fake_registry()
    recorder = RecordingRegistry(live)
    recorded = recorder.run("get_price_history", {"target": "600519.SH"})
    failed = recorder.run("get_price_history", {"target": "UNKNOWN"})
    recorder.run("get_price_history", {"target": ""})  # invalid arguments are not recorded
    path = tmp_path / "snapshot.json"
    recorder.save(path, snapshot="test")

    replay = ReplayRegistry.from_file(path, live.specs())
    replayed = replay.run("get_price_history", '{"target": "600519.SH"}')
    replayed_failure = replay.run("get_price_history", {"target": "UNKNOWN"})
    missing = replay.run("get_fundamentals", {"target": "600519.SH"})

    assert len(json.loads(path.read_text(encoding="utf-8"))["calls"]) == 2
    assert replayed.ok and replayed.data == recorded.data
    assert [item.evidence_id for item in replayed.evidence] == ["price_600519.SH"]
    assert not failed.ok and replayed_failure.error.code == failed.error.code == "not_found"
    assert missing.error.code == "unavailable" and "not recorded" in missing.error.message
    assert replay.misses == [call_key("get_fundamentals", {"target": "600519.SH"})]


def test_replay_falls_back_to_live_and_records():
    live = build_fake_registry()
    replay = ReplayRegistry(live.specs(), {}, fallback=live)

    result = replay.run("get_fundamentals", {"target": "600519.SH"})

    assert result.ok and call_key("get_fundamentals", {"target": "600519.SH"}) in replay.calls


def test_runner_on_stub_service():
    runtime = AgentRuntime(StubService(), build_fake_registry(), None, today=lambda: date(2026, 4, 23))
    agent = AgentService(runtime, trace_sinks=[])
    tasks = [
        {
            "id": "t1",
            "category": "fact",
            "language": "zh",
            "turns": [{"query": "贵州茅台的市盈率是多少", "expect": {"required_tools": ["get_fundamentals"]}}],
        },
        {
            "id": "t2",
            "category": "out_of_scope",
            "language": "zh",
            "turns": [{"query": "今天天气怎么样", "expect": {"behavior": "refuse"}}],
        },
    ]

    records = run_agent_tasks(tasks, agent, mode="workflow")
    report = summarize(records, config={"mode": "workflow"}, repeats=1)

    assert report["summary"]["task_success"] == 1.0
    assert report["by_category"]["out_of_scope"]["tasks"] == 1
    assert report["failures"] == []


def test_gate_threshold_check():
    from evaluation.agent_eval.gate import THRESHOLDS, check

    assert check({"task_success": 0.99, "behavior_accuracy": 1.0, "compliance_clean": 1.0}, THRESHOLDS["holdout"]) == []
    problems = check({"task_success": 0.5, "behavior_accuracy": None, "compliance_clean": 1.0}, THRESHOLDS["holdout"])
    assert problems == ["task_success=0.5 < 0.8", "behavior_accuracy=None < 0.95"]
