from __future__ import annotations

import json

from query_intelligence.agent.tracing import JsonFileTraceSink, build_trace
from scripts.feedback_to_tasks import FORBIDDEN, build_candidates, is_flagged, load_traces, main


def _trace(run_id, query, *, session="s1", turn=0, route="workflow", tools=(), unsupported=()):
    result = {
        "run_id": run_id,
        "turn_index": turn,
        "query": query,
        "route": route,
        "route_reasons": ["simple:single_lookup"] if route == "workflow" else [f"nlu:{route}"],
        "answer_source": "template",
        "spans": [{"node": "route", "started_at": 1_790_000_000.0 + turn, "duration_ms": 5}],
        "tool_calls": [{"tool": name, "arguments": args, "ok": True} for name, args in tools],
        "verification": {"passed": not unsupported, "unsupported_numbers": list(unsupported)},
        "llm": {"model": "scripted"},
    }
    return build_trace(result, session_id=session)


def _write(tmp_path, traces, feedback):
    sink = JsonFileTraceSink(tmp_path / "traces")
    for trace in traces:
        sink.emit(trace)
    path = tmp_path / "feedback.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in feedback), encoding="utf-8")
    return path


def test_rating_normalisation():
    assert is_flagged("down") and is_flagged(-1) and is_flagged(0) and is_flagged(False) and is_flagged("Thumbs_Down")
    assert not is_flagged("up") and not is_flagged(1) and not is_flagged(True)


def test_flagged_traces_become_reviewable_candidate_tasks(tmp_path):
    price = ("get_price_history", {"target": "600519.SH", "days": 10})
    traces = [
        _trace("t-first", "贵州茅台最新收盘价", tools=[price]),
        _trace("t-follow", "它现在能不能抄底", turn=1, tools=[price]),
        _trace("t-refused", "它会涨吗", session="s2", route="refuse"),
        _trace("t-invented", "What is BYD's P/E?", session="s3", unsupported=[21.4]),
        _trace("t-liked", "五粮液的市盈率", session="s4"),
    ]
    feedback = [
        {"trace_id": "t-follow", "rating": "down", "comment": "说得太绝对，像在荐股"},
        {"trace_id": "t-refused", "rating": -1, "comment": "应该问我是哪只股票，而不是拒绝"},
        {"trace_id": "t-invented", "rating": 0, "comment": "the P/E looks made up"},
        {"trace_id": "t-liked", "rating": "up", "comment": "great"},
        {"trace_id": "t-unknown", "rating": "down", "comment": "?"},
        {"trace_id": "t-follow", "rating": "down", "comment": "duplicate report"},
    ]
    feedback_path = _write(tmp_path, traces, feedback)
    out = tmp_path / "candidates.jsonl"

    stats = main(["--feedback", str(feedback_path), "--traces", str(tmp_path / "traces"), "--out", str(out)])
    tasks = {task["review"]["trace_id"]: task for task in map(json.loads, out.read_text(encoding="utf-8").splitlines())}

    assert stats == {"feedback": 6, "flagged": 5, "missing_traces": ["t-unknown"], "duplicates": 1, "candidates": 3}
    follow = tasks["t-follow"]
    assert follow["category"] == "multi_turn" and follow["language"] == "zh"
    assert [turn["query"] for turn in follow["turns"]] == ["贵州茅台最新收盘价", "它现在能不能抄底"]
    last = follow["turns"][-1]["expect"]
    assert last["behavior"] == "answer" and last["must_hedge"] and last["required_entity"] == "600519.SH"
    assert last["any_of_tools"] == ["get_price_history"] and last["forbidden_patterns"] == FORBIDDEN
    assert follow["review"]["status"] == "needs_human_review" and follow["review"]["comment"] == "说得太绝对，像在荐股"
    assert any("required_facts" in item for item in follow["review"]["todo"])

    refused = tasks["t-refused"]
    assert refused["turns"][-1]["expect"]["behavior"] == "clarify" and refused["category"] == "clarify"
    assert refused["review"]["observed"]["route"] == "refuse"

    invented = tasks["t-invented"]
    assert invented["language"] == "en" and invented["turns"][-1]["expect"]["must_state_missing"]


def test_candidates_use_the_eval_task_schema(tmp_path):
    from evaluation.agent_eval.build_test_v2 import FORBIDDEN as TEST_SET_FORBIDDEN
    from evaluation.agent_eval.metrics import score_turn

    traces = load_traces(_write(tmp_path, [_trace("t1", "今天天气怎么样", route="refuse")], []).parent / "traces")
    candidates, _ = build_candidates([{"trace_id": "t1", "rating": "down", "comment": ""}], traces)
    task = candidates[0]

    assert FORBIDDEN == TEST_SET_FORBIDDEN  # one policy for what counts as a trading instruction
    assert set(task) >= {"id", "category", "language", "turns"} and task["id"].startswith("fb_")
    # A thumbs-down on a refusal proposes "answer" but asks the reviewer to confirm it.
    assert task["turns"][0]["expect"]["behavior"] == "answer"
    assert any("confirm the expected behaviour" in item for item in task["review"]["todo"])
    score = score_turn({"route": "refuse"}, task["turns"][0]["expect"])  # the schema is scoreable as is
    assert score["checks"]["behavior"] is False
