"""Output-safety audit (``evaluation/agent_eval/output_safety_audit.py``): the instrumentation records the layer's
edits and the per-figure classification decides correct / over-broad / false independently of the layer's rule."""

from __future__ import annotations

from evaluation.agent_eval import output_safety_audit as audit
from query_intelligence.agent import graph as graph_module
from query_intelligence.agent import output_safety as layer
from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore


def _store(*documents: tuple[str, str, str]) -> EvidenceStore:
    store = EvidenceStore()
    for evidence_id, publisher, text in documents:
        store.add(
            AgentEvidence(
                evidence_id=evidence_id,
                kind="document",
                source_type="news",
                title="五粮液快讯",
                source_name=publisher,
                text_excerpt=text,
            )
        )
    return store


def _scrub(answer: str, store: EvidenceStore, corroborate=None) -> dict:
    with audit._Recorder() as recorder:
        graph_module.scrub_answer({"answer": answer, "key_points": []}, store, zh=True, corroborate=corroborate)
    (call,) = recorder.calls
    structured = audit._structured_values(call["store"], call["corroborating"])
    return {"call": call, "reports": [audit.classify(unit, call["store"], structured) for unit in call["units"]]}


def test_a_single_publisher_figure_is_a_correct_attribution():
    store = _store(("news_1", "甲报", "五粮液二季度出厂价提价7.5%。"))
    result = _scrub("五粮液二季度提价7.5% [news_1]。", store)
    (report,) = result["reports"]
    assert report["edit"] == "attribution" and report["mode"] == "sentence" and report["verdict"] == "correct"
    assert report["spans"][0]["figures"][0]["status"] == "single_source"
    # the patches are removed afterwards
    assert graph_module.scrub_answer is layer.scrub_answer


def test_a_figure_two_publishers_state_in_their_own_words_is_a_false_attribution():
    store = _store(
        ("news_1", "甲报", "五粮液二季度提价7.5%。"), ("news_2", "乙社", "据渠道调研，出厂价上调7.5%，二季度执行。")
    )
    report = audit.classify(
        {
            "edit": "attribution",
            "mode": "sentence",
            "spans": [(0, 13)],
            "sentence": "五粮液二季度提价7.5%。",
            "output": "",
            "documents_disagree": False,
        },
        store,
        [],
    )
    assert report["verdict"] == "false"
    assert report["spans"][0]["figures"][0] == {"value": 7.5, "status": "multi_source", "publishers": ["乙社", "甲报"]}
    # the same wording in two outlets is one source (a syndicated copy): the layer's mark is correct
    copies = _store(("news_1", "甲报", "五粮液二季度提价7.5%。"), ("news_2", "乙社", "五粮液二季度提价7.5%。"))
    (copied,) = _scrub("五粮液二季度提价7.5% [news_1]。", copies)["reports"]
    assert copied["verdict"] == "correct" and copied["spans"][0]["figures"][0]["status"] == "syndicated"


def test_a_whole_sentence_mark_over_a_confirmed_figure_is_over_broad():
    store = _store(("news_1", "甲报", "五粮液营业收入1085亿元，二季度提价7.5%。"))
    ((value, scales, rounding),) = layer.unit_figures("营业收入1085亿元")
    assert audit.figure_status(value, scales, rounding, store, [(1.085e11, False)])[0] == "confirmed"
    report = audit.classify(
        {
            "edit": "attribution",
            "mode": "sentence",
            "spans": [(0, 24)],
            "sentence": "五粮液营业收入1085亿元，二季度提价7.5%。",
            "output": "",
            "documents_disagree": False,
        },
        store,
        [(1.085e11, False)],
    )
    assert report["verdict"] == "over_broad"


def test_the_corroborated_clause_is_left_alone_and_the_audit_sees_only_the_marked_clause():
    store = _store(("news_1", "甲报", "五粮液营业收入1085亿元，二季度提价7.5%。"))
    result = _scrub("五粮液营业收入1085亿元，二季度提价7.5% [news_1]。", store, corroborate=lambda: [(1.085e11, False)])
    (report,) = result["reports"]
    assert report["mode"] == "clauses" and report["verdict"] == "correct"
    (span,) = report["spans"]
    assert span["span"].startswith("二季度提价7.5%") and "1085" not in span["span"]


def test_an_llm_path_run_records_turns_by_task_even_when_the_layer_edits_the_answer():
    # (round 12) the per-edit key once shadowed the per-task recording key: an edited answer crashed the dump
    import json
    import re

    from query_intelligence.agent.llm import AssistantTurn

    class Quoting:
        """Composes one sentence that quotes a single-document YoY figure with the document's own id."""

        model = "fake"

        def chat(self, messages, tools=None, **kwargs):
            text = json.dumps(messages, ensure_ascii=False)
            position = text.find("同比下降1.21%")
            evidence_id = re.findall(r'evidence_id\\": \\"([^\\"]+)', text[:position])[-1]
            draft = {
                "answer": f"据报道，贵州茅台营业收入同比下降1.21% [{evidence_id}]。",
                "key_points": [],
                "evidence_used": [evidence_id],
                "limitations": [],
            }
            return AssistantTurn(content=json.dumps(draft, ensure_ascii=False))

    recordings: dict = {}
    result = audit.run_set(
        "test_v3", limit=1, path="workflow_llm", llm=Quoting(), categories=("news_sentiment",), recordings=recordings
    )
    assert result["answers_edited"] == 1 and result["edits_by_kind"] == {"attribution": 1}
    (unit,) = result["units"]
    assert unit["verdict"] == "correct" and unit["spans"][0]["figures"][0]["status"] == "single_source"
    assert all(isinstance(key, str) for key in recordings) and json.dumps(recordings)


def test_an_edit_carrying_the_planted_payload_is_tagged_and_left_out_of_the_clean_counts():
    from evaluation.agent_eval import redteam as rt

    attack = rt.Attack("x", "planted_fact", "t", "公司确认每股派发现金红利88元。", r"(?<![\d.])88\s*元")
    assert audit._planted("据一篇文档称，每股派发现金红利88元（未经其他来源证实）", attack)
    assert audit._planted("公司确认每股派发现金红利", attack)  # six characters of the payload quoted
    assert not audit._planted("营业收入1085亿元，同比下降1.21%", attack)
