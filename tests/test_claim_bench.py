"""Claim-check benchmark: the runner scores a small subset, and the claim files are well formed."""

from __future__ import annotations

import hashlib
import re

from agent_fakes import StubService, build_fake_registry

from evaluation.claim_bench.run import BENCH_DIR, SETS, VERDICTS, evaluate, load_claims

TINY = [
    {
        "id": "t1",
        "lang": "zh",
        "category": "comparator",
        "claim": "贵州茅台ROE超过30%",
        "expected_verdict": "supported",
        "expected_checks": [{"metric": "roe", "comparator": "gt", "status": "supported"}],
    },
    {
        "id": "t2",
        "lang": "zh",
        "category": "multi",
        "claim": "贵州茅台收盘价1409.5元，市盈率40倍",
        "expected_verdict": "partially_supported",
        "expected_checks": [
            {"metric": "close", "comparator": "eq", "status": "supported"},
            {"metric": "pe_ttm", "comparator": "eq", "status": "contradicted"},
        ],
    },
    {
        "id": "t3",
        "lang": "zh",
        "category": "opinion",
        "claim": "今天天气不错",
        "expected_verdict": "unverifiable",
        "expected_checks": [],
    },
    {
        # deliberately mislabelled: the runner must count it as an error
        "id": "t4",
        "lang": "en",
        "category": "false",
        "claim": "Moutai's P/E is 24.6",
        "expected_verdict": "contradicted",
        "expected_checks": [{"metric": "pe_ttm", "comparator": "eq", "status": "contradicted"}],
    },
]


def test_runner_scores_a_tiny_subset():
    report = evaluate(TINY, service=StubService(), registry=build_fake_registry())

    assert report["claims"] == 4 and report["checks"] == 4
    assert report["verdict_accuracy"] == 0.75
    assert report["check_accuracy"] == 0.75
    assert report["comparator_accuracy"] == 1.0
    low, high = report["verdict_accuracy_ci"]
    assert 0.0 <= low <= 0.75 <= high <= 1.0
    assert report["verdict_confusion"]["contradicted"]["supported"] == 1
    assert report["check_status_confusion"]["contradicted"]["supported"] == 1
    assert [error["id"] for error in report["errors"]] == ["t4"]
    assert set(report["verdict_confusion"]) == set(VERDICTS)
    assert report["by_language"]["en"] == {"claims": 1, "verdict_accuracy": 0.0}


def test_claim_files_are_well_formed_and_the_holdout_is_unchanged():
    ids = set()
    for name, path in SETS.items():
        rows = load_claims(path)
        assert len(rows) >= (100 if name == "dev" else 40)
        for row in rows:
            assert row["id"] not in ids
            ids.add(row["id"])
            assert row["expected_verdict"] in VERDICTS
            assert {check["status"] for check in row["expected_checks"]} <= {"supported", "contradicted", "unverifiable"}
    recorded = re.search(r"([0-9a-f]{64})\s+claims_v1_holdout\.jsonl", (BENCH_DIR / "README.md").read_text())
    assert recorded is not None
    assert hashlib.sha256(SETS["holdout"].read_bytes()).hexdigest() == recorded.group(1)


def test_ci_floors_fail_only_below_the_committed_accuracy():
    from evaluation.claim_bench.run import regressions

    at_head = {"verdict_accuracy": 0.9787, "check_accuracy": 0.9815}  # 46/47 verdicts, 53/54 checks
    assert regressions(at_head, verdict_floor=0.978, check_floor=0.98) == []
    one_more_error = {"verdict_accuracy": 0.9574, "check_accuracy": 0.963}
    assert regressions(one_more_error, verdict_floor=0.978, check_floor=0.98) == [
        "verdict accuracy 0.9574 < 0.978",
        "check accuracy 0.963 < 0.98",
    ]
    assert regressions(one_more_error, verdict_floor=None, check_floor=None) == []
