"""Content-level overlap of a held-out slice with own wording (round-11 review K16).

The text measures (normalised equality, 3-gram Jaccard >= 0.8, difflib >= 0.85) found 3 near copies in the round-10
slice; the review found 7 more slice tasks whose content (same company and number, or the same structure) repeats
fix-era tests and dev tasks in other words. ``slice_overlap.content_report`` adds that check with fixed lexicons.
"""

from __future__ import annotations

import json

import pytest

from evaluation.agent_eval import slice_overlap


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("宁德时代明年一季度净利润会不会超过200亿", {"2e+10"}),
        ("Zijin Mining's 2025 net profit rose about 62% year on year to RMB51.8 billion", {"62%", "5.18e+10"}),
        ("我有1.5万股工商银行，两千股招行", {"15000", "2000"}),
        ("比亚迪的营收有没有宁德时代的三倍半", {"3.5x"}),
        ("平安银行的PB只有招行的六成", {"60%"}),
        # years, dates, quarters, stock codes and the 52 of "52-week" are not numbers
        ("贵州茅台在2026年12月31日那天收盘价", set()),
        ("Give me BYD's share price at the end of Q2 2027 vs its 52-week high", set()),
        ("中芯国际0981.HK和600036", set()),
    ],
)
def test_numbers_are_normalised_and_dates_codes_and_52_week_are_not_numbers(text, expected):
    assert slice_overlap.numbers(text) == expected


def test_companies_take_the_longest_alias_and_english_words_only_at_word_boundaries():
    assert slice_overlap.companies("平安银行和中国平安") == {"平安银行", "中国平安"}
    assert slice_overlap.companies("平安银行的市盈率") == {"平安银行"}
    assert slice_overlap.companies("What's CMB's HK-traded share price?") == {"招商银行"}
    assert slice_overlap.companies("Is ICBC's market cap more than three times Zijin's?") == {"工商银行", "紫金矿业"}
    assert slice_overlap.companies("bybyd and catlike words") == frozenset()


def test_masked_structure_replaces_companies_and_numbers():
    one = slice_overlap.masked("我有招行1500股、平安银行2手、长江电力300股，按收盘价一共值多少")
    two = slice_overlap.masked("我有招行1000股、工行10手和紫金矿业500股，按收盘价一共值多少")
    assert one == "我有жш股жш手жш股按收盘价一共值多少"
    assert two == "我有жш股жш手和жш股按收盘价一共值多少"


def _scan(tmp_path, slice_rows, own_rows, test_source=""):
    root = tmp_path
    (root / "tests").mkdir()
    (root / "tests" / "test_x.py").write_text(test_source, encoding="utf-8")
    _write_jsonl(root / "evaluation" / "agent_eval" / "tasks" / "agent_eval_v1.jsonl", own_rows)
    slice_dir = root / "heldout_rX"
    _write_jsonl(slice_dir / "chat.jsonl", slice_rows)
    slice_overlap.own_texts.cache_clear()
    try:
        return slice_overlap.overlap_report(slice_dir, root=root), slice_overlap.content_report(slice_dir, root=root)
    finally:
        slice_overlap.own_texts.cache_clear()


def test_content_finds_paraphrases_and_structure_copies_that_the_text_measures_miss(tmp_path):
    own = [
        {"id": "dev_threshold", "turns": [{"query": "宁德时代今年四季度净利润能突破200亿吗"}]},
        {"id": "dev_holdings", "turns": [{"query": "我有招行1000股、工行10手和紫金矿业500股，按收盘价一共值多少"}]},
        {"id": "dev_share", "turns": [{"query": "比亚迪和宁德时代净利润加起来，比亚迪占几成"}]},
        {"id": "dev_other", "turns": [{"query": "茅台的市盈率是多少"}]},
    ]
    test_source = 'CASES = ["What were Midea\'s earnings excluding non-recurring items?"]\n'
    slice_rows = [
        # same company + same number, reworded: a paraphrase (difflib >= 0.70)
        {"id": "s_threshold", "turns": [{"query": "宁德时代四季度净利润会不会突破200亿"}]},
        # other companies and counts, same structure: a copy once names and numbers are masked
        {"id": "s_holdings", "turns": [{"query": "我有招行1500股、平安银行2手、长江电力300股，按收盘价一共值多少"}]},
        # same company + the 扣非 metric, reworded
        {"id": "s_dedt", "turns": [{"query": "What was Midea's recurring net profit, excluding non-recurring items?"}]},
        # the same two companies, operation and metric in other words: signature tier only
        {"id": "s_share_words", "turns": [{"query": "宁德时代和比亚迪两家净利润加总后，宁德时代占了多大比例"}]},
        # unrelated
        {"id": "s_unrelated", "turns": [{"query": "上证指数近一年最大回撤是多少"}]},
    ]

    text, content = _scan(tmp_path, slice_rows, own, test_source)

    assert text["tasks_with_overlap"] == []
    assert content["tasks_content"] == ["s_dedt", "s_holdings", "s_threshold"]
    assert "s_share_words" in content["tasks_content_signature"]
    assert "s_unrelated" not in content["tasks_content"] + content["tasks_content_signature"]
    hits = {(hit["task"], hit["tier"]): hit for hit in content["hits"]}
    assert hits[("s_holdings", "content")]["own"] == ["dev:dev_holdings"]
    assert hits[("s_holdings", "content")]["reasons"] == ["structure (companies and numbers masked)+operation"]
    assert hits[("s_threshold", "content")]["own"] == ["dev:dev_threshold"]
    assert "company+number" in hits[("s_threshold", "content")]["reasons"]
    assert hits[("s_dedt", "content")]["own"] == ["tests/test_x.py:1"]
    assert hits[("s_share_words", "content_signature")]["reasons"] == ["companies+operation+metric"]
    # ids, locations and reasons only: the report never carries slice text
    assert "宁德" not in json.dumps(content["hits"], ensure_ascii=False)


def test_the_committed_content_scans_reproduce_their_exclusion_counts():
    """The committed scans of the round-9 and round-10 slices (own texts at the after-fix commits) and their pass
    counts with each exclusion set; recomputed from the committed run files."""
    results = slice_overlap.ROOT / "evaluation" / "results"
    for name, chat_runs, claim_runs in (
        (
            "slice_overlap-heldout_r10-content-at-3480a96",
            ["chat_heldout_r10-auto-nollm-prefix", "chat_heldout_r10-auto-nollm-after-fix"],
            ["claims_heldout_r10-prefix", "claims_heldout_r10-after-fix"],
        ),
        (
            "slice_overlap-heldout_r9-content-at-da9f6af",
            ["chat_heldout_r9-auto-nollm-prefix", "chat_heldout_r9-auto-nollm-after-fix"],
            ["claims_heldout_r9-prefix", "claims_heldout_r9-after-fix"],
        ),
    ):
        committed = json.loads((results / f"{name}.json").read_text(encoding="utf-8"))
        excluded = committed["exclusions"]["excluded_tasks"]
        extra = {key: value for key, value in excluded.items() if key.startswith("review")}
        claim_ids = slice_overlap.slice_claim_ids(slice_overlap.ROOT / committed["slice"])
        again = slice_overlap.exclusions(committed, chat_runs, extra, claim_runs=claim_runs, claim_ids=claim_ids)
        assert again == committed["exclusions"], name
        assert committed["config"]["own_texts_commit"] in name


def test_the_round11_review_exclusion_of_the_round10_slice_is_11_of_38_to_29_of_38():
    committed = json.loads(
        (slice_overlap.ROOT / "evaluation" / "results" / "slice_overlap-heldout_r10-content-at-3480a96.json").read_text(
            encoding="utf-8"
        )
    )
    runs = committed["exclusions"]["runs"]
    assert runs["chat_heldout_r10-auto-nollm-prefix"]["without_text"] == "12/45"
    assert runs["chat_heldout_r10-auto-nollm-after-fix"]["without_text"] == "35/45"
    assert runs["chat_heldout_r10-auto-nollm-prefix"]["without_review_round11"] == "11/38"
    assert runs["chat_heldout_r10-auto-nollm-after-fix"]["without_review_round11"] == "29/38"
