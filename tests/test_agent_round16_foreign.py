"""Round-16 rule L7 (round-12 review), written with the engineer's own wording: a company's Hong Kong line is detected
from the dual-listing table (``data/runtime/dual_listings.csv``: A-share symbol, HKEX code, names and short names,
joined with the alias table's colloquial names), not from a name pattern. Before round 16 the name before 港股 / H股
could not contain 的, so 美的集团's Hong Kong line was answered with the A-share price.

The property test crosses every row of the table and every one of its names with line wordings (港股, H股, a Hong Kong
venue, its HKEX code, a Hong Kong-dollar quote; English H shares / Hong Kong shares / HKD / code), so a new row or a
new wording is checked against every counterpart. A-share questions about the same companies stay in scope.
"""

from __future__ import annotations

import csv
import re

import pytest

from query_intelligence.agent.coverage import (
    _AH_CODES,
    _alias_names,
    _dual_listing_rows,
    _row_names,
    dual_listing_spans,
    foreign_line_kind,
    foreign_line_spans,
    out_of_coverage,
)
from query_intelligence.data_loader import load_dual_listings, load_entities

_ROWS = _dual_listing_rows()
_ALIASES = _alias_names({row["a_symbol"] for row in _ROWS})

_ZH_TEMPLATES = [
    "{n}港股最新报价",
    "{n}的港股现在多少",
    "{n}H股收盘价",
    "{n}的H股市盈率",
    "{n}在港交所的股价",
    "{n}于联交所的成交额",
    "{n}在香港上市的股票价格",
    "{n}{c4}收盘多少",
    "{n}（{c5}）今天的走势",
    "{n}的港币报价",
    "{n}现在值多少港元",
]
_EN_TEMPLATES = [
    "{e} H shares last price",
    "{e}'s Hong Kong shares today",
    "{e} HK stock quote",
    "How much is {e} in HKD?",
    "{e} ({c5}.HK) close",
]
_ZH_CONTROLS = ["{n}A股收盘价", "{n}2025年营收", "{n}1000股市值多少", "{n}在香港有子公司吗", "{n}的市盈率"]


def _names(row: dict[str, str]) -> tuple[list[str], list[str]]:
    names = [*_row_names(row), *_ALIASES.get(row["a_symbol"], [])]
    zh = sorted({name for name in names if re.search(r"[一-鿿]", name)})
    en = sorted({name for name in names if not re.search(r"[一-鿿]", name)})
    return zh, en


def _covers(query: str, name: str) -> bool:
    """A foreign-line span covers the whole name (the agent drops the A-share target it names as a lookalike)."""
    start = query.find(name)
    return start >= 0 and any(lo <= start and start + len(name) <= hi for lo, hi in foreign_line_spans(query))


def test_the_table_is_data_with_a_source_and_matches_the_entity_master():
    rows = load_dual_listings()
    assert len(rows) >= 90 and len({row["a_symbol"] for row in rows}) == len(rows)
    names = {row["symbol"]: row["canonical_name"] for row in load_entities() if row.get("entity_type") == "stock"}
    for row in rows:
        assert re.fullmatch(r"\d{6}\.(?:SH|SZ)", row["a_symbol"]) and re.fullmatch(r"\d{5}", row["h_code"])
        assert names.get(row["a_symbol"]) == row["name_zh"], row
        assert row["hkex_name"] and row["source"].startswith("HKEX ListOfSecurities.xlsx")
    by_symbol = {row["a_symbol"]: row for row in rows}
    assert by_symbol["000333.SZ"]["h_code"] == "00300"  # 美的集团, the round-12 review's case
    assert "美的" in _ALIASES["000333.SZ"]  # a colloquial name joined from the alias table
    # the bare-code table of round 15 is derived from the same rows
    assert _AH_CODES["1211"] and "比亚迪" in _AH_CODES["1211"] and "0300" not in _AH_CODES


@pytest.mark.parametrize("row", _ROWS, ids=[row["a_symbol"] for row in _ROWS])
def test_every_name_of_every_row_with_every_line_wording_is_a_foreign_line(row):
    zh, en = _names(row)
    code5 = row["h_code"]
    checked = 0
    for name in zh:
        for template in _ZH_TEMPLATES:
            query = template.format(n=name, c4=code5[-4:], c5=code5)
            assert out_of_coverage(query) == "foreign_equity", query
            assert foreign_line_kind(query) == "h", query
            assert _covers(query, name), query
            checked += 1
    for name in en:
        for template in _EN_TEMPLATES:
            query = template.format(e=name, c5=code5)
            assert out_of_coverage(query) == "foreign_equity", query
            assert foreign_line_kind(query) == "h", query
            assert _covers(query, name), query
            checked += 1
    assert checked >= len(_ZH_TEMPLATES) + len(_EN_TEMPLATES)


@pytest.mark.parametrize("row", _ROWS, ids=[row["a_symbol"] for row in _ROWS])
def test_a_share_questions_about_the_same_companies_are_not_foreign_lines(row):
    for name in _names(row)[0]:
        for template in _ZH_CONTROLS:
            query = template.format(n=name)
            assert not dual_listing_spans(query), query


def test_gdr_and_adr_wordings_after_a_table_name_name_their_line():
    assert foreign_line_kind("美的集团的GDR价格") == "gdr" and _covers("美的集团的GDR价格", "美的集团")
    assert foreign_line_kind("Midea Group ADRs") == "adr" and _covers("Midea Group ADRs", "Midea Group")


def test_names_with_de_inside_are_matched_by_construction():
    # every table name that contains 的 is tested by the property test above; at least 美的 and 美的集团 are there
    with_de = [name for row in _ROWS for name in _names(row)[0] if "的" in name]
    assert {"美的", "美的集团"} <= set(with_de)


def test_the_csv_on_disk_is_what_the_loader_reads():
    from query_intelligence.data_loader import RUNTIME_DATA_DIR

    with (RUNTIME_DATA_DIR / "dual_listings.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(load_dual_listings())


# ---- end to end: 美的's Hong Kong line refused with the A-share note; its A shares answered ----


@pytest.fixture(scope="module")
def agent():
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
    )
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


@pytest.mark.parametrize(
    "query",
    ["美的的港股最新报价是多少", "美的集团在联交所的收盘价", "What's Midea's Hong Kong share price right now?"],
)
def test_midea_hong_kong_line_is_refused_with_the_a_share_note(agent, query):
    result = agent.chat(query, session_id=f"r16-midea-{abs(hash(query))}")
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"]
    assert "80.1" not in result["answer"]
    assert "美的集团在 A 股上市的股份在覆盖范围内" in result["answer"] or "Midea Group's A shares are covered" in (
        result["answer"]
    )


def test_midea_a_shares_are_still_answered(agent):
    result = agent.chat("美的集团A股最新收盘价", session_id="r16-midea-a")
    assert result["route"] != "refuse" and "80.1" in result["answer"]
