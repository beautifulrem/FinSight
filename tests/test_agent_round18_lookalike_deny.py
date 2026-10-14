"""Round-18 rule N5 (round-14 review): an unknown extension of a covered company's name is another company.

Round 17 read a covered short name followed by a *business-line word* (a word list) as another company, so names with
a word outside the list were still answered with the covered company's figures: "紫金黄金的股价" (Zijin Gold, a Hong
Kong company) got 紫金矿业's 29.79, "Zijin Gold share price" Zijin Mining's tile, and "比亚迪半导体", "宁德新能源",
"招银理财", "中芯南方", "平安科技" their parents' figures. Round 18 denies by default (``nlu/lookalike.py``): a name
word right after a listed company's name or alias (a noun by jieba's dictionary, or any word that occurs in a listed
company's name; in English a capitalised word) makes an unknown company, unless it is a word that can follow a company
name in a question (metrics, time words, legal forms …), the company's own name going on, its own industry, another
listed name, or part of a longer alias in the entity table (registered names such as 宁德时代新能源科技 are aliases).
A verb, pronoun, time word or numeral after the name ("茅台我按1380买的", "换五粮液看看") is the rest of the question.

Property test: every name of the covered companies in the entity table (short names and official names) x twelve
seeded draws from the words of listed companies' names (A-share and HKEX-only names segmented by jieba: 黄金, 半导体,
科技, 能源 …) x four frames never resolves to the covered company and leaves a lookalike trace; controls (the same
names followed by question words, metrics and time words) keep their company. The decision changes over every
evaluation query string are measured by ``evaluation/agent_eval/lookalike_regression.py``.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from query_intelligence.nlu import lookalike

ROOT = Path(__file__).resolve().parents[1]
COVERED = {
    "000001.SZ": "平安银行",
    "000333.SZ": "美的集团",
    "000858.SZ": "五粮液",
    "002594.SZ": "比亚迪",
    "300750.SZ": "宁德时代",
    "600030.SH": "中信证券",
    "600036.SH": "招商银行",
    "600519.SH": "贵州茅台",
    "600900.SH": "长江电力",
    "601318.SH": "中国平安",
    "601398.SH": "工商银行",
    "601899.SH": "紫金矿业",
    "688981.SH": "中芯国际",
}
_ZH_FRAMES = ("{n}的股价", "{n}最新收盘价是多少", "{n}的市盈率是多少", "{n}营收多少")
_SUFFIXES_PER_NAME = 12


@pytest.fixture(scope="module")
def pipeline(shipped_offline_service):
    return shipped_offline_service.nlu_pipeline


def _resolve(pipeline, text: str, *, full: bool) -> tuple[set[str], list[str]]:
    normalized, normal_trace = pipeline.normalizer.normalize(text)
    resolver = pipeline.entity_resolver
    entities, _targets, trace = resolver.resolve_exact(normalized)
    if full and not [entity for entity in entities if entity.get("symbol")]:
        entities, _targets, trace = resolver.resolve(normalized)
    return {str(entity["symbol"]) for entity in entities if entity.get("symbol")}, [*normal_trace, *trace]


def _covered_names() -> dict[str, set[str]]:
    """Every Chinese name of the covered companies in the entity table (official and short) -> their symbols."""
    ids = {}
    with open(ROOT / "data/runtime/entity_master.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["symbol"] in COVERED and row["entity_type"] == "stock":
                ids[str(int(row["entity_id"]))] = row["symbol"]
    names: dict[str, set[str]] = {}
    with open(ROOT / "data/runtime/alias_table.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = ids.get(row["entity_id"])
            text = row["alias_text"]
            # the entity table's quirks (fund names filed as official names of a stock) are not names of the company
            if (
                symbol
                and row["alias_type"] != "symbol"
                and len(text) >= 2
                and not text.isascii()
                and (row["alias_type"] != "official_name" or COVERED[symbol] in text)
            ):
                names.setdefault(text, set()).add(symbol)
    return names


def _all_aliases() -> set[str]:
    with open(ROOT / "data/runtime/alias_table.csv", encoding="utf-8") as handle:
        return {row["normalized_alias"] for row in csv.DictReader(handle)}


def _company_name_words() -> list[str]:
    """Words of two or more characters in the names of listed companies (the A-share stocks of the entity master and
    the HKEX-only names), segmented with jieba's dictionary: 黄金, 半导体, 科技, 能源, 电子, 药业, 置地 …"""
    import jieba

    tokenizer = jieba.Tokenizer()
    names = []
    with open(ROOT / "data/runtime/entity_master.csv", encoding="utf-8") as handle:
        names += [row["canonical_name"] for row in csv.DictReader(handle) if row["entity_type"] == "stock"]
    with open(ROOT / "data/runtime/hk_only_names.csv", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = next(index for index, row in enumerate(rows) if "name_zh" in row)
    column = rows[header].index("name_zh")
    names += [row[column] for row in rows[header + 1 :] if len(row) > column]
    words = {
        word
        for name in names
        for word in tokenizer.cut(name, HMM=False)
        if len(word) >= 2 and all("一" <= char <= "鿿" for char in word)
    }
    return sorted(words)


def _cases() -> list[tuple[str, str, frozenset[str]]]:
    rng = random.Random(18)
    aliases = _all_aliases()
    with open(ROOT / "data/runtime/hk_only_names.csv", encoding="utf-8") as handle:
        hk_names = {row[1] for row in csv.reader(handle) if len(row) > 1}
    words = _company_name_words()
    cases = []
    for name, symbols in sorted(_covered_names().items()):
        chosen: set[str] = set()
        for word in rng.sample(words, len(words)):
            if len(chosen) == _SUFFIXES_PER_NAME:
                break
            extended = name + word
            owner_goes_on = any(extended[: len(name) + 1] in COVERED[symbol] for symbol in symbols)
            if owner_goes_on or lookalike._ZH_CONTINUATION.match(word) or any(a.startswith(extended) for a in aliases):
                continue  # the company's own name, a question word, or a name in the entity table: not unknown
            if any(word[:size] in aliases | hk_names for size in range(2, len(word) + 1)):
                continue  # the word starts with another company's name (神州数码, 凤凰卫视): two companies, not one
            if any(lookalike._describes(word, COVERED[symbol], "") for symbol in symbols):
                continue  # its own industry (平安 + 保险)
            chosen.add(word)
        for word in sorted(chosen):
            frame = _ZH_FRAMES[len(cases) % len(_ZH_FRAMES)]
            cases.append((frame.format(n=name + word), name + word, frozenset(symbols)))
    return cases


CASES = _cases()


def test_the_property_test_covers_every_covered_name():
    names = {name for name in _covered_names()}
    assert {"茅台", "贵州茅台", "比亚迪", "紫金", "紫金矿业", "招银", "宁德", "中芯", "平安", "美的", "长电"} <= names
    assert len(CASES) == len(names) * _SUFFIXES_PER_NAME >= 300


def test_an_unknown_extension_of_a_covered_name_never_resolves_to_the_covered_company(pipeline):
    wrong = []
    for index, (query, _name, symbols) in enumerate(CASES):
        # the fuzzy passes see the question with the lookalike blanked out; a sample of 16 runs them (0.5 s each)
        found, trace = _resolve(pipeline, query, full=index % max(1, len(CASES) // 16) == 0)
        if found & symbols or not any(item.startswith(("lookalike_name:", "hk_only_name:")) for item in trace):
            wrong.append((query, sorted(found), trace[:3]))
    assert not wrong, f"{len(wrong)} of {len(CASES)}: {wrong[:10]}"


@pytest.mark.parametrize(
    ("query", "name"),
    [
        ("紫金黄金的股价", "紫金黄金"),
        ("紫金黄金能买吗", "紫金黄金"),
        ("比亚迪半导体的营收", "比亚迪半导体"),
        ("宁德新能源的净利润", "宁德新能源"),
        ("招银理财的净利润", "招银理财"),
        ("中芯南方营收多少", "中芯南方"),
        ("平安科技的营收", "平安科技"),
        ("紫金有色金属的市盈率", "紫金有色金属"),
        ("Zijin Gold share price", "Zijin Gold"),
        ("What is BYD Semiconductor's revenue?", "BYD Semiconductor"),
    ],
)
def test_the_reviewers_lookalikes_are_set_aside(pipeline, query, name):
    found, trace = _resolve(pipeline, query, full=True)
    assert not found & set(COVERED), (query, found, trace)
    assert any(item.startswith("lookalike_name:") and name in item for item in trace), trace


@pytest.mark.parametrize(
    ("query", "symbol"),
    [
        ("茅台能买吗", "600519.SH"),
        ("茅台和五粮液的市盈率", "600519.SH"),
        ("茅台五粮液哪个市值大", "000858.SZ"),
        ("比亚迪多少钱一股", "002594.SZ"),
        ("紫金有没有分红", "601899.SH"),
        ("平安存款规模多大", "000001.SZ"),
        ("招行今年业绩怎么样", "600036.SH"),
        ("宁德时代新能源科技股份有限公司的营收", "300750.SZ"),
        ("贵州茅台酒股份有限公司的净利润", "600519.SH"),
        ("中芯国际集成电路制造有限公司的营收", "688981.SH"),
        ("宁德最近涨了多少", "300750.SZ"),
        ("长电去年发电量", "600900.SH"),
        ("BYD share price", "002594.SZ"),
        ("What Is BYD Revenue", "002594.SZ"),
        ("Zijin Mining Group P/E", "601899.SH"),
    ],
)
def test_controls_keep_their_company(pipeline, query, symbol):
    found, trace = _resolve(pipeline, query, full=True)
    assert symbol in found, (query, found, trace)
    assert not any(item.startswith("lookalike_name:") for item in trace), trace


@pytest.fixture(scope="module")
def agent(shipped_offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    service = shipped_offline_service
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


@pytest.mark.parametrize(
    ("query", "forbidden", "said"),
    [
        ("紫金黄金的股价", ["29.79"], ["紫金黄金"]),
        ("Zijin Gold share price", ["29.79"], ["Zijin Gold"]),
        ("比亚迪半导体的营收", ["8039.65", "803.97"], ["比亚迪半导体"]),
        ("招银理财的净利润", ["1501.81"], ["招银理财"]),
    ],
)
def test_lookalikes_are_refused_without_the_covered_companys_figures(agent, query, forbidden, said):
    result = agent.chat(query)
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"], result["answer"]
    assert not any(text in result["answer"] for text in forbidden), result["answer"]
    assert all(text in result["answer"] for text in said), result["answer"]


def test_a_name_list_without_separators_is_two_companies(pipeline):
    found, trace = _resolve(pipeline, "茅台五粮液招行谁的市盈率最低", full=True)
    assert {"600519.SH", "000858.SZ", "600036.SH"} <= found, trace


def test_every_frame_is_exercised():
    assert {query.replace(name, "{n}") for query, name, _symbols in CASES} == set(_ZH_FRAMES)
