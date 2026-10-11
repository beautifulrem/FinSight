"""Round-17 rule M3 (round-13 review): a company absent from the entity master is never answered with the figures of
a covered company whose name it resembles.

Two sources of such names, both checked before entity resolution (``nlu/entity_resolver.py``):

* a covered company's *short* name followed by a business-line word ("美的置业", "中芯集成", "平安证券", "Midea Real
  Estate"; ``nlu/lookalike.py``), unless the extended name is the company's own name (中芯 + 国际) or the word names
  its own industry ("平安保险" is 中国平安, an insurer; "招行银行" is 招商银行);
* the Hong Kong-only equities of the HKEX list of securities (``data/runtime/hk_only_names.csv``: "中信国际电讯",
  "紫金黄金国际", "美的置业").

The property test crosses every short name of the covered companies with every business-line word in four frames
(and the English names with the English words), with controls; every Hong Kong-only name is checked too. End to end,
such a question is refused with the name and the covered company it resembles, and never states that company's
figures.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import pytest

from query_intelligence.nlu.lookalike import EN_SUFFIXES, ZH_SUFFIXES

ROOT = Path(__file__).resolve().parents[1]
# the covered companies (those with offline snapshot data)
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
_EN_FRAMES = ("{n} share price", "What is the revenue of {n}?")
_EN_WORDS = (
    "Real Estate", "Properties", "Securities", "Bank", "Insurance", "Life", "Holdings", "International", "Telecom",
    "Electronics", "Capital", "Land", "Development", "Leasing", "Trust", "Investment", "Asset Management", "Futures",
    "Logistics", "Healthcare", "Good Doctor", "Financial",
)  # fmt: skip
# an industry word of the company's own: a descriptor, not another company
_FAMILIES = {"保险": "保险人寿寿险财险", "银行": "银行", "证券": "证券"}


@pytest.fixture(scope="module")
def service(shipped_offline_service):
    return shipped_offline_service


@pytest.fixture(scope="module")
def pipeline(service):
    return service.nlu_pipeline


def _resolve(pipeline, text: str, *, full: bool = True) -> tuple[set[str], list[str]]:
    """The listed symbols and the trace the NLU front end gives ``text`` (normalizer, then exact matching, then the
    fuzzy passes when nothing matched exactly). With ``full=False`` only exact matching runs: the fuzzy passes see
    the question with the lookalike name blanked out (``_mask_lookalikes``), so they cannot reach it; a sample of
    every property test (eight evenly spaced cases; the fuzzy passes take about half a second a question) runs the
    full path."""
    normalized, normal_trace = pipeline.normalizer.normalize(text)
    resolver = pipeline.entity_resolver
    entities, _targets, trace = resolver.resolve_exact(normalized)
    if full and not [entity for entity in entities if entity.get("symbol")]:
        entities, _targets, trace = resolver.resolve(normalized)
    return {str(entity["symbol"]) for entity in entities if entity.get("symbol")}, [*normal_trace, *trace]


def _sampled(index: int, total: int, count: int = 8) -> bool:
    """Eight evenly spaced cases of ``total`` (the first included)."""
    return index % max(1, total // count) == 0


def _short_names() -> dict[str, list[str]]:
    """Short names (common / colloquial aliases) of the covered companies: name -> the symbols it names."""
    ids = {}
    with open(ROOT / "data/runtime/entity_master.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["symbol"] in COVERED and row["entity_type"] == "stock":
                ids[str(int(row["entity_id"]))] = row["symbol"]
    names: dict[str, list[str]] = {}
    with open(ROOT / "data/runtime/alias_table.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = ids.get(row["entity_id"])
            if symbol and row["alias_type"] in {"common_alias", "colloquial_alias"}:
                names.setdefault(row["alias_text"], []).append(symbol)
    return names


def _all_aliases() -> set[str]:
    with open(ROOT / "data/runtime/alias_table.csv", encoding="utf-8") as handle:
        return {row["normalized_alias"] for row in csv.DictReader(handle)}


def _describes(name: str, word: str, symbol: str) -> bool:
    """The word extends the short name into the company's own name, or names the company's own industry."""
    owner = COVERED[symbol]
    if owner.startswith(name + word) or (name + word) in owner:
        return True
    industry = {"601318.SH": "保险", "000001.SZ": "银行", "600036.SH": "银行", "601398.SH": "银行", "600030.SH": "证券"}
    family = next((key for key, members in _FAMILIES.items() if word in members), None)
    return family is not None and family == industry.get(symbol)


def _zh_cases() -> list[tuple[str, str, str, tuple[str, ...]]]:
    aliases = _all_aliases()
    cases = []
    for (name, symbols), word, frame in itertools.product(_short_names().items(), ZH_SUFFIXES, _ZH_FRAMES):
        if any(alias.startswith(name + word) for alias in aliases):
            continue  # a listed company of that name exists (平安银行, 长电科技, 中信证券): not a lookalike
        cases.append((frame.format(n=name + word), name, word, tuple(symbols)))
    return cases


ZH_CASES = _zh_cases()


def test_the_property_test_covers_every_short_name_and_word():
    names = {name for _query, name, _word, _symbols in ZH_CASES}
    assert {"美的", "中芯", "平安", "招行", "工行", "茅台", "紫金", "宁德", "长电"} <= names
    assert len(ZH_CASES) >= 1500


def test_a_short_name_with_a_business_line_word_never_names_the_covered_company(pipeline):
    wrong = []
    for index, (query, name, word, symbols) in enumerate(ZH_CASES):
        found, trace = _resolve(pipeline, query, full=_sampled(index, len(ZH_CASES)))
        own = [symbol for symbol in symbols if symbol in found]
        allowed = [symbol for symbol in symbols if _describes(name, word, symbol)]
        if allowed:
            if not set(allowed) & found:
                wrong.append((query, "descriptor lost", sorted(found)))
            continue
        if own or not any(item.startswith(("lookalike_name:", "hk_only_name:")) for item in trace):
            wrong.append((query, sorted(found), trace[:3]))
    assert not wrong, f"{len(wrong)} of {len(ZH_CASES)}: {wrong[:10]}"


@pytest.mark.parametrize(
    ("query", "symbol"),
    [
        ("美的的股价", "000333.SZ"),
        ("美的集团的股价", "000333.SZ"),
        ("中芯国际最新股价", "688981.SH"),
        ("平安保险的市盈率", "601318.SH"),
        ("平安银行的市净率", "000001.SZ"),
        ("招行银行的股价", "600036.SH"),
        ("比亚迪股份有限公司的营收", "002594.SZ"),
        ("宁德时代新能源科技的营收", "300750.SZ"),
        ("紫金矿业集团股份的市盈率", "601899.SH"),
        ("中信证券的营收", "600030.SH"),
        ("茅台的市盈率", "600519.SH"),
        ("Midea Group revenue", "000333.SZ"),
        ("Ping An Insurance P/E", "601318.SH"),
        ("ICBC Bank net profit", "601398.SH"),
        ("What is Midea's revenue?", "000333.SZ"),
    ],
)
def test_controls_keep_their_company(pipeline, query, symbol):
    found, trace = _resolve(pipeline, query)
    assert symbol in found, (query, found, trace)
    assert not any(item.startswith(("lookalike_name:", "hk_only_name:")) for item in trace)


def _en_cases() -> list[tuple[str, str, str]]:
    aliases = json.loads((ROOT / "data/synonym_dict.json").read_text(encoding="utf-8"))["alias"]
    cases = []
    for alias, target in aliases.items():
        symbol = next((symbol for symbol, name in COVERED.items() if name == target), None)
        if not symbol or not alias.isascii():
            continue
        for word, frame in itertools.product(_EN_WORDS, _EN_FRAMES):
            if alias.lower().endswith(word.lower()) or f"{alias} {word}".lower() in aliases:
                continue
            name = " ".join(part.capitalize() for part in alias.split()) if alias.islower() else alias
            cases.append((frame.format(n=f"{name} {word}"), word, symbol))
    return cases


EN_CASES = _en_cases()


def test_english_names_with_a_business_line_word_never_name_the_covered_company(pipeline):
    assert len(EN_CASES) >= 400 and len(EN_SUFFIXES) >= len(_EN_WORDS)
    own_industry = {
        "601318.SH": {"Insurance", "Life"},
        "000001.SZ": {"Bank"},
        "600036.SH": {"Bank"},
        "601398.SH": {"Bank"},
        "600030.SH": {"Securities"},
    }
    wrong = []
    for index, (query, word, symbol) in enumerate(EN_CASES):
        found, trace = _resolve(pipeline, query, full=_sampled(index, len(EN_CASES)))
        if word in own_industry.get(symbol, set()):
            if symbol not in found:
                wrong.append((query, "descriptor lost", sorted(found)))
            continue
        if symbol in found or not any(item.startswith("lookalike_name:") for item in trace):
            wrong.append((query, sorted(found), trace[:3]))
    assert not wrong, f"{len(wrong)} of {len(EN_CASES)}: {wrong[:10]}"


# ---- Hong Kong-only names (HKEX list of securities) ----


def _hk_rows() -> list[dict[str, str]]:
    from query_intelligence.data_loader import load_hk_only_names

    return load_hk_only_names()


def test_the_hong_kong_only_list_is_data_with_a_source():
    first = (ROOT / "data/runtime/hk_only_names.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "HKEX ListOfSecurities_c.xlsx" in first and "sha256" in first
    rows = _hk_rows()
    assert len(rows) > 2000
    names = {row["name_zh"] for row in rows}
    assert {"美的置业", "中信国际电讯", "紫金黄金国际", "海天国际", "平安好医生"} <= names
    # no A-share name and no H line of a dual-listed company is in it
    assert not names & {"美的集团", "中芯国际", "比亚迪股份", "招商银行", "中国平安"}
    with open(ROOT / "data/runtime/dual_listings.csv", encoding="utf-8") as handle:
        dual = {row["h_code"] for row in csv.DictReader(handle)}
    assert not dual & {row["h_code"] for row in rows}


def test_every_hong_kong_only_name_resolves_to_no_listed_security(pipeline):
    wrong = []
    rows = _hk_rows()
    for index, row in enumerate(rows):
        found, trace = _resolve(pipeline, f"{row['name_zh']}的股价是多少", full=_sampled(index, len(rows)))
        if found or not any(item.startswith("hk_only_name:") for item in trace):
            wrong.append((row["name_zh"], sorted(found)))
    assert not wrong, f"{len(wrong)}: {wrong[:10]}"


def test_a_hong_kong_only_name_followed_by_a_sector_word_is_not_the_company(pipeline):
    _found, trace = _resolve(pipeline, "中国电力行业的市盈率")
    assert not any(item.startswith("hk_only_name:") for item in trace)


def test_legal_form_fragments_are_not_names(pipeline):
    # the alias table has "限公司" as an official name of 国晟科技 (a cut legal name)
    found, _trace = _resolve(pipeline, "比亚迪股份有限公司的营收")
    assert found == {"002594.SZ"}


def test_the_price_tool_does_not_map_a_lookalike(pipeline):
    from query_intelligence.agent.tools.base import ToolFailure
    from query_intelligence.agent.tools.context import ToolContext

    class _Retrieval:  # the tool context only needs the NLU for name resolution
        pass

    context = ToolContext(nlu_pipeline=pipeline, retrieval_pipeline=_Retrieval())  # type: ignore[arg-type]
    for name in ("中信国际电讯", "美的置业", "中芯集成", "平安证券"):
        with pytest.raises(ToolFailure):
            context.resolve_target(name)
    assert context.resolve_target("美的集团").symbol == "000333.SZ"


# ---- end to end ----


@pytest.fixture(scope="module")
def agent(service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


@pytest.mark.parametrize(
    ("query", "forbidden", "said"),
    [
        ("美的置业的收盘价是多少", ["80.1", "13.77"], ["美的置业", "03990", "美的集团"]),
        ("中芯集成现在股价多少", ["111.99"], ["中芯集成", "中芯国际"]),
        ("平安证券去年营收多少", ["10505.06"], ["平安证券", "中国平安"]),
        ("What is the share price of Midea Real Estate?", ["80.1"], ["Midea Real Estate", "Midea Group"]),
        ("中信国际电讯的股价", ["中工国际"], ["中信国际电讯", "01883"]),
        ("What is BYD Electronic's revenue?", ["8039.65", "803.97"], ["BYD Electronic", "BYD"]),
    ],
)
def test_lookalike_companies_are_refused_with_their_name(agent, query, forbidden, said):
    result = agent.chat(query, session_id=f"r17-look-{abs(hash(query))}")
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"]
    assert not any(text in result["answer"] for text in forbidden), result["answer"]
    assert all(text in result["answer"] for text in said), result["answer"]


def test_a_lookalike_follow_up_does_not_inherit_the_session_company(agent):
    session = "r17-look-session"
    agent.chat("美的集团的市盈率", session_id=session)
    result = agent.chat("那美的置业呢", session_id=session)
    assert result["route"] == "refuse"
    assert "13.77" not in result["answer"] and "80.1" not in result["answer"]


def test_a_lookalike_next_to_the_covered_company_answers_the_covered_part(agent):
    result = agent.chat("美的集团和美的置业的市盈率分别多少", session_id="r17-look-mixed")
    assert result["route"] != "refuse" and "13.77" in result["answer"]
    assert "美的置业" in result["answer"] and "不是同一家公司" in result["answer"]
