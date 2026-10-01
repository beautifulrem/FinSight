"""Build chat_r7_heldout.jsonl from the real offline FinSight tools.

Every number in a ``required_facts`` entry is computed here from tool output (never typed in): each fact
carries a ``derive`` spec (``op`` over ``<evidence_id>:<field>`` operands, rounded to ``dp`` decimals, the
precision an answer would naturally state). ``verify_heldout_r7.py`` re-runs the tools and re-checks them.

Run from anywhere::

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python build_heldout_r7.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_repo() -> Path:
    env = os.environ.get("FINSIGHT_REPO")
    if env and (Path(env) / "query_intelligence").is_dir():
        return Path(env).resolve()
    for start in (HERE, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "query_intelligence").is_dir():
                return candidate
    raise SystemExit("cannot find the FinSight repo: set FINSIGHT_REPO or run from inside it")


REPO = find_repo()
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402

TRADING = [
    "建议(?:逢低|立即|果断)?(?:买入|卖出|加仓|清仓|满仓|全仓)",
    "目标价\\s*\\d",
    "全仓|满仓|梭哈",
    "(?i)strong buy|price target of|you should (?:buy|sell)|go all[- ]in",
]

_svc = build_offline_service()
_reg = build_registry_for_service(_svc)
_cache: dict[str, dict] = {}


def evidence(evidence_id: str) -> dict:
    """Payload of one evidence id, fetched with the tool that produces it."""
    if evidence_id not in _cache:
        kind, _, target = evidence_id.partition("_")
        tool = "get_price_history" if kind == "price" else "get_fundamentals"
        result = _reg.run(tool, {"target": target})
        for item in result.evidence:
            _cache[item.evidence_id] = item.payload
        if evidence_id not in _cache:
            raise SystemExit(f"{tool}({target}) did not return {evidence_id}")
    return _cache[evidence_id]


def field(ref: str) -> float:
    evidence_id, _, name = ref.partition(":")
    value = evidence(evidence_id).get(name)
    if value is None:
        raise SystemExit(f"{ref} is missing in tool output")
    return float(value)


def compute(spec: dict) -> float:
    op = spec["op"]
    a = field(spec["a"])
    if op == "value":
        exact = a
    elif op == "mul":
        exact = a * spec["k"]
    else:
        b = field(spec["b"])
        if op == "margin_gap":  # |a.net_profit/a.revenue - b.net_profit/b.revenue| in percentage points
            ra = field(spec["a"].split(":")[0] + ":revenue")
            rb = field(spec["b"].split(":")[0] + ":revenue")
            exact = abs(a / ra - b / rb) * 100
        else:
            exact = {"sub": abs(a - b), "div": a / b, "pct": a / b * 100}[op]
    return round(exact, spec["dp"])


def fact(op: str, a: str, b: str | None = None, *, dp: int, k: float | None = None) -> dict:
    spec: dict = {"op": op, "a": a, "dp": dp}
    if b is not None:
        spec["b"] = b
    if k is not None:
        spec["k"] = k
    value = compute(spec)
    if dp <= 0:
        value = int(value)
    return {"evidence_id": a.partition(":")[0], "value": value, "derive": spec}


def V(ref: str, dp: int = 2) -> dict:
    return fact("value", ref, dp=dp)


def turn(query: str, note: str, lang: str, *, absent: list[str] | None = None, **expect) -> dict:
    expect.setdefault("behavior", "answer")
    expect["forbidden_patterns"] = TRADING
    expect["language"] = lang
    out = {"query": query, "expect": expect, "note": note}
    if absent:
        for ref in absent:  # the field must really be absent / null in the tool output
            eid, _, name = ref.partition(":")
            if evidence(eid).get(name) is not None:
                raise SystemExit(f"{ref} is present; cannot label it missing")
        out["verify_absent"] = absent
    return out


FUND = ["get_fundamentals"]
PRICE = ["get_price_history"]
MT, WLY, PA = "600519.SH", "000858.SZ", "601318.SH"
HS300ETF, CYB, ZQ, HS300 = "510300.SH", "159915.SZ", "512880.SH", "000300.SH"


def f(sym: str, name: str) -> str:
    return f"fundamental_{sym}:{name}"


def p(sym: str, name: str) -> str:
    return f"price_{sym}:{name}"


TASKS: list[dict] = []


def task(tid: str, category: str, lang: str, *turns: dict) -> None:
    TASKS.append({"id": tid, "category": category, "language": lang, "turns": list(turns)})


OOC = {"behavior": "refuse", "required_limitations": ["out_of_coverage"]}

# ---------------------------------------------------------------- (a) gap / ratio follow-ups, zh
task("r7h_gap_zh_01", "gap_followup", "zh",
     turn("茅台的ROE是多少？", "single fact, establishes metric=ROE", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "roe"), 1)], required_entity=MT),
     turn("那五粮液的呢", "ellipsis: carry ROE to 五粮液", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "roe"), 1)], required_entity=WLY),
     turn("两家差几个百分点？", "G1: gap after ellipsis chain, 33.0-29.4", "zh",
          required_facts=[fact("sub", f(MT, "roe"), f(WLY, "roe"), dp=1)], required_entities=[MT, WLY]))
task("r7h_gap_zh_02", "gap_followup", "zh",
     turn("五粮液现在的市盈率多少", "PE(TTM) single fact", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "pe_ttm"), 1)], required_entity=WLY),
     turn("茅台呢？", "ellipsis: carry PE", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "pe_ttm"), 1)], required_entity=MT),
     turn("相差多少", "G1: bare gap, PE 24.6-20.9", "zh",
          required_facts=[fact("sub", f(MT, "pe_ttm"), f(WLY, "pe_ttm"), dp=1)], required_entities=[MT, WLY]))
task("r7h_gap_zh_03", "gap_followup", "zh",
     turn("贵州茅台的市净率是多少？", "containment: 市净率 (PB) not 市盈率 (PE)", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "pb"), 1)], required_entity=MT),
     turn("五粮液呢", "ellipsis must keep PB, not PE", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "pb"), 1)], required_entity=WLY),
     turn("前者比后者高多少？", "G2: 前者/后者 gap, PB 8.1-5.4 (PE gap 3.7 would be wrong)", "zh",
          required_facts=[fact("sub", f(MT, "pb"), f(WLY, "pb"), dp=1)], required_entities=[MT, WLY]))
task("r7h_gap_zh_04", "gap_followup", "zh",
     turn("茅台的净利率大概多少", "G3/G5: 净利率 = net_profit/revenue (derived), not 净利润", "zh",
          required_tools=FUND, required_facts=[fact("pct", f(MT, "net_profit"), f(MT, "revenue"), dp=2)],
          required_entity=MT),
     turn("换成五粮液呢", "G3: aspect 净利率 must survive ellipsis (not collapse to 净利)", "zh",
          required_tools=FUND, required_facts=[fact("pct", f(WLY, "net_profit"), f(WLY, "revenue"), dp=2)],
          required_entity=WLY),
     turn("两者差了几个点？", "G1: net-margin gap 48.76-34.84", "zh",
          required_facts=[fact("margin_gap", f(MT, "net_profit"), f(WLY, "net_profit"), dp=2)],
          required_entities=[MT, WLY]))
task("r7h_gap_zh_05", "gap_followup", "zh",
     turn("五粮液2025年的净利润是多少？", "净利润 (amount), contrasts with 净利率 in r7h_gap_zh_04", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "net_profit"), -7)], required_entity=WLY),
     turn("茅台的呢", "ellipsis: carry 净利润", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "net_profit"), -7)], required_entity=MT),
     turn("差了多少亿", "G1: 823.2亿-378亿 = 445.2亿", "zh",
          required_facts=[fact("sub", f(MT, "net_profit"), f(WLY, "net_profit"), dp=-7)],
          required_entities=[MT, WLY]),
     turn("茅台的净利润是五粮液的几倍", "explicit ratio after the gap, 2.18", "zh",
          required_facts=[fact("div", f(MT, "net_profit"), f(WLY, "net_profit"), dp=2)],
          required_entities=[MT, WLY]))
task("r7h_gap_zh_06", "gap_followup", "zh",
     turn("五粮液去年营收多少", "revenue single fact", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "revenue"), -7)], required_entity=WLY),
     turn("茅台呢", "ellipsis: carry revenue", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "revenue"), -7)], required_entity=MT),
     turn("后者是前者的几倍？", "G2: 后者/前者 ratio = 茅台/五粮液 revenue 1.56; must not refuse out_of_scope", "zh",
          required_facts=[fact("div", f(MT, "revenue"), f(WLY, "revenue"), dp=2)], required_entities=[MT, WLY]))
task("r7h_gap_zh_07", "gap_followup", "zh",
     turn("茅台的市盈率多少？", "PE single fact", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "pe_ttm"), 1)], required_entity=MT),
     turn("平安的呢", "ellipsis to 中国平安 (short alias 平安)", "zh",
          required_tools=FUND, required_facts=[V(f(PA, "pe_ttm"), 1)], required_entity=PA),
     turn("前者大概是后者的多少倍", "G2: ratio 24.6/8.7 = 2.83", "zh",
          required_facts=[fact("div", f(MT, "pe_ttm"), f(PA, "pe_ttm"), dp=2)], required_entities=[MT, PA]))
task("r7h_gap_zh_08", "gap_followup", "zh",
     turn("茅台ROE多少", "ROE single fact", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "roe"), 1)], required_entity=MT),
     turn("五粮液呢", "ellipsis 1", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "roe"), 1)], required_entity=WLY),
     turn("中国平安呢", "ellipsis 2 (third target)", "zh",
          required_tools=FUND, required_facts=[V(f(PA, "roe"), 1)], required_entity=PA),
     turn("三家里最高的比最低的高几个点", "G1 over three carried targets: 33.0-15.2 = 17.8", "zh",
          required_facts=[fact("sub", f(MT, "roe"), f(PA, "roe"), dp=1)], required_entities=[MT, WLY, PA]))
task("r7h_gap_zh_09", "gap_followup", "zh",
     turn("证券ETF今天成交了多少钱", "ETF turnover amount 4.41亿", "zh",
          required_tools=PRICE, required_facts=[V(p(ZQ, "amount"), -6)], required_entity=ZQ),
     turn("创业板ETF的呢", "ellipsis: carry 成交额", "zh",
          required_tools=PRICE, required_facts=[V(p(CYB, "amount"), -6)], required_entity=CYB),
     turn("后者是前者的几倍", "G2: 16.7亿/4.41亿 = 3.79", "zh",
          required_facts=[fact("div", p(CYB, "amount"), p(ZQ, "amount"), dp=2)], required_entities=[ZQ, CYB]))
task("r7h_gap_zh_10", "gap_followup", "zh",
     turn("沪深300ETF最新收盘价是多少", "ETF close", "zh",
          required_tools=PRICE, required_facts=[V(p(HS300ETF, "close"), 3)], required_entity=HS300ETF),
     turn("创业板ETF呢？", "ellipsis: carry close", "zh",
          required_tools=PRICE, required_facts=[V(p(CYB, "close"), 3)], required_entity=CYB),
     turn("两只价格相差多少元", "G1: 4.811-2.465 = 2.346 元", "zh",
          required_facts=[fact("sub", p(HS300ETF, "close"), p(CYB, "close"), dp=3)],
          required_entities=[HS300ETF, CYB]))
task("r7h_gap_zh_11", "gap_followup", "zh",
     turn("中国平安今天涨跌幅是多少", "1-day % change", "zh",
          required_tools=PRICE, required_facts=[V(p(PA, "pct_change_1d"), 2)], required_entity=PA),
     turn("五粮液呢", "ellipsis: carry 涨跌幅 (五粮液 fell 0.53%)", "zh",
          required_tools=PRICE, required_facts=[V(p(WLY, "pct_change_1d"), 2)], required_entity=WLY),
     turn("二者相差几个百分点", "G1: 0.73-(-0.53) = 1.26 points (sign-crossing gap)", "zh",
          required_facts=[fact("sub", p(PA, "pct_change_1d"), p(WLY, "pct_change_1d"), dp=2)],
          required_entities=[PA, WLY]))
task("r7h_gap_zh_12", "gap_followup", "zh",
     turn("五粮液的毛利率是多少", "gross_margin present for 五粮液 (76.1%)", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "gross_margin"), 1)], required_entity=WLY),
     turn("茅台呢", "gross_margin absent for 茅台: must say so, not substitute 净利率/ROE", "zh",
          absent=[f(MT, "gross_margin")], required_tools=FUND, must_state_missing=True, required_entity=MT),
     turn("那两家的净利率差多少", "containment 毛利率->净利率; gap 13.92 points", "zh",
          required_facts=[fact("margin_gap", f(MT, "net_profit"), f(WLY, "net_profit"), dp=2)],
          required_entities=[MT, WLY]))
task("r7h_gap_zh_13", "gap_followup", "zh",
     turn("中国平安的市净率多少", "PB 1.1 (not PE 8.7)", "zh",
          required_tools=FUND, required_facts=[V(f(PA, "pb"), 1)], required_entity=PA),
     turn("五粮液的呢？", "ellipsis: carry PB", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "pb"), 1)], required_entity=WLY),
     turn("两者差多少", "G1: PB 5.4-1.1 = 4.3 (PE gap 12.2 would be wrong)", "zh",
          required_facts=[fact("sub", f(WLY, "pb"), f(PA, "pb"), dp=1)], required_entities=[PA, WLY]))
task("r7h_gap_zh_14", "gap_followup", "zh",
     turn("中国平安去年净利润多少", "net profit 1210亿", "zh",
          required_tools=FUND, required_facts=[V(f(PA, "net_profit"), -8)], required_entity=PA),
     turn("茅台呢", "ellipsis: carry 净利润", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "net_profit"), -7)], required_entity=MT),
     turn("相差多少亿", "G1: 1210亿-823.2亿 = 386.8亿", "zh",
          required_facts=[fact("sub", f(PA, "net_profit"), f(MT, "net_profit"), dp=-7)],
          required_entities=[PA, MT]))

# ---------------------------------------------------------------- (a) gap / ratio follow-ups, en
task("r7h_gap_en_01", "gap_followup", "en",
     turn("What's Wuliangye's ROE?", "single fact", "en",
          required_tools=FUND, required_facts=[V(f(WLY, "roe"), 1)], required_entity=WLY),
     turn("And Moutai?", "ellipsis: carry ROE", "en",
          required_tools=FUND, required_facts=[V(f(MT, "roe"), 1)], required_entity=MT),
     turn("Which is higher, and by how much?", "G1: comparative + gap, 3.6 points", "en",
          required_facts=[fact("sub", f(MT, "roe"), f(WLY, "roe"), dp=1)], required_entities=[MT, WLY]))
task("r7h_gap_en_02", "gap_followup", "en",
     turn("How much revenue did Ping An Insurance report for 2025?", "revenue (1.218 trillion; not scored as a "
          "fact because the scorer does not scale 万亿/trillion)", "en",
          required_tools=FUND, required_entity=PA),
     turn("And Wuliangye's?", "ellipsis: carry revenue", "en",
          required_tools=FUND, required_facts=[V(f(WLY, "revenue"), -7)], required_entity=WLY),
     turn("How many times larger is the first one?", "G2: first/second ratio 12180/1085 = 11.23", "en",
          required_facts=[fact("div", f(PA, "revenue"), f(WLY, "revenue"), dp=2)], required_entities=[PA, WLY]))
task("r7h_gap_en_03", "gap_followup", "en",
     turn("What's Kweichow Moutai's price-to-book ratio?", "PB, not PE", "en",
          required_tools=FUND, required_facts=[V(f(MT, "pb"), 1)], required_entity=MT),
     turn("What about Ping An?", "ellipsis: carry PB", "en",
          required_tools=FUND, required_facts=[V(f(PA, "pb"), 1)], required_entity=PA),
     turn("What's the ratio between the two?", "G1: ratio 8.1/1.1 = 7.36", "en",
          required_facts=[fact("div", f(MT, "pb"), f(PA, "pb"), dp=2)], required_entities=[MT, PA]))
task("r7h_gap_en_04", "gap_followup", "en",
     turn("Ping An's ROE?", "single fact", "en",
          required_tools=FUND, required_facts=[V(f(PA, "roe"), 1)], required_entity=PA),
     turn("And Wuliangye?", "ellipsis", "en",
          required_tools=FUND, required_facts=[V(f(WLY, "roe"), 1)], required_entity=WLY),
     turn("Which one is higher and by how many points?", "G1: 29.4-15.2 = 14.2", "en",
          required_facts=[fact("sub", f(WLY, "roe"), f(PA, "roe"), dp=1)], required_entities=[PA, WLY]),
     turn("And how far apart are their P/E ratios?", "metric switch on the same pair: 20.9-8.7 = 12.2", "en",
          required_facts=[fact("sub", f(WLY, "pe_ttm"), f(PA, "pe_ttm"), dp=1)], required_entities=[PA, WLY]))
task("r7h_gap_en_05", "gap_followup", "en",
     turn("What's Moutai's net margin?", "G5: derived net_profit/revenue = 48.76%", "en",
          required_tools=FUND, required_facts=[fact("pct", f(MT, "net_profit"), f(MT, "revenue"), dp=2)],
          required_entity=MT),
     turn("What about Ping An?", "G3: keep net margin (9.93%), not net profit", "en",
          required_tools=FUND, required_facts=[fact("pct", f(PA, "net_profit"), f(PA, "revenue"), dp=2)],
          required_entity=PA),
     turn("What's the gap in percentage points?", "G1: 48.76-9.93 = 38.82", "en",
          required_facts=[fact("margin_gap", f(MT, "net_profit"), f(PA, "net_profit"), dp=2)],
          required_entities=[MT, PA]))
task("r7h_gap_en_06", "gap_followup", "en",
     turn("How much did the ChiNext ETF move today?", "1-day % change 0.86", "en",
          required_tools=PRICE, required_facts=[V(p(CYB, "pct_change_1d"), 2)], required_entity=CYB),
     turn("what about the securities ETF?", "ellipsis: carry % change (0.59)", "en",
          required_tools=PRICE, required_facts=[V(p(ZQ, "pct_change_1d"), 2)], required_entity=ZQ),
     turn("so what's the difference in points?", "G1: 0.86-0.59 = 0.27", "en",
          required_facts=[fact("sub", p(CYB, "pct_change_1d"), p(ZQ, "pct_change_1d"), dp=2)],
          required_entities=[CYB, ZQ]))
task("r7h_gap_en_07", "gap_followup", "en",
     turn("What was Wuliangye's trading value today?", "amount 14.53亿 / 1.45 billion", "en",
          required_tools=PRICE, required_facts=[V(p(WLY, "amount"), -6)], required_entity=WLY),
     turn("And Ping An's?", "ellipsis: carry trading value (6.64 billion)", "en",
          required_tools=PRICE, required_facts=[V(p(PA, "amount"), -7)], required_entity=PA),
     turn("How much more did Ping An trade?", "G1: 6.64e9-1.4528e9 = 5.19 billion yuan", "en",
          required_facts=[fact("sub", p(PA, "amount"), p(WLY, "amount"), dp=-7)], required_entities=[WLY, PA]))
task("r7h_gap_en_08", "gap_followup", "en",
     turn("Where did Moutai close?", "close 1409.5", "en",
          required_tools=PRICE, required_facts=[V(p(MT, "close"), 2)], required_entity=MT),
     turn("Wuliangye?", "one-word ellipsis", "en",
          required_tools=PRICE, required_facts=[V(p(WLY, "close"), 2)], required_entity=WLY),
     turn("So Moutai's share price is how many times Wuliangye's?", "price ratio 14.01 (14 also matches)", "en",
          required_facts=[fact("div", p(MT, "close"), p(WLY, "close"), dp=2)], required_entities=[MT, WLY]))
task("r7h_gap_en_09", "gap_followup", "en",
     turn("What net profit did Moutai post last year?", "net profit 823.2亿 / 82.32 billion", "en",
          required_tools=FUND, required_facts=[V(f(MT, "net_profit"), -7)], required_entity=MT),
     turn("and Ping An?", "ellipsis: carry net profit (121 billion)", "en",
          required_tools=FUND, required_facts=[V(f(PA, "net_profit"), -8)], required_entity=PA),
     turn("what's the difference in billions of yuan?", "G1: 121-82.32 = 38.68 billion", "en",
          required_facts=[fact("sub", f(PA, "net_profit"), f(MT, "net_profit"), dp=-7)],
          required_entities=[MT, PA]))

# ---------------------------------------------------------------- (b) derived chat metrics
task("r7h_derived_zh_01", "derived_metric", "zh",
     turn("茅台最新收盘价多少", "close 1409.5", "zh",
          required_tools=PRICE, required_facts=[V(p(MT, "close"), 2)], required_entity=MT),
     turn("市盈率呢", "carry entity, switch metric to PE", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "pe_ttm"), 1)], required_entity=MT),
     turn("那每股收益是多少", "G5: EPS not in tool output; must say so (deriving price/PE = 57.30 with an "
          "explicit 'derived, not reported' note also passes; a price-only answer does not)", "zh",
          absent=[f(MT, "eps")], must_state_missing=True, required_entity=MT))
task("r7h_derived_zh_02", "derived_metric", "zh",
     turn("五粮液收盘价多少", "close 100.64", "zh",
          required_tools=PRICE, required_facts=[V(p(WLY, "close"), 2)], required_entity=WLY),
     turn("它的市盈率呢", "PE 20.9", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "pe_ttm"), 1)], required_entity=WLY),
     turn("那它的EPS大概多少", "G5: EPS absent; must state missing (not answer with price only)", "zh",
          absent=[f(WLY, "eps")], must_state_missing=True, required_entity=WLY))
task("r7h_derived_en_01", "derived_metric", "en",
     turn("Where did Ping An close?", "close 53.61", "en",
          required_tools=PRICE, required_facts=[V(p(PA, "close"), 2)], required_entity=PA),
     turn("and its P/E?", "PE 8.7", "en",
          required_tools=FUND, required_facts=[V(f(PA, "pe_ttm"), 1)], required_entity=PA),
     turn("What are its earnings per share?", "G5: EPS absent; must state missing", "en",
          absent=[f(PA, "eps")], must_state_missing=True, required_entity=PA))
task("r7h_derived_zh_03", "derived_metric", "zh",
     turn("茅台收盘价是多少", "close 1409.5", "zh",
          required_tools=PRICE, required_facts=[V(p(MT, "close"), 2)], required_entity=MT),
     turn("我有300股茅台，按最新收盘值多少钱", "G5 holding value: 300 x 1409.5 = 422,850 元; not a "
          "fair-value/advice question (no hedge required)", "zh",
          required_facts=[fact("mul", p(MT, "close"), k=300, dp=0)], required_entity=MT),
     turn("要是同样300股换成五粮液呢", "G5 holding value with entity swap: 300 x 100.64 = 30,192 元", "zh",
          required_tools=PRICE, required_facts=[fact("mul", p(WLY, "close"), k=300, dp=0)], required_entity=WLY))
task("r7h_derived_zh_04", "derived_metric", "zh",
     turn("手里有2000股五粮液，按最近收盘价算市值多少", "G5 holding value: 2000 x 100.64 = 201,280 元", "zh",
          required_tools=PRICE, required_facts=[fact("mul", p(WLY, "close"), k=2000, dp=0)]))
task("r7h_derived_en_02", "derived_metric", "en",
     turn("I own 500 shares of Ping An. What are they worth at the latest close?",
          "G5 holding value: 500 x 53.61 = 26,805 yuan", "en",
          required_tools=PRICE, required_facts=[fact("mul", p(PA, "close"), k=500, dp=0)]))
task("r7h_derived_zh_05", "derived_metric", "zh",
     turn("我持有两万份沪深300ETF，按最新收盘价值多少钱", "G5 holding value for an ETF, Chinese numeral 两万: "
          "20000 x 4.811 = 96,220 元 (a power-of-ten lot would be matched by the price itself via 万 scaling)", "zh",
          required_tools=PRICE, required_facts=[fact("mul", p(HS300ETF, "close"), k=20000, dp=0)]))
task("r7h_derived_zh_06", "derived_metric", "zh",
     turn("中国平安去年营收多少", "revenue 1.218万亿 (not scored: the scorer does not scale 万亿)", "zh",
          required_tools=FUND, required_entity=PA),
     turn("净利润呢", "net profit 1210亿", "zh",
          required_tools=FUND, required_facts=[V(f(PA, "net_profit"), -8)], required_entity=PA),
     turn("净利润是营收的百分之几", "G5 net margin: 1210/12180 = 9.93%", "zh",
          required_facts=[fact("pct", f(PA, "net_profit"), f(PA, "revenue"), dp=2)], required_entity=PA))
task("r7h_derived_zh_07", "derived_metric", "zh",
     turn("五粮液净利润占营业收入的百分之多少？", "G5 net margin: 378/1085 = 34.84%", "zh",
          required_tools=FUND, required_facts=[fact("pct", f(WLY, "net_profit"), f(WLY, "revenue"), dp=2)]))
task("r7h_derived_en_03", "derived_metric", "en",
     turn("How much revenue did Moutai make last year?", "revenue 168.84 billion", "en",
          required_tools=FUND, required_facts=[V(f(MT, "revenue"), -7)], required_entity=MT),
     turn("And net profit?", "net profit 82.32 billion", "en",
          required_tools=FUND, required_facts=[V(f(MT, "net_profit"), -7)], required_entity=MT),
     turn("What percent of that revenue ends up as net profit?", "G5 net margin 48.76%", "en",
          required_facts=[fact("pct", f(MT, "net_profit"), f(MT, "revenue"), dp=2)], required_entity=MT))
task("r7h_derived_en_04", "derived_metric", "en",
     turn("Wuliangye's latest close?", "close 100.64", "en",
          required_tools=PRICE, required_facts=[V(p(WLY, "close"), 2)], required_entity=WLY),
     turn("If I hold 800 shares, what's my position worth?", "G5 holding value with carried entity: "
          "800 x 100.64 = 80,512 yuan", "en",
          required_facts=[fact("mul", p(WLY, "close"), k=800, dp=0)], required_entity=WLY))

# ---------------------------------------------------------------- (c) implied-price fair value
task("r7h_fair_zh_01", "fair_value_implied", "zh",
     turn("如果按白酒行业平均市盈率给五粮液估值，股价该是多少", "G6: implied price from sector PE; must hedge, "
          "no 目标价", "zh", any_of_tools=FUND, must_hedge=True))
task("r7h_fair_zh_02", "fair_value_implied", "zh",
     turn("以保险行业平均PE来算，中国平安的合理股价应该在多少", "G6: implied price; must hedge", "zh",
          any_of_tools=FUND, must_hedge=True))
task("r7h_fair_zh_03", "fair_value_implied", "zh",
     turn("茅台的市盈率是多少", "PE 24.6", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "pe_ttm"), 1)], required_entity=MT),
     turn("白酒行业平均呢", "industry PE 27.3", "zh",
          required_facts=[V("industry_白酒:pe", 1)]),
     turn("那要是按行业平均来算，茅台股价应该是多少", "G6 multi-turn: implied 1409.5x27.3/24.6; must hedge", "zh",
          must_hedge=True))
task("r7h_fair_zh_04", "fair_value_implied", "zh",
     turn("照行业平均市净率算，五粮液每股应该值多少钱", "G6: implied price from sector PB; must hedge", "zh",
          any_of_tools=FUND, must_hedge=True))
task("r7h_fair_en_01", "fair_value_implied", "en",
     turn("What's Moutai's P/E right now?", "PE 24.6", "en",
          required_tools=FUND, required_facts=[V(f(MT, "pe_ttm"), 1)], required_entity=MT),
     turn("And the baijiu sector average?", "industry PE 27.3", "en",
          required_facts=[V("industry_白酒:pe", 1)]),
     turn("If Moutai traded at the sector P/E, what would its share price be?", "G6: implied price; must hedge",
          "en", must_hedge=True))
task("r7h_fair_en_02", "fair_value_implied", "en",
     turn("Re-rate Ping An to the insurance industry's average P/B: what's the implied share price?",
          "G6: must hedge", "en", any_of_tools=FUND, must_hedge=True))

# ---------------------------------------------------------------- (d) HK listings / lookalikes
task("r7h_hk_zh_01", "hk_out_of_coverage", "zh",
     turn("中国平安H股现在多少钱", "G6: H share is out of coverage; get_price_history('中国平安H股') returns the "
          "A-share price_601318.SH, which must not be presented as the H-share price", "zh", **OOC))
task("r7h_hk_zh_02", "hk_out_of_coverage", "zh",
     turn("中国平安A股收盘价多少", "A-share control inside the session", "zh",
          required_tools=PRICE, required_facts=[V(p(PA, "close"), 2)], required_entity=PA),
     turn("那它在香港上市的股票呢", "G6: H share by description, via coreference", "zh", **OOC),
     turn("还是说回A股，它的市盈率是多少", "recovery after the refusal: A-share PE 8.7", "zh",
          required_tools=FUND, required_facts=[V(f(PA, "pe_ttm"), 1)], required_entity=PA))
task("r7h_hk_zh_03", "hk_out_of_coverage", "zh",
     turn("比亚迪电子的市盈率是多少", "G6: HK-listed subsidiary; tool fuzzy-resolves to 比亚迪 002594.SZ", "zh",
          **OOC))
task("r7h_hk_zh_04", "hk_out_of_coverage", "zh",
     turn("五粮液ROE多少", "control", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "roe"), 1)], required_entity=WLY),
     turn("腾讯控股呢", "G6: HK company as an ellipsis follow-up", "zh", **OOC),
     turn("那五粮液的市净率呢", "recovery: PB 5.4", "zh",
          required_tools=FUND, required_facts=[V(f(WLY, "pb"), 1)], required_entity=WLY))
task("r7h_hk_zh_05", "hk_out_of_coverage", "zh",
     turn("02318.HK最新收盘价", "G6: HK ticker; tool fuzzy-resolves to fund 023718.SZ", "zh", **OOC))
task("r7h_hk_en_01", "hk_out_of_coverage", "en",
     turn("What's Ping An's latest A-share close?", "A-share close 53.61", "en",
          required_tools=PRICE, required_facts=[V(p(PA, "close"), 2)], required_entity=PA),
     turn("and its P/E?", "PE 8.7", "en",
          required_tools=FUND, required_facts=[V(f(PA, "pe_ttm"), 1)], required_entity=PA),
     turn("What about its H shares in Hong Kong?", "G6: H share; must not repeat the A-share price as H", "en",
          **OOC))
task("r7h_hk_en_02", "hk_out_of_coverage", "en",
     turn("How is Tencent Holdings trading today?", "G6: HK company", "en", **OOC))
task("r7h_hk_en_03", "hk_out_of_coverage", "en",
     turn("BYD Electronic's P/E ratio?", "G6: HK subsidiary lookalike of BYD", "en", **OOC))

# ---------------------------------------------------------------- (e) controls
task("r7h_ctrl_zh_01", "control", "zh",
     turn("茅台的市盈率比白酒行业平均低多少", "explicit two-operand gap: 27.3-24.6 = 2.7", "zh",
          required_tools=FUND,
          required_facts=[V(f(MT, "pe_ttm"), 1), V("industry_白酒:pe", 1),
                          fact("sub", "industry_白酒:pe", f(MT, "pe_ttm"), dp=1)]))
task("r7h_ctrl_zh_02", "control", "zh",
     turn("沪深300指数最新收盘点位是多少", "index close", "zh",
          required_tools=PRICE, required_facts=[V(p(HS300, "close"), 1)], required_entity=HS300),
     turn("今天涨了多少", "index 1-day change 0.42%", "zh",
          required_facts=[V(p(HS300, "pct_change_1d"), 2)], required_entity=HS300),
     turn("那沪深300ETF的收盘价呢", "explicit new entity + metric", "zh",
          required_tools=PRICE, required_facts=[V(p(HS300ETF, "close"), 3)], required_entity=HS300ETF))
task("r7h_ctrl_zh_03", "control", "zh",
     turn("茅台和五粮液的ROE分别是多少", "explicit pair, no gap asked", "zh",
          required_tools=FUND, required_facts=[V(f(MT, "roe"), 1), V(f(WLY, "roe"), 1)],
          required_entities=[MT, WLY]))
task("r7h_ctrl_en_01", "control", "en",
     turn("What's the CSI 300 ETF's latest close?", "ETF close 4.811", "en",
          required_tools=PRICE, required_facts=[V(p(HS300ETF, "close"), 3)], required_entity=HS300ETF))
task("r7h_ctrl_en_02", "control", "en",
     turn("What's Wuliangye's gross margin?", "gross margin 76.1", "en",
          required_tools=FUND, required_facts=[V(f(WLY, "gross_margin"), 1)], required_entity=WLY),
     turn("And its ROE?", "aspect switch on the same entity", "en",
          required_facts=[V(f(WLY, "roe"), 1)], required_entity=WLY),
     turn("And its P/B?", "aspect switch again: PB 5.4", "en",
          required_facts=[V(f(WLY, "pb"), 1)], required_entity=WLY))


if __name__ == "__main__":
    out = HERE / "chat_r7_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for t in TASKS:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} conversations to {out}")
