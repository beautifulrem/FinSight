"""Build chat_r12_heldout.jsonl and claims_r12_heldout.jsonl from the real offline FinSight tools.

Round-12 independent held-out slice, written by the round-12 reviewer at bb99360 before any round-16 fix. The
engineers do not open it; it is run once before (here, first_run/) and once after their fixes.

It uses the SHIPPED offline default: the snapshot extension (data/snapshot/structured_data_ext.json) AND the
corrections layer (data/snapshot/structured_data_fix.json). Run the chat slice with
``QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`` (the runner is pinned to the v1 seed otherwise).

Every number in a chat ``required_facts`` entry is computed here from tool output (never typed in): each fact carries
a ``derive`` spec over ``<evidence_id>:<field>`` operands (dotted fields reach nested payloads, e.g.
``price_300750.SZ:range_52w.high``), rounded to ``dp`` decimals. Every claim's expected verdict comes from the
predicates in ``claim(...)`` (all parts true -> supported, all false -> contradicted, mixed -> partially_supported).
``verify_heldout_r12.py`` re-runs the tools and re-checks both files with its own implementation of every op.

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python build_heldout_r12.py
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

_svc = build_offline_service(snapshot_ext=True, snapshot_fix=True)
_reg = build_registry_for_service(_svc)
_cache: dict[str, dict] = {}
_industry_returned: dict[str, list[str]] = {}


def evidence(evidence_id: str) -> dict:
    if evidence_id not in _cache:
        kind, _, target = evidence_id.partition("_")
        tool = "get_price_history" if kind == "price" else "get_fundamentals"
        result = _reg.run(tool, {"target": target})
        for item in result.evidence:
            _cache[item.evidence_id] = item.payload
        if tool == "get_fundamentals":
            _industry_returned[target] = [i.evidence_id for i in result.evidence if i.evidence_id.startswith("industry_")]
        if evidence_id not in _cache:
            raise SystemExit(f"{tool}({target}) did not return {evidence_id}")
    return _cache[evidence_id]


def field(ref: str) -> float:
    evidence_id, _, name = ref.partition(":")
    value: object = evidence(evidence_id)
    for part in name.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        raise SystemExit(f"{ref} is missing in tool output")
    return float(value)  # type: ignore[arg-type]


def compute(spec: dict) -> float:
    """Exact value of a derive spec (verify_heldout_r12.py has its own implementation)."""
    op = spec["op"]
    if op == "lin":  # sum of k * value: a holding of several positions
        exact = sum(k * field(ref) for k, ref in spec["terms"])
    elif op == "spread":  # highest minus lowest of several operands
        values = [field(ref) for ref in spec["refs"]]
        exact = max(values) - min(values)
    elif op == "spread_ratio":  # highest divided by lowest of several operands
        values = [field(ref) for ref in spec["refs"]]
        exact = max(values) / min(values)
    elif op == "sumn":  # total of several operands
        exact = sum(field(ref) for ref in spec["refs"])
    elif op == "share_total":  # the first operand as a percent of the total of all operands
        values = [field(ref) for ref in spec["refs"]]
        exact = values[0] / sum(values) * 100
    elif op == "pnl":  # |k * (price - cost)|: a position's paper gain or loss against a stated cost
        exact = abs(spec["k"] * (field(spec["a"]) - spec["cost"]))
    elif op == "mean":  # arithmetic mean of several operands
        values = [field(ref) for ref in spec["refs"]]
        exact = sum(values) / len(values)
    elif op == "pnl_multi":  # |sum k_i * price - sum k_i * cost_i|: several purchases of one stock at stated costs
        price = field(spec["a"])
        exact = abs(sum(k * (price - cost) for k, cost in spec["lots"]))
    elif op == "diff_sum":  # |a - (sum of refs)|: one value against the total of several others
        exact = abs(field(spec["a"]) - sum(field(ref) for ref in spec["refs"]))
    else:
        a = field(spec["a"])
        if op == "value":
            exact = a
        elif op == "mul":
            exact = a * spec["k"]
        else:
            b = field(spec["b"])
            exact = {
                "sub": abs(a - b),
                "sum": a + b,
                "div": a / b,
                "pct": a / b * 100,
                "rel": abs(a - b) / abs(b) * 100,
            }[op]
    return round(exact, spec["dp"])


def first_ref(spec: dict) -> str:
    if spec["op"] == "lin":
        return spec["terms"][0][1]
    if spec["op"] in {"spread", "spread_ratio", "sumn", "share_total", "mean"}:
        return spec["refs"][0]
    return spec["a"]


def fact(op: str, a: str | None = None, b: str | None = None, *, dp: int, k: float | None = None,
         terms: list | None = None, refs: list | None = None, cost: float | None = None,
         lots: list | None = None) -> dict:
    spec: dict = {"op": op, "dp": dp}
    for key, value in (("a", a), ("b", b), ("k", k), ("terms", terms), ("refs", refs), ("cost", cost),
                       ("lots", lots)):
        if value is not None:
            spec[key] = value
    value = compute(spec)
    if dp <= 0:
        value = int(value)
    return {"evidence_id": first_ref(spec).partition(":")[0], "value": value, "derive": spec}


def V(ref: str, dp: int = 2) -> dict:
    return fact("value", ref, dp=dp)


def turn(query: str, note: str, lang: str, *, absent: list[str] | None = None, **expect) -> dict:
    expect.setdefault("behavior", "answer")
    expect["forbidden_patterns"] = TRADING
    expect["language"] = lang
    out = {"query": query, "expect": expect, "note": note}
    if absent:
        for ref in absent:
            eid, _, name = ref.partition(":")
            if evidence(eid).get(name) is not None:
                raise SystemExit(f"{ref} is present; cannot label it missing")
        out["verify_absent"] = absent
    return out


FUND = ["get_fundamentals"]
PRICE = ["get_price_history"]
CATL, BYD, CMB, ICBC, PAB = "300750.SZ", "002594.SZ", "600036.SH", "601398.SH", "000001.SZ"
MIDEA, CYPC, ZIJIN, CITIC, SMIC = "000333.SZ", "600900.SH", "601899.SH", "600030.SH", "688981.SH"
PINGAN = "601318.SH"
GOLD, MOUTAI, WLY = "518880.SH", "600519.SH", "000858.SZ"


def f(sym: str, name: str) -> str:
    return f"fundamental_{sym}:{name}"


def p(sym: str, name: str) -> str:
    return f"price_{sym}:{name}"


TASKS: list[dict] = []


def task(tid: str, category: str, lang: str, *turns: dict) -> None:
    TASKS.append({"id": tid, "category": category, "language": lang, "turns": list(turns)})


OOC = {"behavior": "refuse", "required_limitations": ["out_of_coverage"]}


def dated(t: dict, ref: str, date: str) -> dict:
    """The asked date must be the date of the referenced bar (checked by the build and verify scripts)."""
    eid, _, name = ref.partition(":")
    value: object = evidence(eid)
    for part in name.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value != date:
        raise SystemExit(f"{ref} is {value}, not {date}")
    t["verify_date"] = {"ref": ref, "equals": date}
    return t


def after_data(t: dict, evidence_id: str, date: str) -> dict:
    """The asked date is later than the last bar in the data: the answer must say the day is not in the data."""
    as_of = evidence(evidence_id).get("as_of")
    if not as_of or as_of >= date:
        raise SystemExit(f"{evidence_id} as_of {as_of} is not before {date}")
    t["verify_after_data"] = {"evidence_id": evidence_id, "date": date}
    return t


DEF = ["explain_concept"]
CSI500 = "510500.SH"

# ---------------------------------------------------------------- (a) a total against one value whose name comes before the anchor
task("r12h_tvo_zh_01", "total_vs_one_anchor", "zh",
     turn("美的集团加上长江电力的净利润，同紫金矿业比，哪边多出多少", "439.45+345.03 = 784.48 vs 517.77: 266.71亿", "zh",
          required_tools=FUND,
          required_facts=[fact("diff_sum", f(ZIJIN, "net_profit"), refs=[f(MIDEA, "net_profit"), f(CYPC, "net_profit")], dp=-6)],
          required_entities=[MIDEA, CYPC, ZIJIN]))
task("r12h_tvo_zh_02", "total_vs_one_anchor", "zh",
     turn("中信证券和平安银行两家营收加总后，与美的集团相比差多少", "748.54+1314.42 = 2062.96 vs 4585.02: 2522.06亿", "zh",
          required_tools=FUND,
          required_facts=[fact("diff_sum", f(MIDEA, "revenue"), refs=[f(CITIC, "revenue"), f(PAB, "revenue")], dp=-6)],
          required_entities=[CITIC, PAB, MIDEA]))
task("r12h_tvo_zh_03", "total_vs_one_anchor", "zh",
     turn("把比亚迪和中芯国际的市值合在一起，跟紫金矿业比较，谁大，大多少", "7595.54+9588.97 = 17184.51 vs 7921.40: 9263.11亿", "zh",
          required_tools=FUND,
          required_facts=[fact("diff_sum", f(ZIJIN, "total_mv"), refs=[f(BYD, "total_mv"), f(SMIC, "total_mv")], dp=-6)],
          required_entities=[BYD, SMIC, ZIJIN]))
task("r12h_tvo_en_01", "total_vs_one_anchor", "en",
     turn("Taken together, are BYD's and Midea's revenues larger than ICBC's? By how much?",
          "8039.65+4585.02 = 12624.67 vs 8382.70: 4241.97亿 larger", "en", required_tools=FUND,
          required_facts=[fact("diff_sum", f(ICBC, "revenue"), refs=[f(BYD, "revenue"), f(MIDEA, "revenue")], dp=-6)],
          required_entities=[BYD, MIDEA, ICBC]))
task("r12h_tvo_en_02", "total_vs_one_anchor", "en",
     turn("Compared with CMB, how much more or less do Zijin Mining and CITIC Securities earn combined?",
          "517.77+300.76 = 818.53 vs 1501.81: 683.28亿 less", "en", required_tools=FUND,
          required_facts=[fact("diff_sum", f(CMB, "net_profit"), refs=[f(ZIJIN, "net_profit"), f(CITIC, "net_profit")], dp=-6)],
          required_entities=[ZIJIN, CITIC, CMB]))

# ---------------------------------------------------------------- (b) an English total question that opens with an adverb
task("r12h_adv_en_01", "leading_adverb", "en",
     turn("Altogether, what's the net profit of CMB and Midea?", "1501.81+439.45 = 1941.26亿", "en", required_tools=FUND,
          required_facts=[fact("sum", f(CMB, "net_profit"), f(MIDEA, "net_profit"), dp=-6)], required_entities=[CMB, MIDEA]))
task("r12h_adv_en_02", "leading_adverb", "en",
     turn("In total, how much revenue did Zijin Mining and China Yangtze Power report?", "3490.79+862.42 = 4353.21亿", "en",
          required_tools=FUND, required_facts=[fact("sum", f(ZIJIN, "revenue"), f(CYPC, "revenue"), dp=-6)],
          required_entities=[ZIJIN, CYPC]))
task("r12h_adv_en_03", "leading_adverb", "en",
     turn("Combined: SMIC's and CITIC Securities' market value?", "9588.97+4040.44 = 13629.41亿", "en", required_tools=FUND,
          required_facts=[fact("sum", f(SMIC, "total_mv"), f(CITIC, "total_mv"), dp=-6)], required_entities=[SMIC, CITIC]))

# ---------------------------------------------------------------- (c) several purchases in other written forms
task("r12h_lots_zh_01", "multi_purchase_forms", "zh",
     turn("我在28元买进长江电力500股，之后27元又买进500股，按现价总共赚了多少", "1000x28.54 - 27500 = +1,040", "zh",
          required_tools=PRICE, required_facts=[fact("pnl_multi", p(CYPC, "close"), lots=[[500, 28.0], [500, 27.0]], dp=0)],
          required_entity=CYPC))
task("r12h_lots_zh_02", "multi_purchase_forms", "zh",
     turn("招行第一笔38块入了400股，第二笔44块又补了600股，现在整体浮盈还是浮亏，金额多少", "1000x41.26 - 41600 = -340", "zh",
          required_tools=PRICE, required_facts=[fact("pnl_multi", p(CMB, "close"), lots=[[400, 38.0], [600, 44.0]], dp=0)],
          required_entity=CMB))
task("r12h_lots_en_01", "multi_purchase_forms", "en",
     turn("Bought 200 Midea at 85, then 300 more at 76 — where does that leave me now?", "500x80.1 - 39800 = +250", "en",
          required_tools=PRICE, required_facts=[fact("pnl_multi", p(MIDEA, "close"), lots=[[200, 85.0], [300, 76.0]], dp=0)],
          required_entity=MIDEA))
task("r12h_lots_zh_03", "multi_purchase_forms", "zh",
     turn("工商银行收盘价是多少", "8.28", "zh", required_tools=PRICE, required_facts=[V(p(ICBC, "close"), 2)], required_entity=ICBC),
     turn("我先后两次买的：7.6元买了2000股，8.8元买了1000股，合计盈亏多少", "3000x8.28 - 24000 = +840", "zh",
          required_facts=[fact("pnl_multi", p(ICBC, "close"), lots=[[2000, 7.6], [1000, 8.8]], dp=0)], required_entity=ICBC))
task("r12h_lots_en_02", "multi_purchase_forms", "en",
     turn("My CATL position: 100 shares at 320 and 100 shares at 270. Paper P&L?", "200x291.11 - 59000 = -778", "en",
          required_tools=PRICE, required_facts=[fact("pnl_multi", p(CATL, "close"), lots=[[100, 320.0], [100, 270.0]], dp=0)],
          required_entity=CATL))

# ---------------------------------------------------------------- (d) compound counts (万 + 千, 千 + 百)
task("r12h_count_zh_01", "compound_count", "zh",
     turn("两万五千股工商银行按收盘价是多少钱", "25000x8.28 = 207,000", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(ICBC, "close"), k=25000, dp=0)], required_entity=ICBC))
task("r12h_count_zh_02", "compound_count", "zh",
     turn("1万5千股中信证券，现在市值", "15000x25.86 = 387,900", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(CITIC, "close"), k=15000, dp=0)], required_entity=CITIC))
task("r12h_count_zh_03", "compound_count", "zh",
     turn("我拿着4千2百股美的集团，值多少", "4200x80.1 = 336,420", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(MIDEA, "close"), k=4200, dp=0)], required_entity=MIDEA))
task("r12h_count_en_01", "compound_count", "en",
     turn("What are 3.5万 shares of Zijin Mining worth?", "35000x29.79 = 1,042,650", "en", required_tools=PRICE,
          required_facts=[fact("mul", p(ZIJIN, "close"), k=35000, dp=0)], required_entity=ZIJIN))

# ---------------------------------------------------------------- (e) holdings: lists and a position stated before the value question
task("r12h_hold_zh_01", "holding_list_carry", "zh",
     turn("我的账户：长江电力3000股、中信证券2000股，按最近收盘价一共多少钱", "3000x28.54 + 2000x25.86 = 137,340", "zh",
          required_tools=PRICE,
          required_facts=[fact("lin", terms=[[3000, p(CYPC, "close")], [2000, p(CITIC, "close")]], dp=0)],
          required_entities=[CYPC, CITIC]))
task("r12h_hold_en_01", "holding_list_carry", "en",
     turn("I've got 500 BYD shares plus 2,000 of ICBC; combined market value?", "500x83.31 + 2000x8.28 = 58,215", "en",
          required_tools=PRICE,
          required_facts=[fact("lin", terms=[[500, p(BYD, "close")], [2000, p(ICBC, "close")]], dp=0)],
          required_entities=[BYD, ICBC]))
task("r12h_hold_zh_02", "holding_list_carry", "zh",
     turn("手里有美的集团300股", "a stated position: the close 80.1 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(MIDEA, "close"), 2)], required_entity=MIDEA),
     turn("加上中芯国际100股，一共值多少钱", "300x80.1 + 100x111.99 = 35,229", "zh",
          required_facts=[fact("lin", terms=[[300, p(MIDEA, "close")], [100, p(SMIC, "close")]], dp=0)],
          required_entities=[MIDEA, SMIC]))
task("r12h_hold_en_02", "holding_list_carry", "en",
     turn("I hold 1,000 shares of China Yangtze Power.", "a stated position: the close 28.54 at least", "en",
          required_tools=PRICE, required_facts=[V(p(CYPC, "close"), 2)], required_entity=CYPC),
     turn("Plus 400 Midea shares — what's the total worth?", "1000x28.54 + 400x80.1 = 60,580", "en",
          required_facts=[fact("lin", terms=[[1000, p(CYPC, "close")], [400, p(MIDEA, "close")]], dp=0)],
          required_entities=[CYPC, MIDEA]))
task("r12h_hold_zh_03", "holding_list_carry", "zh",
     turn("我持仓紫金矿业2000股", "a stated position: the close 29.79 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(ZIJIN, "close"), 2)], required_entity=ZIJIN),
     turn("还有平安银行1000股，合计市值", "2000x29.79 + 1000x11.57 = 71,150", "zh",
          required_facts=[fact("lin", terms=[[2000, p(ZIJIN, "close")], [1000, p(PAB, "close")]], dp=0)],
          required_entities=[ZIJIN, PAB]))

# ---------------------------------------------------------------- (f) two operations in one message
task("r12h_twoop_zh_01", "two_operations", "zh",
     turn("美的集团和长江电力的净利润加起来是多少？美的占比是多少？", "784.48亿; 439.45/784.48 = 56.02%", "zh",
          required_tools=FUND,
          required_facts=[fact("sum", f(MIDEA, "net_profit"), f(CYPC, "net_profit"), dp=-6),
                          fact("share_total", refs=[f(MIDEA, "net_profit"), f(CYPC, "net_profit")], dp=2)],
          required_entities=[MIDEA, CYPC]))
task("r12h_twoop_zh_02", "two_operations", "zh",
     turn("中信证券和招商银行，哪家ROE更高？高出几个百分点？", "招行 11.8 vs 9.4: 2.4", "zh", required_tools=FUND,
          required_facts=[fact("sub", f(CMB, "roe"), f(CITIC, "roe"), dp=2)], required_entities=[CITIC, CMB]))
task("r12h_twoop_zh_03", "two_operations", "zh",
     turn("工商银行营收是平安银行的几倍？再算一下两者之和", "6.38; 9697.12亿", "zh", required_tools=FUND,
          required_facts=[fact("div", f(ICBC, "revenue"), f(PAB, "revenue"), dp=2),
                          fact("sum", f(ICBC, "revenue"), f(PAB, "revenue"), dp=-6)],
          required_entities=[ICBC, PAB]))
task("r12h_twoop_en_01", "two_operations", "en",
     turn("Which has the lower P/B, Midea or BYD, and by how much?", "2.89 vs 3.17: 0.28", "en", required_tools=FUND,
          required_facts=[fact("sub", f(BYD, "pb"), f(MIDEA, "pb"), dp=2)], required_entities=[MIDEA, BYD]))

# ---------------------------------------------------------------- (g) common short names
task("r12h_short_zh_01", "short_name", "zh",
     turn("紫金的净利润是多少", "517.77亿", "zh", required_tools=FUND, required_facts=[V(f(ZIJIN, "net_profit"), -6)],
          required_entity=ZIJIN))
task("r12h_short_zh_02", "short_name", "zh",
     turn("宁德的总市值", "13470.39亿", "zh", required_tools=FUND, required_facts=[V(f(CATL, "total_mv"), -6)],
          required_entity=CATL))
task("r12h_short_zh_03", "short_name", "zh",
     turn("长电和中信证券的市盈率哪个高", "19.29 vs 10.19", "zh", required_tools=FUND,
          required_facts=[V(f(CYPC, "pe_ttm"), 2), V(f(CITIC, "pe_ttm"), 2)], required_entities=[CYPC, CITIC]))

# ---------------------------------------------------------------- (h) definitions of the metrics the product shows
task("r12h_def_zh_01", "metric_definition", "zh",
     turn("市盈率TTM怎么理解", "a definition from the glossary tool", "zh", required_tools=DEF))
task("r12h_def_zh_02", "metric_definition", "zh",
     turn("总市值和流通市值有什么不同", "a definition from the glossary tool", "zh", required_tools=DEF))
task("r12h_def_en_01", "metric_definition", "en",
     turn("Explain what return on equity measures", "a definition from the glossary tool", "en", required_tools=DEF))
task("r12h_def_zh_03", "metric_definition", "zh",
     turn("毛利率的含义是什么", "a definition from the glossary tool", "zh", required_tools=DEF))

# ---------------------------------------------------------------- (i) a future market value or level in a stated year or horizon
task("r12h_fmv_zh_01", "future_market_value", "zh",
     turn("比亚迪的市值到2029年有没有可能超过1.5万亿", "future market value: hedge", "zh", any_of_tools=FUND + PRICE,
          must_hedge=True))
task("r12h_fmv_en_01", "future_market_value", "en",
     turn("Could ICBC's market value reach 4 trillion yuan by 2028?", "future market value: hedge", "en",
          any_of_tools=FUND + PRICE, must_hedge=True))
task("r12h_fmv_zh_02", "future_market_value", "zh",
     turn("紫金矿业市值三年后能否翻一番", "future market value: hedge", "zh", any_of_tools=FUND + PRICE, must_hedge=True))

# ---------------------------------------------------------------- (j) dividends not in the data, asked as a forecast
task("r12h_div_zh_01", "dividend_forecast", "zh",
     turn("招商银行今年能派多少股息", "no dividend data; future payout: hedge and say it is not in the data", "zh",
          any_of_tools=FUND + PRICE, must_hedge=True, must_state_missing=True))
task("r12h_div_en_01", "dividend_forecast", "en",
     turn("How much dividend will Midea pay per share this year?", "no dividend data; hedge and say so", "en",
          any_of_tools=FUND + PRICE, must_hedge=True, must_state_missing=True))

# ---------------------------------------------------------------- (k) the gap to the 52-week high in yuan
task("r12h_gapy_zh_01", "price_gap_yuan", "zh",
     turn("中芯国际收盘价离52周最高收盘价差几元", "173 - 111.99 = 61.01", "zh", required_tools=PRICE,
          required_facts=[fact("sub", p(SMIC, "range_52w.high"), p(SMIC, "close"), dp=2)], required_entity=SMIC))
task("r12h_gapy_en_01", "price_gap_yuan", "en",
     turn("How many yuan below its 52-week closing high is CMB?", "43.45 - 41.26 = 2.19", "en", required_tools=PRICE,
          required_facts=[fact("sub", p(CMB, "range_52w.high"), p(CMB, "close"), dp=2)], required_entity=CMB))

# ---------------------------------------------------------------- (l) an ETF holding plus a stock holding
task("r12h_etfmix_zh_01", "etf_plus_stock", "zh",
     turn("我有2000份中证500ETF和300股中信证券，总共值多少", "2000x7.472 + 300x25.86 = 22,702", "zh", required_tools=PRICE,
          required_facts=[fact("lin", terms=[[2000, p(CSI500, "close")], [300, p(CITIC, "close")]], dp=0)],
          required_entities=[CSI500, CITIC]))

# ---------------------------------------------------------------- (m) a stated past date that is a 52-week extreme
task("r12h_date_zh_01", "dated_close", "zh",
     dated(turn("宁德时代2026年5月6日的收盘价", "460 (the 52-week high bar)", "zh", required_tools=PRICE,
                required_facts=[V(p(CATL, "range_52w.high"), 2)], required_entity=CATL),
           p(CATL, "range_52w.high_date"), "2026-05-06"))

# ---------------------------------------------------------------- (n) controls (wordings the round-15 rules cover)
task("r12h_ctrl_zh_01", "control", "zh",
     turn("平安银行和招商银行的净利润合计", "426.33+1501.81 = 1928.14亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(PAB, "net_profit"), f(CMB, "net_profit"), dp=-6)], required_entities=[PAB, CMB]))
task("r12h_ctrl_zh_02", "control", "zh",
     turn("工商银行、中信证券两家ROE的均值", "(8.68+9.4)/2 = 9.04", "zh", required_tools=FUND,
          required_facts=[fact("mean", refs=[f(ICBC, "roe"), f(CITIC, "roe")], dp=2)], required_entities=[ICBC, CITIC]))
task("r12h_ctrl_zh_03", "control", "zh",
     turn("我有600股比亚迪，按收盘价值多少", "600x83.31 = 49,986", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(BYD, "close"), k=600, dp=0)], required_entity=BYD))
task("r12h_ctrl_zh_04", "control", "zh",
     turn("我先在70元买了500股美的集团，后来78元又买了500股，现在盈亏多少", "1000x80.1 - 74000 = +6,100", "zh",
          required_tools=PRICE, required_facts=[fact("pnl_multi", p(MIDEA, "close"), lots=[[500, 70.0], [500, 78.0]], dp=0)],
          required_entity=MIDEA))
task("r12h_ctrl_zh_05", "control", "zh",
     turn("美的集团的港股现价", "H share: refuse out of coverage", "zh", **OOC))
task("r12h_ctrl_zh_06", "control", "zh",
     turn("宁德时代明年的净利润会是多少", "FY2027 earnings: hedge", "zh", any_of_tools=FUND, must_hedge=True))
task("r12h_ctrl_zh_07", "control", "zh",
     turn("紫金矿业营收是中信证券的多少倍", "3490.79/748.54 = 4.66", "zh", required_tools=FUND,
          required_facts=[fact("div", f(ZIJIN, "revenue"), f(CITIC, "revenue"), dp=2)], required_entities=[ZIJIN, CITIC]))
task("r12h_ctrl_zh_08", "control", "zh",
     turn("中芯国际和紫金矿业的市值加起来比宁德时代多多少", "9588.97+7921.40 = 17510.37 vs 13470.39: 4039.98亿", "zh",
          required_tools=FUND,
          required_facts=[fact("diff_sum", f(CATL, "total_mv"), refs=[f(SMIC, "total_mv"), f(ZIJIN, "total_mv")], dp=-6)],
          required_entities=[SMIC, ZIJIN, CATL]))


# ================================================================ claims

def _yi(ref: str) -> float:
    return field(ref) / 1e8


CLAIMS: list[dict] = []


def claim(cid: str, lang: str, category: str, text: str, parts: list[tuple[str, object]],
          also_acceptable: tuple[str, ...] = (), fixed: str | None = None) -> None:
    """``fixed``: a verdict that does not come from the data (``unverifiable`` for a line or a period the data does
    not cover); otherwise all parts true -> supported, all false -> contradicted, mixed -> partially_supported."""
    results = [{"part": description, "true": bool(predicate())} for description, predicate in parts]  # type: ignore[operator]
    if fixed is not None:
        verdict = fixed
    else:
        truths = [item["true"] for item in results]
        verdict = "supported" if all(truths) else ("contradicted" if not any(truths) else "partially_supported")
    CLAIMS.append({"id": cid, "lang": lang, "category": category, "claim": text, "expected_verdict": verdict,
                   "also_acceptable": list(also_acceptable), "parts": results})


def near(value: float, target: float, tol: float = 0.05) -> bool:
    return abs(value - target) <= abs(target) * tol


def ratio(a: str, b: str) -> float:
    return field(a) / field(b)


def mean(*refs: str) -> float:
    return sum(field(r) for r in refs) / len(refs)


def yoy(sym: str, name: str) -> float:
    return field(f(sym, name))


# ---- (A) growth first, then the amount after 至 / 达到 / "to"
claim("r12c01", "zh", "amount_after_growth", "长江电力2025年归母净利润同比增长6.1%，达到345.0亿元", [
    ("长江电力 netprofit_yoy ≈ 6.1", lambda: abs(yoy(CYPC, "netprofit_yoy") - 6.1) <= 0.1),
    ("长江电力 net_profit ≈ 345.0亿", lambda: abs(_yi(f(CYPC, "net_profit")) - 345.0) <= 0.5),
])
claim("r12c02", "zh", "amount_after_growth", "美的集团营业总收入同比增长约12%，至4585亿元", [
    ("美的 revenue_yoy ≈ 12", lambda: near(yoy(MIDEA, "revenue_yoy"), 12, 0.05)),
    ("美的 revenue ≈ 4585亿", lambda: abs(_yi(f(MIDEA, "revenue")) - 4585) <= 5),
])
claim("r12c03", "en", "amount_after_growth", "CITIC Securities' revenue climbed 28.8% to RMB74.9 billion", [
    ("中信 revenue_yoy ≈ 28.8", lambda: abs(yoy(CITIC, "revenue_yoy") - 28.8) <= 0.1),
    ("中信 revenue ≈ 74.9 bn", lambda: abs(field(f(CITIC, "revenue")) / 1e9 - 74.9) <= 0.1),
])
claim("r12c04", "zh", "amount_after_growth", "紫金矿业营收同比增长15%，至3200亿元", [
    ("紫金 revenue_yoy ≈ 15", lambda: near(yoy(ZIJIN, "revenue_yoy"), 15, 0.03)),
    ("紫金 revenue ≈ 3200亿 (actual 3490.79: false)", lambda: near(_yi(f(ZIJIN, "revenue")), 3200, 0.02)),
])

# ---- (B) a direction without a number
claim("r12c05", "zh", "direction_only", "比亚迪去年的归母净利润比前一年少了", [
    ("比亚迪 netprofit_yoy < 0", lambda: yoy(BYD, "netprofit_yoy") < 0),
])
claim("r12c06", "zh", "direction_only", "五粮液营收同比增长", [
    ("五粮液 revenue_yoy > 0 (actual -54.55: false)", lambda: yoy(WLY, "revenue_yoy") > 0),
])
claim("r12c07", "en", "direction_only", "Midea's revenue increased last year", [
    ("美的 revenue_yoy > 0", lambda: yoy(MIDEA, "revenue_yoy") > 0),
])
claim("r12c08", "zh", "direction_only", "中信证券的净利润较上年有所减少", [
    ("中信 netprofit_yoy < 0 (actual +37.26: false)", lambda: yoy(CITIC, "netprofit_yoy") < 0),
])

# ---- (C) price against book value
claim("r12c09", "zh", "book_value", "平安银行股价低于每股净资产", [
    ("平安银行 PB < 1", lambda: field(f(PAB, "pb")) < 1),
])
claim("r12c10", "en", "book_value", "Zijin Mining trades at less than book value", [
    ("紫金 PB < 1 (actual 3.92: false)", lambda: field(f(ZIJIN, "pb")) < 1),
])
claim("r12c11", "zh", "book_value", "招商银行跌破净资产", [
    ("招行 PB < 1", lambda: field(f(CMB, "pb")) < 1),
])

# ---- (D) a second clause that keeps the metric and states only a direction or a comparison
claim("r12c12", "en", "elided_second_clause", "CATL's net profit grew about 42%, and Zijin's grew even faster", [
    ("宁德 netprofit_yoy ≈ 42", lambda: near(yoy(CATL, "netprofit_yoy"), 42, 0.02)),
    ("紫金 netprofit_yoy > 宁德 netprofit_yoy", lambda: yoy(ZIJIN, "netprofit_yoy") > yoy(CATL, "netprofit_yoy")),
])
claim("r12c13", "en", "elided_second_clause", "Ping An Bank's revenue fell about 10%, and CMB's fell too", [
    ("平安银行 revenue_yoy ≈ -10", lambda: near(yoy(PAB, "revenue_yoy"), -10, 0.05)),
    ("招行 revenue_yoy < 0 (actual +0.01: false)", lambda: yoy(CMB, "revenue_yoy") < 0),
])
claim("r12c14", "zh", "elided_second_clause", "美的集团净利润增长约15%，比亚迪也实现了增长", [
    ("美的 netprofit_yoy ≈ 15", lambda: near(yoy(MIDEA, "netprofit_yoy"), 15, 0.02)),
    ("比亚迪 netprofit_yoy > 0 (actual -18.82: false)", lambda: yoy(BYD, "netprofit_yoy") > 0),
])
claim("r12c15", "zh", "elided_second_clause", "工商银行市净率0.74倍，招商银行更低", [
    ("工行 PB ≈ 0.74", lambda: abs(field(f(ICBC, "pb")) - 0.74) <= 0.005),
    ("招行 PB < 工行 PB (actual 0.91 > 0.74: false)", lambda: field(f(CMB, "pb")) < field(f(ICBC, "pb"))),
])

# ---- (E) one company's growth rate against another's
claim("r12c16", "zh", "growth_comparison", "宁德时代营收增速高于比亚迪", [
    ("宁德 revenue_yoy > 比亚迪 revenue_yoy", lambda: yoy(CATL, "revenue_yoy") > yoy(BYD, "revenue_yoy")),
])
claim("r12c17", "zh", "growth_comparison", "长江电力净利润增速快过中信证券", [
    ("长江电力 netprofit_yoy > 中信 netprofit_yoy (6.13 < 37.26: false)",
     lambda: yoy(CYPC, "netprofit_yoy") > yoy(CITIC, "netprofit_yoy")),
])

# ---- (F) a common short name as the subject
claim("r12c18", "zh", "short_name", "紫金的市盈率不到12倍", [
    ("紫金 PE < 12", lambda: field(f(ZIJIN, "pe_ttm")) < 12),
])
claim("r12c19", "zh", "short_name", "宁德的市值已在1.3万亿元之上", [
    ("宁德 total_mv > 1.3万亿", lambda: field(f(CATL, "total_mv")) > 1.3e12),
])

# ---- (G) the highest / lowest within a named group of three
claim("r12c20", "zh", "group_extreme", "在美的、比亚迪、宁德时代三者中，美的集团市盈率最低", [
    ("美的 PE < 比亚迪 PE and < 宁德 PE",
     lambda: field(f(MIDEA, "pe_ttm")) < min(field(f(BYD, "pe_ttm")), field(f(CATL, "pe_ttm")))),
])
claim("r12c21", "zh", "group_extreme", "比亚迪、美的、长江电力当中，长江电力的净利润最高", [
    ("长江电力 net_profit > 比亚迪 and > 美的 (439.45 > 345.03: false)",
     lambda: field(f(CYPC, "net_profit")) > max(field(f(BYD, "net_profit")), field(f(MIDEA, "net_profit")))),
])

# ---- (H) a foreign line
claim("r12c22", "zh", "foreign_line", "美的集团H股市盈率约12倍", [], fixed="unverifiable")

# ---- (I) controls
claim("r12c23", "zh", "control", "中芯国际市盈率超过130倍", [
    ("中芯 PE > 130", lambda: field(f(SMIC, "pe_ttm")) > 130),
])
claim("r12c24", "zh", "control", "招商银行2025年营收约3375亿元", [
    ("招行 revenue ≈ 3375亿", lambda: near(_yi(f(CMB, "revenue")), 3375, 0.01)),
])
claim("r12c25", "zh", "control", "比亚迪总市值约9000亿元", [
    ("比亚迪 total_mv ≈ 9000亿 (actual 7595.54: false)", lambda: near(_yi(f(BYD, "total_mv")), 9000, 0.05)),
])
claim("r12c26", "en", "control", "SMIC's 2025 net profit was about RMB5 billion", [
    ("中芯 net_profit ≈ 5 bn", lambda: near(field(f(SMIC, "net_profit")) / 1e9, 5, 0.05)),
])


if __name__ == "__main__":
    out = HERE / "chat_r12_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for t in TASKS:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} conversations ({sum(len(t['turns']) for t in TASKS)} turns) to {out}")
    out = HERE / "claims_r12_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for c in CLAIMS:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(CLAIMS)} claims to {out}")
