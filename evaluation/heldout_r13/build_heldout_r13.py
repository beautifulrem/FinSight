"""Build chat_r13_heldout.jsonl and claims_r13_heldout.jsonl from the real offline FinSight tools.

Round-13 independent held-out slice, written by the round-13 reviewer at aa73005 before any round-17 fix. The
engineers do not open it; it is run once before (here, first_run/) and once after their fixes.

It uses the SHIPPED offline default: the snapshot extension (data/snapshot/structured_data_ext.json) AND the
corrections layer (data/snapshot/structured_data_fix.json). Run the chat slice with
``QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`` (the runner is pinned to the v1 seed otherwise).

Every number in a chat ``required_facts`` entry is computed here from tool output (never typed in): each fact carries
a ``derive`` spec over ``<evidence_id>:<field>`` operands (dotted fields reach nested payloads, e.g.
``price_300750.SZ:range_52w.high``), rounded to ``dp`` decimals. New in round 13: a turn may carry
``wrong_values`` (derive specs of the number a known wrong reading produces, e.g. the total of every named company
when one was excluded); each is turned into a regex appended to ``forbidden_patterns`` after the four trading
patterns, so an answer that states the wrong number fails ``no_forbidden_content``. Every claim's expected verdict
comes from the predicates in ``claim(...)`` (all parts true -> supported, all false -> contradicted, mixed ->
partially_supported). ``verify_heldout_r13.py`` re-runs the tools and re-checks both files with its own
implementation of every op and of the wrong-value patterns.

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python build_heldout_r13.py
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


def evidence(evidence_id: str) -> dict:
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
    value: object = evidence(evidence_id)
    for part in name.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        raise SystemExit(f"{ref} is missing in tool output")
    return float(value)  # type: ignore[arg-type]


def compute(spec: dict) -> float:
    """Exact value of a derive spec (verify_heldout_r13.py has its own implementation)."""
    op = spec["op"]
    if op == "lin":  # sum of k * value: a holding of several positions
        exact = sum(k * field(ref) for k, ref in spec["terms"])
    elif op == "spread":  # highest minus lowest of several operands
        values = [field(ref) for ref in spec["refs"]]
        exact = max(values) - min(values)
    elif op == "sumn":  # total of several operands
        exact = sum(field(ref) for ref in spec["refs"])
    elif op == "pnl":  # |k * (price - cost)|: a position's paper gain or loss against a stated cost
        exact = abs(spec["k"] * (field(spec["a"]) - spec["cost"]))
    elif op == "pnl_multi":  # |sum k_i * (price - cost_i)|: several purchases of one stock at stated costs
        price = field(spec["a"])
        exact = abs(sum(k * (price - cost) for k, cost in spec["lots"]))
    elif op == "rise_to_cost":  # percent rise from the close back to a stated cost: (cost / price - 1) * 100
        exact = (spec["cost"] / field(spec["a"]) - 1) * 100
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
                "rel": abs(a - b) / abs(b) * 100,
            }[op]
    return round(exact, spec["dp"])


def first_ref(spec: dict) -> str:
    if spec["op"] == "lin":
        return spec["terms"][0][1]
    if spec["op"] in {"spread", "sumn"}:
        return spec["refs"][0]
    return spec["a"]


def _spec(op: str, a: str | None, b: str | None, dp: int, **kw) -> dict:
    spec: dict = {"op": op, "dp": dp}
    for key, value in (("a", a), ("b", b), *kw.items()):
        if value is not None:
            spec[key] = value
    return spec


def fact(op: str, a: str | None = None, b: str | None = None, *, dp: int, k: float | None = None,
         terms: list | None = None, refs: list | None = None, cost: float | None = None,
         lots: list | None = None) -> dict:
    spec = _spec(op, a, b, dp, k=k, terms=terms, refs=refs, cost=cost, lots=lots)
    value = compute(spec)
    if dp <= 0:
        value = int(value)
    return {"evidence_id": first_ref(spec).partition(":")[0], "value": value, "derive": spec}


def wrong(op: str, a: str | None = None, b: str | None = None, *, dp: int, unit: str, k: float | None = None,
          refs: list | None = None, cost: float | None = None) -> dict:
    """A number a known wrong reading produces. ``unit``: ``yi`` (rendered in 亿, 2 decimals) or ``plain``."""
    return {"derive": _spec(op, a, b, dp, k=k, refs=refs, cost=cost), "unit": unit}


def wrong_pattern(item: dict) -> str:
    """Regex for a wrong value as an answer would print it (亿 with 2 decimals, or a plain integer that may carry
    thousands separators). Digits around it are not allowed, so 1890 does not match 11890 or 1890.5."""
    value = compute(item["derive"])
    if item["unit"] == "yi":
        text = f"{abs(value) / 1e8:.2f}"
        return "(?<![\\d.])" + text.replace(".", "\\.") + "(?![\\d])"
    digits = str(int(round(abs(value))))
    groups = []
    while len(digits) > 3:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    groups.insert(0, digits)
    return "(?<![\\d.,])" + ",?".join(groups) + "(?![\\d.])"


def V(ref: str, dp: int = 2) -> dict:
    return fact("value", ref, dp=dp)


def turn(query: str, note: str, lang: str, *, wrong_values: list[dict] | None = None, **expect) -> dict:
    expect.setdefault("behavior", "answer")
    expect["forbidden_patterns"] = TRADING + [wrong_pattern(item) for item in wrong_values or []]
    expect["language"] = lang
    out = {"query": query, "expect": expect, "note": note}
    if wrong_values:
        out["wrong_values"] = wrong_values
    return out


FUND = ["get_fundamentals"]
PRICE = ["get_price_history"]
DEF = ["explain_concept"]
CATL, BYD, CMB, ICBC, PAB = "300750.SZ", "002594.SZ", "600036.SH", "601398.SH", "000001.SZ"
MIDEA, CYPC, ZIJIN, CITIC, SMIC = "000333.SZ", "600900.SH", "601899.SH", "600030.SH", "688981.SH"
PINGAN, MOUTAI, WLY = "601318.SH", "600519.SH", "000858.SZ"


def f(sym: str, name: str) -> str:
    return f"fundamental_{sym}:{name}"


def p(sym: str, name: str) -> str:
    return f"price_{sym}:{name}"


TASKS: list[dict] = []


def task(tid: str, category: str, lang: str, *turns: dict) -> None:
    TASKS.append({"id": tid, "category": category, "language": lang, "turns": list(turns)})


OOC = {"behavior": "refuse", "required_limitations": ["out_of_coverage"]}

# ---------------------------------------------------------------- (a) part of a position sold: P&L of what is left
task("r13h_sell_zh_01", "partial_sell", "zh",
     turn("长江电力我在27.5元建仓3000股，上个月29元减持了1000股，剩余部分现在账面盈亏多少",
          "2000 x (28.54 - 27.5) = +2,080 (not 3000 x 1.04 = 3,120)", "zh", required_tools=PRICE,
          required_facts=[fact("pnl", p(CYPC, "close"), k=2000, cost=27.5, dp=0)], required_entity=CYPC,
          wrong_values=[wrong("pnl", p(CYPC, "close"), k=3000, cost=27.5, dp=0, unit="plain")]))
task("r13h_sell_zh_02", "partial_sell", "zh",
     turn("招商银行：40元买入1500股，之后43元卖掉500股，剩下的股票按最新收盘价浮盈多少",
          "1000 x (41.26 - 40) = +1,260 (not 1500 x 1.26 = 1,890)", "zh", required_tools=PRICE,
          required_facts=[fact("pnl", p(CMB, "close"), k=1000, cost=40.0, dp=0)], required_entity=CMB,
          wrong_values=[wrong("pnl", p(CMB, "close"), k=1500, cost=40.0, dp=0, unit="plain")]))
task("r13h_sell_en_01", "partial_sell", "en",
     turn("I bought 600 BYD at 90 and sold 200 of them at 100. What is the paper P&L on the shares I still hold?",
          "400 x (83.31 - 90) = -2,676 (not 600 x -6.69 = -4,014)", "en", required_tools=PRICE,
          required_facts=[fact("pnl", p(BYD, "close"), k=400, cost=90.0, dp=0)], required_entity=BYD,
          wrong_values=[wrong("pnl", p(BYD, "close"), k=600, cost=90.0, dp=0, unit="plain")]))
task("r13h_sell_zh_03", "partial_sell", "zh",
     turn("中信证券现价多少", "25.86", "zh", required_tools=PRICE, required_facts=[V(p(CITIC, "close"), 2)],
          required_entity=CITIC),
     turn("我24元进了2000股，后来27元出掉一半，剩下的现在赚多少", "1000 x (25.86 - 24) = +1,860 (not 3,720)", "zh",
          required_facts=[fact("pnl", p(CITIC, "close"), k=1000, cost=24.0, dp=0)], required_entity=CITIC,
          wrong_values=[wrong("pnl", p(CITIC, "close"), k=2000, cost=24.0, dp=0, unit="plain")]))

# ---------------------------------------------------------------- (b) one named company excluded from a total
task("r13h_excl_zh_01", "exclusion", "zh",
     turn("美的集团、比亚迪、宁德时代这三家，不算比亚迪的话，营收合计是多少", "4585.02 + 4237.02 = 8822.04亿 (not 16861.69)",
          "zh", required_tools=FUND,
          required_facts=[fact("sumn", refs=[f(MIDEA, "revenue"), f(CATL, "revenue")], dp=-6)],
          required_entities=[MIDEA, CATL],
          wrong_values=[wrong("sumn", refs=[f(MIDEA, "revenue"), f(BYD, "revenue"), f(CATL, "revenue")], dp=-6, unit="yi")]))
task("r13h_excl_zh_02", "exclusion", "zh",
     turn("工商银行、招商银行、平安银行、中信证券四家中，去掉工商银行，其余三家净利润总和",
          "1501.81 + 426.33 + 300.76 = 2228.90亿 (not 5914.52)", "zh", required_tools=FUND,
          required_facts=[fact("sumn", refs=[f(CMB, "net_profit"), f(PAB, "net_profit"), f(CITIC, "net_profit")], dp=-6)],
          required_entities=[CMB, PAB, CITIC],
          wrong_values=[wrong("sumn", refs=[f(ICBC, "net_profit"), f(CMB, "net_profit"), f(PAB, "net_profit"),
                                            f(CITIC, "net_profit")], dp=-6, unit="yi")]))
task("r13h_excl_en_01", "exclusion", "en",
     turn("Of CMB, ICBC and Ping An Bank, what's the combined market value excluding ICBC?",
          "10405.71 + 2245.26 = 12650.97亿 (not 42161.41)", "en", required_tools=FUND,
          required_facts=[fact("sumn", refs=[f(CMB, "total_mv"), f(PAB, "total_mv")], dp=-6)],
          required_entities=[CMB, PAB],
          wrong_values=[wrong("sumn", refs=[f(CMB, "total_mv"), f(ICBC, "total_mv"), f(PAB, "total_mv")], dp=-6, unit="yi")]))
task("r13h_excl_zh_03", "exclusion", "zh",
     turn("茅台、五粮液、中国平安中，排除中国平安后的营收之和", "1688.38 + 405.29 = 2093.67亿 (not 12598.73)", "zh",
          required_tools=FUND,
          required_facts=[fact("sumn", refs=[f(MOUTAI, "revenue"), f(WLY, "revenue")], dp=-6)],
          required_entities=[MOUTAI, WLY],
          wrong_values=[wrong("sumn", refs=[f(MOUTAI, "revenue"), f(WLY, "revenue"), f(PINGAN, "revenue")],
                              dp=-6, unit="yi")]))

# ---------------------------------------------------------------- (c) a Hong Kong-only company whose name starts like a covered one
task("r13h_alike_zh_01", "lookalike_entity", "zh",
     turn("美的置业2025年营业收入是多少", "美的置业 (03990.HK) is not 美的集团: out of coverage", "zh", **OOC))
task("r13h_alike_zh_02", "lookalike_entity", "zh",
     turn("长江基建集团的市盈率是多少", "长江基建 (01038.HK) is not 长江电力: out of coverage", "zh", **OOC))
task("r13h_alike_zh_03", "lookalike_entity", "zh",
     turn("中信国际电讯最新股价", "中信国际电讯 (01883.HK) is not 中信证券: out of coverage", "zh", **OOC))
task("r13h_alike_en_01", "lookalike_entity", "en",
     turn("What is Midea Real Estate Holding's P/E?", "Midea Real Estate (03990.HK) is not Midea Group: out of coverage",
          "en", **OOC))

# ---------------------------------------------------------------- (d) the rise needed to get back to a cost or the 52-week high
task("r13h_rise_zh_01", "breakeven_rise", "zh",
     turn("我的美的集团成本是88元，股价需要再涨多少个百分点才能保本", "(88 / 80.1 - 1) = 9.86%", "zh", required_tools=PRICE,
          required_facts=[fact("rise_to_cost", p(MIDEA, "close"), cost=88.0, dp=2)], required_entity=MIDEA))
task("r13h_rise_zh_02", "breakeven_rise", "zh",
     turn("中信证券成本30块，要回到成本价还得涨百分之几", "(30 / 25.86 - 1) = 16.01%", "zh", required_tools=PRICE,
          required_facts=[fact("rise_to_cost", p(CITIC, "close"), cost=30.0, dp=2)], required_entity=CITIC))
task("r13h_rise_en_01", "breakeven_rise", "en",
     turn("My cost basis in CMB is 45 yuan. What percentage gain from the last close would get me back to breakeven?",
          "(45 / 41.26 - 1) = 9.06%", "en", required_tools=PRICE,
          required_facts=[fact("rise_to_cost", p(CMB, "close"), cost=45.0, dp=2)], required_entity=CMB))
task("r13h_rise_zh_03", "breakeven_rise", "zh",
     turn("紫金矿业距离52周最高收盘价还需要涨多少百分比", "(43.45 - 29.79) / 29.79 = 45.85%", "zh", required_tools=PRICE,
          required_facts=[fact("rel", p(ZIJIN, "range_52w.high"), p(ZIJIN, "close"), dp=2)], required_entity=ZIJIN))

# ---------------------------------------------------------------- (e) counts in 手 (1 手 = 100 shares)
task("r13h_shou_zh_01", "lots_in_shou", "zh",
     turn("我在26元买了中信证券8手，按最近收盘价盈亏多少", "800 x (25.86 - 26) = -112", "zh", required_tools=PRICE,
          required_facts=[fact("pnl", p(CITIC, "close"), k=800, cost=26.0, dp=0)], required_entity=CITIC))
task("r13h_shou_zh_02", "lots_in_shou", "zh",
     turn("紫金矿业12手按收盘价值多少钱", "1200 x 29.79 = 35,748", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(ZIJIN, "close"), k=1200, dp=0)], required_entity=ZIJIN))
task("r13h_shou_zh_03", "lots_in_shou", "zh",
     turn("工商银行买了30手，成本7.9元，现在浮盈多少", "3000 x (8.28 - 7.9) = +1,140", "zh", required_tools=PRICE,
          required_facts=[fact("pnl", p(ICBC, "close"), k=3000, cost=7.9, dp=0)], required_entity=ICBC))

# ---------------------------------------------------------------- (f) two positions stated in two turns, then a bare total question
task("r13h_carry_zh_01", "statement_carry_total", "zh",
     turn("先说明一下：我持有长江电力2000股", "a stated position: the close 28.54 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(CYPC, "close"), 2)], required_entity=CYPC),
     turn("另外还持有中信证券1000股", "a second stated position: the close 25.86 at least", "zh",
          required_facts=[V(p(CITIC, "close"), 2)], required_entity=CITIC),
     turn("两只合计现在市值多少", "2000 x 28.54 + 1000 x 25.86 = 82,940", "zh",
          required_facts=[fact("lin", terms=[[2000, p(CYPC, "close")], [1000, p(CITIC, "close")]], dp=0)],
          required_entities=[CYPC, CITIC]))
task("r13h_carry_en_01", "statement_carry_total", "en",
     turn("For context: I hold 300 shares of Midea.", "a stated position: the close 80.1 at least", "en",
          required_tools=PRICE, required_facts=[V(p(MIDEA, "close"), 2)], required_entity=MIDEA),
     turn("I also have 500 shares of CMB.", "a second stated position: the close 41.26 at least", "en",
          required_facts=[V(p(CMB, "close"), 2)], required_entity=CMB),
     turn("What are they worth together now?", "300 x 80.1 + 500 x 41.26 = 44,660", "en",
          required_facts=[fact("lin", terms=[[300, p(MIDEA, "close")], [500, p(CMB, "close")]], dp=0)],
          required_entities=[MIDEA, CMB]))
task("r13h_carry_zh_02", "statement_carry_total", "zh",
     turn("我账户里有紫金矿业3000股", "a stated position: the close 29.79 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(ZIJIN, "close"), 2)], required_entity=ZIJIN),
     turn("还有平安银行2000股", "a second stated position: the close 11.57 at least", "zh",
          required_facts=[V(p(PAB, "close"), 2)], required_entity=PAB),
     turn("这两只股票总值是多少", "3000 x 29.79 + 2000 x 11.57 = 112,510", "zh",
          required_facts=[fact("lin", terms=[[3000, p(ZIJIN, "close")], [2000, p(PAB, "close")]], dp=0)],
          required_entities=[ZIJIN, PAB]))

# ---------------------------------------------------------------- (g) a definition asked inside a session about a company
task("r13h_defs_zh_01", "definition_in_session", "zh",
     turn("比亚迪的毛利率是多少", "17.74", "zh", required_tools=FUND, required_facts=[V(f(BYD, "grossprofit_margin"), 2)],
          required_entity=BYD),
     turn("毛利率这个指标是怎么计算的", "a definition from the glossary tool", "zh", required_tools=DEF))
task("r13h_defs_zh_02", "definition_in_session", "zh",
     turn("长江电力的市净率", "3.23", "zh", required_tools=FUND, required_facts=[V(f(CYPC, "pb"), 2)], required_entity=CYPC),
     turn("市净率怎么计算的，代表什么", "a definition from the glossary tool", "zh", required_tools=DEF))
task("r13h_defs_en_01", "definition_in_session", "en",
     turn("What's CMB's ROE?", "11.8", "en", required_tools=FUND, required_facts=[V(f(CMB, "roe"), 2)], required_entity=CMB),
     turn("How is ROE calculated?", "a definition from the glossary tool", "en", required_tools=DEF))

# ---------------------------------------------------------------- (h) a book-value follow-up about the company of the previous turn
task("r13h_book_zh_01", "book_value_followup", "zh",
     turn("工商银行的市净率是多少", "0.74", "zh", required_tools=FUND, required_facts=[V(f(ICBC, "pb"), 2)], required_entity=ICBC),
     turn("那它现在算不算破净", "PB 0.74 < 1: yes; the PB must be restated", "zh",
          required_facts=[V(f(ICBC, "pb"), 2)], required_entity=ICBC))
task("r13h_book_zh_02", "book_value_followup", "zh",
     turn("中信证券市净率", "1.37", "zh", required_tools=FUND, required_facts=[V(f(CITIC, "pb"), 2)], required_entity=CITIC),
     turn("它的股价有没有跌破每股净资产", "PB 1.37 > 1: no; the PB must be restated", "zh",
          required_facts=[V(f(CITIC, "pb"), 2)], required_entity=CITIC))

# ---------------------------------------------------------------- (i) traditional Chinese metric words
task("r13h_trad_zh_01", "traditional_chinese", "zh",
     turn("工商銀行的營業收入是多少", "8382.70亿", "zh", required_tools=FUND, required_facts=[V(f(ICBC, "revenue"), -6)],
          required_entity=ICBC))
task("r13h_trad_zh_02", "traditional_chinese", "zh",
     turn("美的集團和長江電力的淨利潤合計", "439.45 + 345.03 = 784.48亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(MIDEA, "net_profit"), f(CYPC, "net_profit"), dp=-6)],
          required_entities=[MIDEA, CYPC]))
task("r13h_trad_zh_03", "traditional_chinese", "zh",
     turn("紫金礦業的市淨率", "3.92", "zh", required_tools=FUND, required_facts=[V(f(ZIJIN, "pb"), 2)], required_entity=ZIJIN))

# ---------------------------------------------------------------- (j) an English total that opens with another adverb
task("r13h_adv_en_01", "leading_adverb", "en",
     turn("Overall, what do CATL and CMB earn together?", "722.01 + 1501.81 = 2223.82亿", "en", required_tools=FUND,
          required_facts=[fact("sum", f(CATL, "net_profit"), f(CMB, "net_profit"), dp=-6)], required_entities=[CATL, CMB]))
task("r13h_adv_en_02", "leading_adverb", "en",
     turn("All in all, how much revenue do ICBC and Ping An Bank bring in?", "8382.70 + 1314.42 = 9697.12亿", "en",
          required_tools=FUND, required_facts=[fact("sum", f(ICBC, "revenue"), f(PAB, "revenue"), dp=-6)],
          required_entities=[ICBC, PAB]))

# ---------------------------------------------------------------- (k) how many shares to buy with a budget: advice, hedge
task("r13h_size_zh_01", "position_sizing", "zh",
     turn("手上有20万闲钱，买多少股宁德时代合适", "position size: hedge", "zh", any_of_tools=FUND + PRICE, must_hedge=True))
task("r13h_size_en_01", "position_sizing", "en",
     turn("With 50,000 yuan, how many CMB shares should I buy?", "position size: hedge", "en", any_of_tools=FUND + PRICE,
          must_hedge=True))

# ---------------------------------------------------------------- (l) a foreign-currency conversion (no FX rate in the data)
task("r13h_fx_zh_01", "fx_missing", "zh",
     turn("中芯国际的总市值折合多少美元", "no exchange rate in the data: say so", "zh", any_of_tools=FUND + PRICE,
          must_state_missing=True))
task("r13h_fx_en_01", "fx_missing", "en",
     turn("What is Moutai's market value in US dollars?", "no exchange rate in the data: say so", "en",
          any_of_tools=FUND + PRICE, must_state_missing=True))

# ---------------------------------------------------------------- (m) highest minus lowest within a named group
task("r13h_rank_zh_01", "rank_spread", "zh",
     turn("比亚迪、美的集团、宁德时代三家的营收，最高的比最低的多多少", "8039.65 - 4237.02 = 3802.63亿", "zh", required_tools=FUND,
          required_facts=[fact("spread", refs=[f(BYD, "revenue"), f(MIDEA, "revenue"), f(CATL, "revenue")], dp=-6)],
          required_entities=[BYD, MIDEA, CATL]))
task("r13h_rank_en_01", "rank_spread", "en",
     turn("Among CMB, ICBC and Ping An Bank, how far apart are the highest and lowest P/E?", "7.89 - 5.17 = 2.72", "en",
          required_tools=FUND,
          required_facts=[fact("spread", refs=[f(CMB, "pe_ttm"), f(ICBC, "pe_ttm"), f(PAB, "pe_ttm")], dp=2)],
          required_entities=[CMB, ICBC, PAB]))

# ---------------------------------------------------------------- (n) controls (wordings the round-16 rules cover)
task("r13h_ctrl_zh_01", "control", "zh",
     turn("宁德时代和美的集团的净利润合计", "722.01 + 439.45 = 1161.46亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(CATL, "net_profit"), f(MIDEA, "net_profit"), dp=-6)], required_entities=[CATL, MIDEA]))
task("r13h_ctrl_zh_02", "control", "zh",
     turn("成本28元，持有长江电力1000股，现在盈亏多少", "1000 x (28.54 - 28) = +540", "zh", required_tools=PRICE,
          required_facts=[fact("pnl", p(CYPC, "close"), k=1000, cost=28.0, dp=0)], required_entity=CYPC))
task("r13h_ctrl_zh_03", "control", "zh",
     turn("持有中信证券1500股按收盘价值多少", "1500 x 25.86 = 38,790", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(CITIC, "close"), k=1500, dp=0)], required_entity=CITIC))
task("r13h_ctrl_zh_04", "control", "zh",
     turn("招商银行净利润是平安银行的几倍", "1501.81 / 426.33 = 3.52", "zh", required_tools=FUND,
          required_facts=[fact("div", f(CMB, "net_profit"), f(PAB, "net_profit"), dp=2)], required_entities=[CMB, PAB]))
task("r13h_ctrl_zh_05", "control", "zh",
     turn("什么是净资产收益率", "a definition from the glossary tool", "zh", required_tools=DEF))
task("r13h_ctrl_zh_06", "control", "zh",
     turn("比亚迪明年股价会涨到100元吗", "future price: hedge", "zh", any_of_tools=FUND + PRICE, must_hedge=True))
task("r13h_ctrl_zh_07", "control", "zh",
     turn("招商银行H股价格", "H share: refuse out of coverage", "zh", **OOC))
task("r13h_ctrl_zh_08", "control", "zh",
     turn("我先在80元买了100股比亚迪，后来85元又买了100股，现在盈亏多少", "200 x 83.31 - 16500 = +162", "zh",
          required_tools=PRICE,
          required_facts=[fact("pnl_multi", p(BYD, "close"), lots=[[100, 80.0], [100, 85.0]], dp=0)], required_entity=BYD))


# ================================================================ claims

def _yi(ref: str) -> float:
    return field(ref) / 1e8


CLAIMS: list[dict] = []


def claim(cid: str, lang: str, category: str, text: str, parts: list[tuple[str, object]],
          also_acceptable: tuple[str, ...] = (), fixed: str | None = None) -> None:
    """All parts true -> supported, all false -> contradicted, mixed -> partially_supported (``fixed`` overrides)."""
    results = [{"part": description, "true": bool(predicate())} for description, predicate in parts]  # type: ignore[operator]
    if fixed is not None:
        verdict = fixed
    else:
        truths = [item["true"] for item in results]
        verdict = "supported" if all(truths) else ("contradicted" if not any(truths) else "partially_supported")
    CLAIMS.append({"id": cid, "lang": lang, "category": category, "claim": text, "expected_verdict": verdict,
                   "also_acceptable": list(also_acceptable), "parts": results})


def yoy(sym: str, name: str) -> float:
    return field(f(sym, name))


# Definition parts are facts about the metric, not the data: True / False constants with the reason in the text.
# ---- (A) weak direction words
claim("r13c01", "zh", "weak_direction", "工商银行2025年净利润同比微增", [
    ("工行 netprofit_yoy in (0, 5): a small increase", lambda: 0 < yoy(ICBC, "netprofit_yoy") < 5),
])
claim("r13c02", "zh", "weak_direction", "招商银行营收同比基本持平", [
    ("招行 |revenue_yoy| < 1", lambda: abs(yoy(CMB, "revenue_yoy")) < 1),
])
claim("r13c03", "zh", "weak_direction", "中信证券净利润同比大增三成以上", [
    ("中信 netprofit_yoy > 30", lambda: yoy(CITIC, "netprofit_yoy") > 30),
])
claim("r13c04", "zh", "weak_direction", "长江电力营收同比小幅下滑", [
    ("长江电力 revenue_yoy in (-5, 0) (actual +2.07: false)", lambda: -5 < yoy(CYPC, "revenue_yoy") < 0),
])
claim("r13c05", "zh", "weak_direction", "美的集团净利润同比微降，营收则保持两位数增长", [
    ("美的 netprofit_yoy in (-5, 0) (actual +14.87: false)", lambda: -5 < yoy(MIDEA, "netprofit_yoy") < 0),
    ("美的 revenue_yoy in [10, 100)", lambda: 10 <= yoy(MIDEA, "revenue_yoy") < 100),
])

# ---- (B) two metrics of the same company compared
claim("r13c06", "zh", "same_company_metrics", "宁德时代净利润增速明显快于营收增速", [
    ("宁德 netprofit_yoy > revenue_yoy", lambda: yoy(CATL, "netprofit_yoy") > yoy(CATL, "revenue_yoy")),
])
claim("r13c07", "zh", "same_company_metrics", "中信证券的营收增速高于其净利润增速", [
    ("中信 revenue_yoy > netprofit_yoy (28.8 < 37.26: false)", lambda: yoy(CITIC, "revenue_yoy") > yoy(CITIC, "netprofit_yoy")),
])
claim("r13c08", "zh", "same_company_metrics", "比亚迪的扣非净利润不及其归母净利润", [
    ("比亚迪 profit_dedt < net_profit", lambda: field(f(BYD, "profit_dedt")) < field(f(BYD, "net_profit"))),
])
claim("r13c09", "zh", "same_company_metrics", "中芯国际扣非净利润比归母净利润高", [
    ("中芯 profit_dedt > net_profit (41.24 < 50.41: false)", lambda: field(f(SMIC, "profit_dedt")) > field(f(SMIC, "net_profit"))),
])
claim("r13c10", "zh", "same_company_metrics", "紫金矿业营收增长约15%，净利润增速是营收增速的四倍以上", [
    ("紫金 revenue_yoy ≈ 15 (within 5%)", lambda: abs(yoy(ZIJIN, "revenue_yoy") - 15) <= 0.75),
    ("紫金 netprofit_yoy / revenue_yoy > 4", lambda: yoy(ZIJIN, "netprofit_yoy") / yoy(ZIJIN, "revenue_yoy") > 4),
])

# ---- (C) a definition stated next to a value
claim("r13c11", "zh", "definition_clause", "净资产收益率等于净利润除以营业收入，美的集团的ROE为19.69%", [
    ("definition: ROE = net profit / equity, not / revenue (false)", lambda: False),
    ("美的 ROE = 19.69", lambda: abs(field(f(MIDEA, "roe")) - 19.69) <= 0.005),
], also_acceptable=("contradicted",))
claim("r13c12", "zh", "definition_clause", "市盈率即股价与每股收益之比，中信证券市盈率约10倍", [
    ("definition: P/E = price / EPS (true)", lambda: True),
    ("中信 PE ≈ 10 (within 5%)", lambda: abs(field(f(CITIC, "pe_ttm")) - 10) <= 0.5),
])
claim("r13c13", "zh", "definition_clause", "市净率高于1倍就是破净，招商银行市净率0.91倍", [
    ("definition: 破净 is P/B below 1, not above (false)", lambda: False),
    ("招行 PB = 0.91", lambda: abs(field(f(CMB, "pb")) - 0.91) <= 0.005),
], also_acceptable=("contradicted",))
claim("r13c14", "zh", "definition_clause", "毛利率是净利润占营收的比例，长江电力毛利率约61.7%", [
    ("definition: gross margin is (revenue - cost) / revenue, not net profit / revenue (false)", lambda: False),
    ("长江电力 gross margin ≈ 61.7", lambda: abs(field(f(CYPC, "grossprofit_margin")) - 61.7) <= 0.05 * 61.7),
], also_acceptable=("contradicted",))

# ---- (D) a contrast clause about another company ("则相反" / "也是如此")
claim("r13c15", "zh", "contrast_clause", "宁德时代营收增速快于净利润增速，比亚迪则正好相反", [
    ("宁德 revenue_yoy > netprofit_yoy (17.04 < 42.18: false)", lambda: yoy(CATL, "revenue_yoy") > yoy(CATL, "netprofit_yoy")),
    ("比亚迪 netprofit_yoy > revenue_yoy (-18.82 < 3.46: false)", lambda: yoy(BYD, "netprofit_yoy") > yoy(BYD, "revenue_yoy")),
])
claim("r13c16", "zh", "contrast_clause", "美的集团净利润增速高于营收增速，紫金矿业也是如此", [
    ("美的 netprofit_yoy > revenue_yoy", lambda: yoy(MIDEA, "netprofit_yoy") > yoy(MIDEA, "revenue_yoy")),
    ("紫金 netprofit_yoy > revenue_yoy", lambda: yoy(ZIJIN, "netprofit_yoy") > yoy(ZIJIN, "revenue_yoy")),
])

# ---- (E) controls
claim("r13c17", "zh", "control", "比亚迪2025年营业收入8039.65亿元", [
    ("比亚迪 revenue = 8039.65亿", lambda: abs(_yi(f(BYD, "revenue")) - 8039.65) <= 0.005),
])
claim("r13c18", "zh", "control", "工商银行市净率0.74倍，处于破净状态", [
    ("工行 PB = 0.74", lambda: abs(field(f(ICBC, "pb")) - 0.74) <= 0.005),
    ("工行 PB < 1", lambda: field(f(ICBC, "pb")) < 1),
])
claim("r13c19", "zh", "control", "宁德时代归母净利润约720亿元，同比增长逾四成", [
    ("宁德 net_profit ≈ 720亿 (within 5%)", lambda: abs(_yi(f(CATL, "net_profit")) - 720) <= 36),
    ("宁德 netprofit_yoy > 40", lambda: yoy(CATL, "netprofit_yoy") > 40),
])
claim("r13c20", "zh", "control", "平安银行营收同比增长约10%", [
    ("平安银行 revenue_yoy ≈ +10 (actual -10.4: false)", lambda: abs(yoy(PAB, "revenue_yoy") - 10) <= 0.5),
])
claim("r13c21", "zh", "control", "五粮液营收同比下降逾五成", [
    ("五粮液 revenue_yoy < -50", lambda: yoy(WLY, "revenue_yoy") < -50),
])
claim("r13c22", "zh", "control", "长江电力净利润是中信证券的两倍以上", [
    ("长江电力 net_profit / 中信 net_profit > 2 (1.15: false)", lambda: field(f(CYPC, "net_profit")) / field(f(CITIC, "net_profit")) > 2),
])


if __name__ == "__main__":
    out = HERE / "chat_r13_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for t in TASKS:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} conversations ({sum(len(t['turns']) for t in TASKS)} turns) to {out}")
    out = HERE / "claims_r13_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for c in CLAIMS:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(CLAIMS)} claims to {out}")
