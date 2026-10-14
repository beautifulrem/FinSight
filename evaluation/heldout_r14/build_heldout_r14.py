"""Build chat_r14_heldout.jsonl and claims_r14_heldout.jsonl from the real offline FinSight tools.

Round-14 independent held-out slice, written by the round-14 reviewer at 64cfbeb before any round-18 fix. The
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
partially_supported). ``verify_heldout_r14.py`` re-runs the tools and re-checks both files with its own
implementation of every op and of the wrong-value patterns.

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python build_heldout_r14.py
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
    """Exact value of a derive spec (verify_heldout_r14.py has its own implementation)."""
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
    elif op == "realised":  # round 14: |k * (sale - cost)|, a closed position (``a`` only anchors the evidence id)
        exact = abs(spec["k"] * (spec["sale"] - spec["cost"]))
    elif op == "pnl_avg":  # round 14: buys at stated costs, then ``sold`` shares; the rest against the average cost
        bought = sum(k for k, _c in spec["lots"])
        average = sum(k * c for k, c in spec["lots"]) / bought
        exact = abs((bought - spec["sold"]) * (field(spec["a"]) - average))
    elif op == "short_pnl":  # round 14: a short position's gain or loss, |k * (open - price)|
        exact = abs(spec["k"] * (spec["cost"] - field(spec["a"])))
    elif op == "weight":  # round 14: the first holding's share of the total value, in percent
        values = [k * field(ref) for k, ref in spec["terms"]]
        exact = values[0] / sum(values) * 100
    elif op == "range_pos":  # round 14: where the close sits in the 52-week closing range, in percent
        low, high = field(spec["lo"]), field(spec["hi"])
        exact = (field(spec["a"]) - low) / (high - low) * 100
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
    if spec["op"] in {"lin", "weight"}:
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
         lots: list | None = None, sale: float | None = None, sold: float | None = None,
         lo: str | None = None, hi: str | None = None) -> dict:
    spec = _spec(op, a, b, dp, k=k, terms=terms, refs=refs, cost=cost, lots=lots, sale=sale, sold=sold, lo=lo, hi=hi)
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


def turn(query: str, note: str, lang: str, *, wrong_values: list[dict] | None = None,
         extra_forbidden: list[str] | None = None, **expect) -> dict:
    """``extra_forbidden`` (round 14): literal regexes appended after the wrong-value patterns (a loss stated for a
    short position that gained)."""
    expect.setdefault("behavior", "answer")
    expect["forbidden_patterns"] = TRADING + [wrong_pattern(item) for item in wrong_values or []] + list(extra_forbidden or [])
    expect["language"] = lang
    out = {"query": query, "expect": expect, "note": note}
    if wrong_values:
        out["wrong_values"] = wrong_values
    if extra_forbidden:
        out["extra_forbidden"] = list(extra_forbidden)
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

# ---------------------------------------------------------------- (a) a closed position: the realised P&L, not a paper P&L at the close
task("r14h_closed_zh_01", "closed_position", "zh",
     turn("去年在32元买的紫金矿业2000股，上周在35元一次性清仓了，这笔一共赚了多少",
          "2000 x (35 - 32) = +6,000 realised (not a paper 2000 x (29.79 - 32) = -4,420)", "zh", required_tools=PRICE,
          required_facts=[fact("realised", p(ZIJIN, "close"), k=2000, sale=35.0, cost=32.0, dp=0)],
          wrong_values=[wrong("pnl", p(ZIJIN, "close"), k=2000, cost=32.0, dp=0, unit="plain")]))
task("r14h_closed_zh_02", "closed_position", "zh",
     turn("比亚迪我95块进的600股，前天88块全部抛了，亏了多少钱",
          "600 x (88 - 95) = -4,200 realised (not 600 x (83.31 - 95) = -7,014)", "zh", required_tools=PRICE,
          required_facts=[fact("realised", p(BYD, "close"), k=600, sale=88.0, cost=95.0, dp=0)],
          wrong_values=[wrong("pnl", p(BYD, "close"), k=600, cost=95.0, dp=0, unit="plain")]))
task("r14h_closed_en_01", "closed_position", "en",
     turn("I closed out my whole CMB position yesterday: 1,200 shares bought at 36, sold at 42. What did I make?",
          "1200 x (42 - 36) = +7,200 realised (not 1200 x (41.26 - 36) = 6,312)", "en", required_tools=PRICE,
          required_facts=[fact("realised", p(CMB, "close"), k=1200, sale=42.0, cost=36.0, dp=0)],
          wrong_values=[wrong("pnl", p(CMB, "close"), k=1200, cost=36.0, dp=0, unit="plain")]))
task("r14h_closed_zh_03", "closed_position", "zh",
     turn("中信证券2000股，成本22元，昨天26元全部卖出，盈利多少",
          "2000 x (26 - 22) = +8,000 realised (not 2000 x 3.86 = 7,720)", "zh", required_tools=PRICE,
          required_facts=[fact("realised", p(CITIC, "close"), k=2000, sale=26.0, cost=22.0, dp=0)],
          wrong_values=[wrong("pnl", p(CITIC, "close"), k=2000, cost=22.0, dp=0, unit="plain")]))
task("r14h_closed_en_02", "closed_position", "en",
     turn("Sold all 300 of my Midea shares at 85 after buying at 70 — total profit?",
          "300 x (85 - 70) = +4,500 realised (not 300 x 10.1 = 3,030)", "en", required_tools=PRICE,
          required_facts=[fact("realised", p(MIDEA, "close"), k=300, sale=85.0, cost=70.0, dp=0)],
          wrong_values=[wrong("pnl", p(MIDEA, "close"), k=300, cost=70.0, dp=0, unit="plain")]))

# ---------------------------------------------------------------- (b) a price and a count separated by a comma are two numbers
task("r14h_comma_zh_01", "comma_count", "zh",
     turn("招行持仓：买入均价39，2000股，按最新收盘价市值多少", "2000 x 41.26 = 82,520 (not '392000 股')", "zh",
          required_tools=PRICE, required_facts=[fact("mul", p(CMB, "close"), k=2000, dp=0)], required_entity=CMB,
          wrong_values=[wrong("mul", p(CMB, "close"), k=392000, dp=0, unit="plain")]))
task("r14h_comma_zh_02", "comma_count", "zh",
     turn("工行均价7，5000股，现在市值多少", "5000 x 8.28 = 41,400 (not '75000 股')", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(ICBC, "close"), k=5000, dp=0)], required_entity=ICBC,
          wrong_values=[wrong("mul", p(ICBC, "close"), k=75000, dp=0, unit="plain")]))
task("r14h_comma_zh_03", "comma_count", "zh",
     turn("比亚迪成本价85，600股，按收盘价值多少钱", "600 x 83.31 = 49,986 (not '85600 股')", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(BYD, "close"), k=600, dp=0)], required_entity=BYD,
          wrong_values=[wrong("mul", p(BYD, "close"), k=85600, dp=0, unit="plain")]))

# ---------------------------------------------------------------- (c) two purchases then a sale: the rest against the average cost
task("r14h_lots_zh_01", "buys_then_sale", "zh",
     turn("我紫金矿业先在28买了1000股，接着32又加仓了1000股，然后30卖掉800股，剩下的浮动盈亏是多少",
          "avg 30; 1200 x (29.79 - 30) = -252 (not the first lot only: 200 x 1.79 = 358)", "zh", required_tools=PRICE,
          required_facts=[fact("pnl_avg", p(ZIJIN, "close"), lots=[[1000, 28.0], [1000, 32.0]], sold=800, dp=0)],
          required_entity=ZIJIN,
          wrong_values=[wrong("pnl", p(ZIJIN, "close"), k=200, cost=28.0, dp=0, unit="plain")]))
task("r14h_lots_zh_02", "buys_then_sale", "zh",
     turn("工商银行7块买了3000股，7.6又加了2000股，8.1卖出2500股，剩下的赚了多少",
          "avg 7.24; 2500 x (8.28 - 7.24) = +2,600 (not the first lot only: 500 x 1.28 = 640)", "zh", required_tools=PRICE,
          required_facts=[fact("pnl_avg", p(ICBC, "close"), lots=[[3000, 7.0], [2000, 7.6]], sold=2500, dp=0)],
          required_entity=ICBC,
          wrong_values=[wrong("pnl", p(ICBC, "close"), k=500, cost=7.0, dp=0, unit="plain")]))

# ---------------------------------------------------------------- (d) a position with its cost stated in one turn, the P&L asked in the next
task("r14h_cost_zh_01", "stated_cost_carry", "zh",
     turn("我持有长江电力1500股，平均成本26.8元", "a stated position: the close 28.54 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(CYPC, "close"), 2)], required_entity=CYPC),
     turn("按最新收盘价算我赚了多少", "1500 x (28.54 - 26.8) = +2,610", "zh",
          required_facts=[fact("pnl", p(CYPC, "close"), k=1500, cost=26.8, dp=0)], required_entity=CYPC))
task("r14h_cost_zh_02", "stated_cost_carry", "zh",
     turn("仓里有五粮液200股，买入均价130", "a stated position: the close 100.64 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(WLY, "close"), 2)], required_entity=WLY),
     turn("现在亏了多少", "200 x (100.64 - 130) = -5,872", "zh",
          required_facts=[fact("pnl", p(WLY, "close"), k=200, cost=130.0, dp=0)], required_entity=WLY))
task("r14h_cost_en_01", "stated_cost_carry", "en",
     turn("I'm holding 400 shares of Zijin, average cost 33.", "a stated position: the close 29.79 at least", "en",
          required_tools=PRICE, required_facts=[V(p(ZIJIN, "close"), 2)], required_entity=ZIJIN),
     turn("How much am I down at the last close?", "400 x (29.79 - 33) = -1,284", "en",
          required_facts=[fact("pnl", p(ZIJIN, "close"), k=400, cost=33.0, dp=0)], required_entity=ZIJIN))
task("r14h_cost_zh_03", "stated_cost_carry", "zh",
     turn("我有招行800股，成本45元", "a stated position: the close 41.26 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(CMB, "close"), 2)], required_entity=CMB),
     turn("要涨多少才能回本", "(45 / 41.26 - 1) = 9.06%", "zh",
          required_facts=[fact("rise_to_cost", p(CMB, "close"), cost=45.0, dp=2)], required_entity=CMB))

# ---------------------------------------------------------------- (e) a sale in a follow-up turn about a stated position
task("r14h_sale_en_01", "sale_followup", "en",
     turn("I've got 1,500 BYD at an average of 92.", "a stated position: the close 83.31 at least", "en",
          required_tools=PRICE, required_facts=[V(p(BYD, "close"), 2)], required_entity=BYD),
     turn("Sold 500 of them at 95 today. What's the paper P&L on the remaining shares?",
          "1000 x (83.31 - 92) = -8,690 (not 1500 x -8.69 = -13,035)", "en",
          required_facts=[fact("pnl", p(BYD, "close"), k=1000, cost=92.0, dp=0)], required_entity=BYD,
          wrong_values=[wrong("pnl", p(BYD, "close"), k=1500, cost=92.0, dp=0, unit="plain")]))
task("r14h_sale_zh_01", "sale_followup", "zh",
     turn("我有紫金矿业3000股，成本27", "a stated position: the close 29.79 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(ZIJIN, "close"), 2)], required_entity=ZIJIN),
     turn("今天在31卖了1000股，剩下的浮盈多少", "2000 x (29.79 - 27) = +5,580 (not 3000 x 2.79 = 8,370)", "zh",
          required_facts=[fact("pnl", p(ZIJIN, "close"), k=2000, cost=27.0, dp=0)], required_entity=ZIJIN,
          wrong_values=[wrong("pnl", p(ZIJIN, "close"), k=3000, cost=27.0, dp=0, unit="plain")]))
task("r14h_sale_en_02", "sale_followup", "en",
     turn("I hold 1,000 Ping An at 50.", "a stated position: the close 58.03 at least", "en", required_tools=PRICE,
          required_facts=[V(p(PINGAN, "close"), 2)], required_entity=PINGAN),
     turn("Offloaded a quarter of them at 60 - and the shares I kept, where do they stand now?", "750 x (58.03 - 50) = +6,022.5 (not 1000 x 8.03 = 8,030)",
          "en", required_facts=[fact("pnl", p(PINGAN, "close"), k=750, cost=50.0, dp=1)], required_entity=PINGAN,
          wrong_values=[wrong("pnl", p(PINGAN, "close"), k=1000, cost=50.0, dp=0, unit="plain")]))

# ---------------------------------------------------------------- (f) an unlisted subsidiary or business named with a covered short name
task("r14h_alike_zh_01", "lookalike_unlisted", "zh",
     turn("中芯北方的营收", "中芯北方 (an unlisted SMIC subsidiary) is not 中芯国际: out of coverage", "zh", **OOC))
task("r14h_alike_zh_02", "lookalike_unlisted", "zh",
     turn("比亚迪锂电池的净利润", "比亚迪锂电池 (FinDreams Battery, unlisted) is not 比亚迪: out of coverage", "zh", **OOC))
task("r14h_alike_zh_03", "lookalike_unlisted", "zh",
     turn("平安理财的净利润", "平安理财 (an unlisted wealth-management subsidiary) is not 中国平安 / 平安银行: out of coverage",
          "zh", **OOC))
task("r14h_alike_zh_04", "lookalike_unlisted", "zh",
     turn("宁德邦普的营收", "宁德邦普 (Brunp, an unlisted CATL subsidiary) is not 宁德时代: out of coverage", "zh", **OOC))
task("r14h_alike_en_01", "lookalike_unlisted", "en",
     turn("Ping An Technology revenue", "Ping An Technology (unlisted) is not Ping An Insurance: out of coverage", "en", **OOC))

# ---------------------------------------------------------------- (g) Hong Kong financial vocabulary in traditional characters
task("r14h_hkterm_zh_01", "hk_terms", "zh",
     turn("中國平安純利", "純利 = 净利润: 1347.78亿", "zh", required_tools=FUND,
          required_facts=[V(f(PINGAN, "net_profit"), -6)], required_entity=PINGAN))
task("r14h_hkterm_zh_02", "hk_terms", "zh",
     turn("長江電力股東應佔溢利", "股東應佔溢利 = 归母净利润: 345.03亿", "zh", required_tools=FUND,
          required_facts=[V(f(CYPC, "net_profit"), -6)], required_entity=CYPC))
task("r14h_hkterm_zh_03", "hk_terms", "zh",
     turn("中信證券股本回報率", "股本回報率 = ROE: 9.4", "zh", required_tools=FUND,
          required_facts=[V(f(CITIC, "roe"), 2)], required_entity=CITIC))
task("r14h_hkterm_zh_04", "hk_terms", "zh",
     turn("招商銀行的收市價", "收市價 = 收盘价: 41.26", "zh", required_tools=PRICE,
          required_facts=[V(p(CMB, "close"), 2)], required_entity=CMB))

# ---------------------------------------------------------------- (h) full-width Latin letters
task("r14h_width_zh_01", "fullwidth_latin", "zh",
     turn("长江电力的ＲＯＥ", "ＲＯＥ = ROE: 15.59", "zh", required_tools=FUND, required_facts=[V(f(CYPC, "roe"), 2)],
          required_entity=CYPC))
task("r14h_width_zh_02", "fullwidth_latin", "zh",
     turn("美的的ＰＢ", "ＰＢ = PB: 2.89", "zh", required_tools=FUND, required_facts=[V(f(MIDEA, "pb"), 2)],
          required_entity=MIDEA))
task("r14h_width_en_01", "fullwidth_latin", "en",
     turn("ＣＭＢ close price", "ＣＭＢ = CMB: 41.26", "en", required_tools=PRICE, required_facts=[V(p(CMB, "close"), 2)],
          required_entity=CMB))

# ---------------------------------------------------------------- (i) a correction of the company in the next turn keeps the asked metric
task("r14h_fix_zh_01", "correction_turn", "zh",
     turn("招行的市净率", "0.91", "zh", required_tools=FUND, required_facts=[V(f(CMB, "pb"), 2)], required_entity=CMB),
     turn("说错了，我想问的是工行", "the P/B of ICBC: 0.74", "zh", required_facts=[V(f(ICBC, "pb"), 2)], required_entity=ICBC))
task("r14h_fix_zh_02", "correction_turn", "zh",
     turn("比亚迪营收多少", "8039.65亿", "zh", required_tools=FUND, required_facts=[V(f(BYD, "revenue"), -6)],
          required_entity=BYD),
     turn("不对不对，是美的", "Midea's revenue: 4585.02亿", "zh", required_facts=[V(f(MIDEA, "revenue"), -6)],
          required_entity=MIDEA))
task("r14h_fix_en_01", "correction_turn", "en",
     turn("What's Moutai's ROE?", "32.53", "en", required_tools=FUND, required_facts=[V(f(MOUTAI, "roe"), 2)],
          required_entity=MOUTAI),
     turn("Sorry, I meant Wuliangye.", "Wuliangye's ROE: 6.89", "en", required_facts=[V(f(WLY, "roe"), 2)],
          required_entity=WLY))

# ---------------------------------------------------------------- (j) a short position gains when the price falls
task("r14h_short_zh_01", "short_position", "zh",
     turn("我融券做空了中信证券1000股，开仓价28", "a stated short: the close 25.86 at least", "zh", required_tools=PRICE,
          required_facts=[V(p(CITIC, "close"), 2)], required_entity=CITIC),
     turn("现在这笔是赚是亏，多少", "short: 1000 x (28 - 25.86) = +2,140, a gain", "zh",
          required_facts=[fact("short_pnl", p(CITIC, "close"), k=1000, cost=28.0, dp=0)], required_entity=CITIC,
          extra_forbidden=["(?:亏|浮亏|亏损)[^。；;]{0,12}2,?140"]))
task("r14h_short_en_01", "short_position", "en",
     turn("I shorted 500 BYD at 95 on margin.", "a stated short: the close 83.31 at least", "en", required_tools=PRICE,
          required_facts=[V(p(BYD, "close"), 2)], required_entity=BYD),
     turn("Where does that trade stand now?", "short: 500 x (95 - 83.31) = +5,845, a gain", "en",
          required_facts=[fact("short_pnl", p(BYD, "close"), k=500, cost=95.0, dp=0)], required_entity=BYD,
          extra_forbidden=["(?i)(?:loss|down|lost|losing)[^.;]{0,24}5,?845"]))

# ---------------------------------------------------------------- (k) one holding's share of a two-stock portfolio
task("r14h_weight_zh_01", "portfolio_weight", "zh",
     turn("我持有工行5000股、长江电力1000股，长江电力占这两只总市值的百分之几",
          "28,540 / (28,540 + 41,400) = 40.81%", "zh", required_tools=PRICE,
          required_facts=[fact("weight", terms=[[1000, p(CYPC, "close")], [5000, p(ICBC, "close")]], dp=2)],
          required_entities=[ICBC, CYPC]))
task("r14h_weight_en_01", "portfolio_weight", "en",
     turn("I own 100 Moutai and 1,000 CMB — what share of the total is Moutai?",
          "140,950 / (140,950 + 41,260) = 77.36%", "en", required_tools=PRICE,
          required_facts=[fact("weight", terms=[[100, p(MOUTAI, "close")], [1000, p(CMB, "close")]], dp=2)],
          required_entities=[MOUTAI, CMB]))

# ---------------------------------------------------------------- (l) the 52-week closing range: its width and the close's place in it
task("r14h_range_zh_01", "range_52w", "zh",
     turn("紫金矿业52周最高和最低收盘价相差多少", "43.45 - 25.1 = 18.35", "zh", required_tools=PRICE,
          required_facts=[fact("spread", refs=[p(ZIJIN, "range_52w.high"), p(ZIJIN, "range_52w.low")], dp=2)],
          required_entity=ZIJIN))
task("r14h_range_zh_02", "range_52w", "zh",
     turn("比亚迪现价在52周区间里大概处在百分之多少的位置", "(83.31 - 78.2) / (110.6 - 78.2) = 15.77%", "zh",
          required_tools=PRICE,
          required_facts=[fact("range_pos", p(BYD, "close"), lo=p(BYD, "range_52w.low"), hi=p(BYD, "range_52w.high"), dp=2)],
          required_entity=BYD))
task("r14h_range_en_01", "range_52w", "en",
     turn("How wide was CMB's 52-week closing range?", "43.45 - 35.5 = 7.95", "en", required_tools=PRICE,
          required_facts=[fact("spread", refs=[p(CMB, "range_52w.high"), p(CMB, "range_52w.low")], dp=2)],
          required_entity=CMB))

# ---------------------------------------------------------------- (m) market nicknames
task("r14h_nick_zh_01", "nickname", "zh",
     turn("宇宙行2025年的归母净利润是多少", "宇宙行 = 工商银行: 3685.62亿", "zh", required_tools=FUND,
          required_facts=[V(f(ICBC, "net_profit"), -6)], required_entity=ICBC))
task("r14h_nick_zh_02", "nickname", "zh",
     turn("平保的市盈率", "平保 = 中国平安: 7.8", "zh", required_tools=FUND, required_facts=[V(f(PINGAN, "pe_ttm"), 2)],
          required_entity=PINGAN))

# ---------------------------------------------------------------- (n) figures derived from two fields of the data
task("r14h_derive_zh_01", "derived_figure", "zh",
     turn("按市值和股价算，茅台一共有多少股", "17650.75亿 / 1409.5 = 12.52亿股", "zh", required_tools=FUND,
          required_facts=[fact("div", f(MOUTAI, "total_mv"), p(MOUTAI, "close"), dp=-6)], required_entity=MOUTAI))
task("r14h_derive_zh_02", "derived_figure", "zh",
     turn("工行一年净利润摊到每天大概多少", "3685.62亿 / 365 = 10.10亿", "zh", required_tools=FUND,
          required_facts=[fact("mul", f(ICBC, "net_profit"), k=1 / 365, dp=-6)], required_entity=ICBC))

# ---------------------------------------------------------------- (o) controls (forms the round-17 rules cover)
task("r14h_ctrl_zh_01", "control", "zh",
     turn("我在1500元买了100股茅台，1600元卖了其中50股，剩下的浮亏多少", "50 x (1409.5 - 1500) = -4,525", "zh",
          required_tools=PRICE, required_facts=[fact("pnl", p(MOUTAI, "close"), k=50, cost=1500.0, dp=0)],
          required_entity=MOUTAI))
task("r14h_ctrl_zh_02", "control", "zh",
     turn("宁德时代和比亚迪的营收合计", "4237.02 + 8039.65 = 12276.67亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(CATL, "revenue"), f(BYD, "revenue"), dp=-6)], required_entities=[CATL, BYD]))
task("r14h_ctrl_zh_03", "control", "zh",
     turn("紫金矿业的市盈率", "11.71", "zh", required_tools=FUND, required_facts=[V(f(ZIJIN, "pe_ttm"), 2)],
          required_entity=ZIJIN))
task("r14h_ctrl_zh_04", "control", "zh",
     turn("美的集团75元买了400股，80元又补了400股，85元卖出300股，剩余部分账面盈亏",
          "avg 77.5; 500 x (80.1 - 77.5) = +1,300", "zh", required_tools=PRICE,
          required_facts=[fact("pnl_avg", p(MIDEA, "close"), lots=[[400, 75.0], [400, 80.0]], sold=300, dp=0)],
          required_entity=MIDEA))
task("r14h_ctrl_zh_05", "control", "zh",
     turn("长江电力的毛利率", "61.67", "zh", required_tools=FUND, required_facts=[V(f(CYPC, "grossprofit_margin"), 2)],
          required_entity=CYPC))
task("r14h_ctrl_zh_06", "control", "zh",
     turn("中信证券要回到52周最高收盘价还要涨多少", "(30.3 - 25.86) / 25.86 = 17.17%", "zh", required_tools=PRICE,
          required_facts=[fact("rel", p(CITIC, "range_52w.high"), p(CITIC, "close"), dp=2)], required_entity=CITIC))
task("r14h_ctrl_en_01", "control", "en",
     turn("I bought 500 CATL at 260, added another 500 at 300, then sold 600 at 310. P&L on what's left?",
          "avg 280; 400 x (291.11 - 280) = +4,444", "en", required_tools=PRICE,
          required_facts=[fact("pnl_avg", p(CATL, "close"), lots=[[500, 260.0], [500, 300.0]], sold=600, dp=0)],
          required_entity=CATL))

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


def _near(value: float, target: float, rel: float = 0.05) -> bool:
    return abs(value - target) <= rel * abs(target)


# ---- (A) one company of a named group excluded from a statement about the rest
claim("r14c01", "zh", "exclusion_claim", "在招行、工行和平安银行三家中，除平安银行以外，营收都实现了同比增长", [
    ("招行 revenue_yoy > 0", lambda: yoy(CMB, "revenue_yoy") > 0),
    ("工行 revenue_yoy > 0", lambda: yoy(ICBC, "revenue_yoy") > 0),
])
claim("r14c02", "zh", "exclusion_claim", "宁德时代、比亚迪、美的三家里，除比亚迪外净利润同比均增长超一成", [
    ("宁德 netprofit_yoy > 10", lambda: yoy(CATL, "netprofit_yoy") > 10),
    ("美的 netprofit_yoy > 10", lambda: yoy(MIDEA, "netprofit_yoy") > 10),
])
claim("r14c03", "zh", "exclusion_claim", "除工商银行外，招行与平安银行净利润都在下降", [
    ("招行 netprofit_yoy < 0 (+1.05: false)", lambda: yoy(CMB, "netprofit_yoy") < 0),
    ("平安银行 netprofit_yoy < 0", lambda: yoy(PAB, "netprofit_yoy") < 0),
])
claim("r14c04", "en", "exclusion_claim", "Among CMB, ICBC and Ping An Bank, all but Ping An Bank grew net profit last year", [
    ("CMB netprofit_yoy > 0", lambda: yoy(CMB, "netprofit_yoy") > 0),
    ("ICBC netprofit_yoy > 0", lambda: yoy(ICBC, "netprofit_yoy") > 0),
])

# ---- (B) Hong Kong financial vocabulary in traditional characters
claim("r14c05", "zh", "hk_terms_claim", "比亞迪2025年純利326.19億元人民幣", [
    ("比亚迪 net_profit = 326.19亿", lambda: abs(_yi(f(BYD, "net_profit")) - 326.19) <= 0.005),
])
claim("r14c06", "zh", "hk_terms_claim", "長江電力純利按年增長約6%", [
    ("长江电力 netprofit_yoy ≈ 6 (within 5%)", lambda: _near(yoy(CYPC, "netprofit_yoy"), 6.0)),
])
claim("r14c07", "zh", "hk_terms_claim", "中信證券股東應佔溢利約300.8億元", [
    ("中信 net_profit ≈ 300.8亿", lambda: _near(_yi(f(CITIC, "net_profit")), 300.8)),
])
claim("r14c08", "zh", "hk_terms_claim", "工商銀行市賬率0.74倍", [
    ("工行 PB = 0.74", lambda: abs(field(f(ICBC, "pb")) - 0.74) <= 0.005),
])
claim("r14c09", "zh", "hk_terms_claim", "紫金礦業純利按年增長逾八成", [
    ("紫金 netprofit_yoy > 80 (62.01: false)", lambda: yoy(ZIJIN, "netprofit_yoy") > 80),
])

# ---- (C) two companies' total against a third
claim("r14c10", "zh", "sum_vs_single", "招商银行与平安银行净利润之和不及工商银行一家", [
    ("招行 + 平安银行 net_profit < 工行", lambda: field(f(CMB, "net_profit")) + field(f(PAB, "net_profit")) < field(f(ICBC, "net_profit"))),
])
claim("r14c11", "zh", "sum_vs_single", "比亚迪和中芯国际的营收加起来超过了中国平安一家", [
    ("比亚迪 + 中芯 revenue > 平安 (8712.88 < 10505.06: false)",
     lambda: field(f(BYD, "revenue")) + field(f(SMIC, "revenue")) > field(f(PINGAN, "revenue"))),
])
claim("r14c12", "zh", "sum_vs_single", "五粮液、紫金矿业两家净利润合计不足茅台", [
    ("五粮液 + 紫金 net_profit < 茅台", lambda: field(f(WLY, "net_profit")) + field(f(ZIJIN, "net_profit")) < field(f(MOUTAI, "net_profit"))),
])

# ---- (D) a decline (or rise) compared with another metric of the same company in words
claim("r14c13", "zh", "relative_change", "平安银行营收同比下降10.4%，净利润降幅相对较小", [
    ("平安银行 revenue_yoy = -10.4", lambda: abs(yoy(PAB, "revenue_yoy") + 10.4) <= 0.05),
    ("平安银行 |netprofit_yoy| < |revenue_yoy|, both falling", lambda: 0 > yoy(PAB, "netprofit_yoy") > yoy(PAB, "revenue_yoy")),
])
claim("r14c14", "zh", "relative_change", "五粮液净利润降幅小于营收降幅", [
    ("五粮液 |netprofit_yoy| < |revenue_yoy| (71.89 > 54.55: false)",
     lambda: abs(yoy(WLY, "netprofit_yoy")) < abs(yoy(WLY, "revenue_yoy"))),
])
claim("r14c15", "zh", "relative_change", "紫金矿业营收增长约15%，净利润增幅相对较小", [
    ("紫金 revenue_yoy ≈ 15 (within 5%)", lambda: _near(yoy(ZIJIN, "revenue_yoy"), 15.0)),
    ("紫金 netprofit_yoy < revenue_yoy (62.01 > 14.96: false)", lambda: yoy(ZIJIN, "netprofit_yoy") < yoy(ZIJIN, "revenue_yoy")),
])

# ---- (E) a round threshold written as a unit word
claim("r14c16", "zh", "unit_threshold", "宁德时代市值站上万亿", [
    ("宁德 total_mv > 10000亿", lambda: _yi(f(CATL, "total_mv")) > 10000),
])
claim("r14c17", "zh", "unit_threshold", "长江电力市值突破八千亿", [
    ("长江电力 total_mv > 8000亿 (6983.23: false)", lambda: _yi(f(CYPC, "total_mv")) > 8000),
])
claim("r14c18", "zh", "unit_threshold", "美的集团市值突破六千亿", [
    ("美的 total_mv > 6000亿", lambda: _yi(f(MIDEA, "total_mv")) > 6000),
])

# ---- (F) controls
claim("r14c19", "zh", "control", "美的集团市盈率23.5倍", [
    ("美的 PE = 23.5 (13.77: false)", lambda: abs(field(f(MIDEA, "pe_ttm")) - 23.5) <= 0.005),
])
claim("r14c20", "zh", "control", "长江电力ROE为25%", [
    ("长江电力 ROE = 25 (15.59: false)", lambda: abs(field(f(CYPC, "roe")) - 25) <= 0.005),
])
claim("r14c21", "zh", "control", "中信证券营收同比增长28.8%", [
    ("中信 revenue_yoy = 28.8", lambda: abs(yoy(CITIC, "revenue_yoy") - 28.8) <= 0.05),
])
claim("r14c22", "zh", "control", "紫金矿业市净率3.92倍", [
    ("紫金 PB = 3.92", lambda: abs(field(f(ZIJIN, "pb")) - 3.92) <= 0.005),
])
claim("r14c23", "zh", "control", "贵州茅台2025年营业收入1688.38亿元", [
    ("茅台 revenue = 1688.38亿", lambda: abs(_yi(f(MOUTAI, "revenue")) - 1688.38) <= 0.005),
])
claim("r14c24", "zh", "control", "招商银行ROE为11.8%", [
    ("招行 ROE = 11.8", lambda: abs(field(f(CMB, "roe")) - 11.8) <= 0.005),
])


if __name__ == "__main__":
    out = HERE / "chat_r14_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for t in TASKS:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} conversations ({sum(len(t['turns']) for t in TASKS)} turns) to {out}")
    out = HERE / "claims_r14_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for c in CLAIMS:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(CLAIMS)} claims to {out}")
