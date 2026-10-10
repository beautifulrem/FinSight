"""Build chat_r11_heldout.jsonl and claims_r11_heldout.jsonl from the real offline FinSight tools.

Round-11 independent held-out slice, written by the round-11 reviewer at 4ad3380 before any round-15 fix. The
engineers do not open it; it is run once before (here, first_run/) and once after their fixes.

It uses the SHIPPED offline default: the snapshot extension (data/snapshot/structured_data_ext.json) AND the
corrections layer (data/snapshot/structured_data_fix.json). Run the chat slice with
``QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`` (the runner is pinned to the v1 seed otherwise).

Every number in a chat ``required_facts`` entry is computed here from tool output (never typed in): each fact carries
a ``derive`` spec over ``<evidence_id>:<field>`` operands (dotted fields reach nested payloads, e.g.
``price_300750.SZ:range_52w.high``), rounded to ``dp`` decimals. Every claim's expected verdict comes from the
predicates in ``claim(...)`` (all parts true -> supported, all false -> contradicted, mixed -> partially_supported).
``verify_heldout_r11.py`` re-runs the tools and re-checks both files with its own implementation of every op.

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python build_heldout_r11.py
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
    """Exact value of a derive spec (verify_heldout_r11.py has its own implementation)."""
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


# ---------------------------------------------------------------- (a) totals asked with words outside the usual set
task("r11h_sum_zh_01", "sum_words", "zh",
     turn("长江电力和美的集团的营收拢共多少", "862.42+4585.02 = 5447.44亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(CYPC, "revenue"), f(MIDEA, "revenue"), dp=-6)], required_entities=[CYPC, MIDEA]))
task("r11h_sum_zh_02", "sum_words", "zh",
     turn("中信证券、紫金矿业净利润统共多少亿", "300.76+517.77 = 818.53亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(CITIC, "net_profit"), f(ZIJIN, "net_profit"), dp=-6)],
          required_entities=[CITIC, ZIJIN]))
task("r11h_sum_zh_03", "sum_words", "zh",
     turn("平安银行跟招商银行的营收合并算是多少", "1314.42+3375.32 = 4689.74亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(PAB, "revenue"), f(CMB, "revenue"), dp=-6)], required_entities=[PAB, CMB]))
task("r11h_sum_en_01", "sum_words", "en",
     turn("What do Midea and Zijin Mining earn jointly?", "439.45+517.77 = 957.22亿 (95.72 bn)", "en",
          required_tools=FUND, required_facts=[fact("sum", f(MIDEA, "net_profit"), f(ZIJIN, "net_profit"), dp=-6)],
          required_entities=[MIDEA, ZIJIN]))
task("r11h_sum_zh_04", "sum_words", "zh",
     turn("比亚迪去年净利润", "326.19亿", "zh", required_tools=FUND, required_facts=[V(f(BYD, "net_profit"), -6)],
          required_entity=BYD),
     turn("美的的呢", "ellipsis: 439.45亿", "zh", required_tools=FUND, required_facts=[V(f(MIDEA, "net_profit"), -6)],
          required_entity=MIDEA),
     turn("它俩一起挣了多少", "326.19+439.45 = 765.64亿", "zh",
          required_facts=[fact("sum", f(BYD, "net_profit"), f(MIDEA, "net_profit"), dp=-6)],
          required_entities=[BYD, MIDEA]))

# ---------------------------------------------------------------- (b) the mean of several operands
task("r11h_mean_zh_01", "mean", "zh",
     turn("工商银行、招商银行、中信证券的ROE平均下来是多少", "(8.68+11.8+9.4)/3 = 9.96", "zh", required_tools=FUND,
          required_facts=[fact("mean", refs=[f(ICBC, "roe"), f(CMB, "roe"), f(CITIC, "roe")], dp=2)],
          required_entities=[ICBC, CMB, CITIC]))
task("r11h_mean_zh_02", "mean", "zh",
     turn("宁德时代、美的集团两家市净率的均值", "(3.55+2.89)/2 = 3.22", "zh", required_tools=FUND,
          required_facts=[fact("mean", refs=[f(CATL, "pb"), f(MIDEA, "pb")], dp=2)], required_entities=[CATL, MIDEA]))
task("r11h_mean_en_01", "mean", "en",
     turn("Mean P/E across Zijin Mining, Midea and CITIC Securities?", "(11.71+13.77+10.19)/3 = 11.89", "en",
          required_tools=FUND,
          required_facts=[fact("mean", refs=[f(ZIJIN, "pe_ttm"), f(MIDEA, "pe_ttm"), f(CITIC, "pe_ttm")], dp=2)],
          required_entities=[ZIJIN, MIDEA, CITIC]))
task("r11h_mean_zh_03", "mean", "zh",
     turn("长江电力的毛利率", "61.67%", "zh", required_tools=FUND, required_facts=[V(f(CYPC, "grossprofit_margin"), 2)],
          required_entity=CYPC),
     turn("紫金矿业的呢", "ellipsis: 27.73%", "zh", required_tools=FUND,
          required_facts=[V(f(ZIJIN, "grossprofit_margin"), 2)], required_entity=ZIJIN),
     turn("两家平均一下是多少", "(61.67+27.73)/2 = 44.70", "zh",
          required_facts=[fact("mean", refs=[f(CYPC, "grossprofit_margin"), f(ZIJIN, "grossprofit_margin")], dp=2)],
          required_entities=[CYPC, ZIJIN]))
task("r11h_mean_zh_04", "mean", "zh",
     turn("比亚迪、宁德时代、中芯国际的营收先加起来再除以3", "(8039.65+4237.02+673.23)/3 = 4316.63亿", "zh",
          required_tools=FUND,
          required_facts=[fact("mean", refs=[f(BYD, "revenue"), f(CATL, "revenue"), f(SMIC, "revenue")], dp=-6)],
          required_entities=[BYD, CATL, SMIC]))

# ---------------------------------------------------------------- (c) one value against the total of others
task("r11h_vsum_en_01", "sum_vs_one", "en",
     turn("Is CATL's net profit bigger than BYD's and Ping An Bank's put together, and by how much?",
          "722.01 vs 326.19+426.33 = 752.52: no, short by 30.51亿", "en", required_tools=FUND,
          required_facts=[fact("diff_sum", f(CATL, "net_profit"), refs=[f(BYD, "net_profit"), f(PAB, "net_profit")], dp=-6)],
          required_entities=[CATL, BYD, PAB]))
task("r11h_vsum_zh_01", "sum_vs_one", "zh",
     turn("工商银行一家的营收，跟招商银行、比亚迪两家合起来比，是多是少，差多少", "8382.70 vs 11414.97: 3032.27亿 less", "zh",
          required_tools=FUND,
          required_facts=[fact("diff_sum", f(ICBC, "revenue"), refs=[f(CMB, "revenue"), f(BYD, "revenue")], dp=-6)],
          required_entities=[ICBC, CMB, BYD]))
task("r11h_vsum_zh_02", "sum_vs_one", "zh",
     turn("中信证券和平安银行的净利润加在一起，能超过招商银行吗？差多少", "727.09 vs 1501.81: 774.72亿 short", "zh",
          required_tools=FUND,
          required_facts=[fact("diff_sum", f(CMB, "net_profit"), refs=[f(CITIC, "net_profit"), f(PAB, "net_profit")], dp=-6)],
          required_entities=[CITIC, PAB, CMB]))
task("r11h_vsum_en_02", "sum_vs_one", "en",
     turn("Do Zijin Mining and China Yangtze Power together have a larger market cap than CATL? What's the difference?",
          "7921.40+6983.23 = 14904.63 vs 13470.39: 1434.24亿 larger", "en", required_tools=FUND,
          required_facts=[fact("diff_sum", f(CATL, "total_mv"), refs=[f(ZIJIN, "total_mv"), f(CYPC, "total_mv")], dp=-6)],
          required_entities=[ZIJIN, CYPC, CATL]))

# ---------------------------------------------------------------- (d) one operand's share of a three-operand total
task("r11h_stot3_zh_01", "share_of_total3", "zh",
     turn("平安银行营收在平安、招行、工行三家营收总和里占百分之几", "1314.42/13072.44 = 10.05%", "zh", required_tools=FUND,
          required_facts=[fact("share_total", refs=[f(PAB, "revenue"), f(CMB, "revenue"), f(ICBC, "revenue")], dp=2)],
          required_entities=[PAB, CMB, ICBC]))
task("r11h_stot3_zh_02", "share_of_total3", "zh",
     turn("紫金矿业、美的集团、比亚迪三家总市值合计中，紫金占多大比重", "7921.40/21628.52 = 36.62%", "zh", required_tools=FUND,
          required_facts=[fact("share_total", refs=[f(ZIJIN, "total_mv"), f(MIDEA, "total_mv"), f(BYD, "total_mv")], dp=2)],
          required_entities=[ZIJIN, MIDEA, BYD]))
task("r11h_stot3_en_01", "share_of_total3", "en",
     turn("Of the combined revenue of CITIC Securities, Zijin Mining and Midea, what share is CITIC's?",
          "748.54/8824.35 = 8.48%", "en", required_tools=FUND,
          required_facts=[fact("share_total", refs=[f(CITIC, "revenue"), f(ZIJIN, "revenue"), f(MIDEA, "revenue")], dp=2)],
          required_entities=[CITIC, ZIJIN, MIDEA]))

# ---------------------------------------------------------------- (e) a total and a share asked in one message
task("r11h_sumshare_zh_01", "sum_and_share", "zh",
     turn("长江电力净利润", "345.03亿", "zh", required_tools=FUND, required_facts=[V(f(CYPC, "net_profit"), -6)],
          required_entity=CYPC),
     turn("中信证券呢", "ellipsis: 300.76亿", "zh", required_tools=FUND, required_facts=[V(f(CITIC, "net_profit"), -6)],
          required_entity=CITIC),
     turn("两家合计多少，长江电力占了几成", "645.79亿; 345.03/645.79 = 53.43%", "zh",
          required_facts=[fact("sum", f(CYPC, "net_profit"), f(CITIC, "net_profit"), dp=-6),
                          fact("share_total", refs=[f(CYPC, "net_profit"), f(CITIC, "net_profit")], dp=2)],
          required_entities=[CYPC, CITIC]))
task("r11h_sumshare_en_01", "sum_and_share", "en",
     turn("Midea's revenue?", "4585.02亿", "en", required_tools=FUND, required_facts=[V(f(MIDEA, "revenue"), -7)],
          required_entity=MIDEA),
     turn("And BYD's?", "8039.65亿", "en", required_tools=FUND, required_facts=[V(f(BYD, "revenue"), -7)],
          required_entity=BYD),
     turn("Total of the two, and Midea's share of that total?", "12624.67亿; 36.32%", "en",
          required_facts=[fact("sum", f(MIDEA, "revenue"), f(BYD, "revenue"), dp=-6),
                          fact("share_total", refs=[f(MIDEA, "revenue"), f(BYD, "revenue")], dp=2)],
          required_entities=[MIDEA, BYD]))

# ---------------------------------------------------------------- (f) an ellipsis naming two new targets
task("r11h_ell2_zh_01", "ellipsis_two", "zh",
     turn("中信证券的市净率", "1.37", "zh", required_tools=FUND, required_facts=[V(f(CITIC, "pb"), 2)], required_entity=CITIC),
     turn("招行和紫金的呢", "PB 0.91 and 3.92", "zh", required_tools=FUND,
          required_facts=[V(f(CMB, "pb"), 2), V(f(ZIJIN, "pb"), 2)], required_entities=[CMB, ZIJIN]))
task("r11h_ell2_en_01", "ellipsis_two", "en",
     turn("What's Midea's ROE?", "19.69", "en", required_tools=FUND, required_facts=[V(f(MIDEA, "roe"), 2)],
          required_entity=MIDEA),
     turn("And BYD's and CATL's?", "13.24 and 21.42", "en", required_tools=FUND,
          required_facts=[V(f(BYD, "roe"), 2), V(f(CATL, "roe"), 2)], required_entities=[BYD, CATL]))

# ---------------------------------------------------------------- (g) a holding carried into a follow-up that adds one
task("r11h_holdfu_zh_01", "holding_follow_up", "zh",
     turn("我有800股长江电力，值多少", "800x28.54 = 22,832", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(CYPC, "close"), k=800, dp=0)], required_entity=CYPC),
     turn("再加2手紫金矿业，一共多少", "22,832 + 200x29.79 = 28,790", "zh",
          required_facts=[fact("lin", terms=[[800, p(CYPC, "close")], [200, p(ZIJIN, "close")]], dp=0)],
          required_entities=[CYPC, ZIJIN]))
task("r11h_holdfu_en_01", "holding_follow_up", "en",
     turn("I hold 300 Midea shares, what's that worth?", "300x80.1 = 24,030", "en", required_tools=PRICE,
          required_facts=[fact("mul", p(MIDEA, "close"), k=300, dp=0)], required_entity=MIDEA),
     turn("Add 500 ICBC shares to that. Total now?", "24,030 + 500x8.28 = 28,170", "en",
          required_facts=[fact("lin", terms=[[300, p(MIDEA, "close")], [500, p(ICBC, "close")]], dp=0)],
          required_entities=[MIDEA, ICBC]))
task("r11h_holdfu_zh_02", "holding_follow_up", "zh",
     turn("持有中信证券1200股，市值多少", "1200x25.86 = 31,032", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(CITIC, "close"), k=1200, dp=0)], required_entity=CITIC),
     turn("另有平安银行3手，两只合起来呢", "31,032 + 300x11.57 = 34,503", "zh",
          required_facts=[fact("lin", terms=[[1200, p(CITIC, "close")], [300, p(PAB, "close")]], dp=0)],
          required_entities=[CITIC, PAB]))

# ---------------------------------------------------------------- (h) holdings in other written forms
task("r11h_holdform_zh_01", "holding_forms", "zh",
     turn("我有2.5万股平安银行，按收盘价市值多少", "25000x11.57 = 289,250", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(PAB, "close"), k=25000, dp=0)], required_entity=PAB))
task("r11h_holdform_zh_02", "holding_forms", "zh",
     turn("持仓：美的集团 600 股，紫金矿业 1,200 股。合计市值？", "600x80.1 + 1200x29.79 = 83,808", "zh",
          required_tools=PRICE,
          required_facts=[fact("lin", terms=[[600, p(MIDEA, "close")], [1200, p(ZIJIN, "close")]], dp=0)],
          required_entities=[MIDEA, ZIJIN]))
task("r11h_holdform_zh_03", "holding_forms", "zh",
     turn("三千股中信证券加五百股比亚迪，市值总共是多少", "3000x25.86 + 500x83.31 = 119,235", "zh", required_tools=PRICE,
          required_facts=[fact("lin", terms=[[3000, p(CITIC, "close")], [500, p(BYD, "close")]], dp=0)],
          required_entities=[CITIC, BYD]))
task("r11h_holdform_en_01", "holding_forms", "en",
     turn("My portfolio: 1,500 CMB and 2,000 ICBC shares. Total value?", "1500x41.26 + 2000x8.28 = 78,450", "en",
          required_tools=PRICE,
          required_facts=[fact("lin", terms=[[1500, p(CMB, "close")], [2000, p(ICBC, "close")]], dp=0)],
          required_entities=[CMB, ICBC]))

# ---------------------------------------------------------------- (i) gain or loss over several purchases
task("r11h_avgcost_zh_01", "multi_purchase_pnl", "zh",
     turn("美的集团我分两次买：第一次300股75元，第二次200股85元，现在盈亏多少", "500x80.1 - (22500+17000) = +550", "zh",
          required_tools=PRICE, required_facts=[fact("pnl_multi", p(MIDEA, "close"), lots=[[300, 75.0], [200, 85.0]], dp=0)],
          required_entity=MIDEA))
task("r11h_avgcost_en_01", "multi_purchase_pnl", "en",
     turn("I bought 100 CATL shares at 300 and another 100 at 280. Gain or loss now?", "200x291.11 - 58000 = +222", "en",
          required_tools=PRICE,
          required_facts=[fact("pnl_multi", p(CATL, "close"), lots=[[100, 300.0], [100, 280.0]], dp=0)],
          required_entity=CATL))
task("r11h_avgcost_zh_02", "multi_purchase_pnl", "zh",
     turn("紫金矿业先30元买1000股，后来又25元补了1000股，按现价赚还是亏，多少", "2000x29.79 - 55000 = +4,580", "zh",
          required_tools=PRICE,
          required_facts=[fact("pnl_multi", p(ZIJIN, "close"), lots=[[1000, 30.0], [1000, 25.0]], dp=0)],
          required_entity=ZIJIN))

# ---------------------------------------------------------------- (j) price-derived: the rise needed to get back to a high
task("r11h_recover_zh_01", "price_recovery", "zh",
     turn("比亚迪股价回到52周最高收盘价还需要上涨百分之几", "(110.6-83.31)/83.31 = 32.76%", "zh", required_tools=PRICE,
          required_facts=[fact("rel", p(BYD, "range_52w.high"), p(BYD, "close"), dp=2)], required_entity=BYD))
task("r11h_recover_en_01", "price_recovery", "en",
     turn("What percent gain would take SMIC back to its 52-week closing high?", "(173-111.99)/111.99 = 54.48%", "en",
          required_tools=PRICE, required_facts=[fact("rel", p(SMIC, "range_52w.high"), p(SMIC, "close"), dp=2)],
          required_entity=SMIC))
task("r11h_recover_zh_02", "price_recovery", "zh",
     turn("美的集团现在比一年来的最低收盘价高出多少元", "80.1-71.81 = 8.29", "zh", required_tools=PRICE,
          required_facts=[fact("sub", p(MIDEA, "close"), p(MIDEA, "range_52w.low"), dp=2)], required_entity=MIDEA))

# ---------------------------------------------------------------- (k) market cap over net profit
task("r11h_mvnp_zh_01", "mv_over_profit", "zh",
     turn("长江电力的市值相当于几年的净利润", "6983.23/345.03 = 20.24", "zh", required_tools=FUND,
          required_facts=[fact("div", f(CYPC, "total_mv"), f(CYPC, "net_profit"), dp=2)], required_entity=CYPC))
task("r11h_mvnp_en_01", "mv_over_profit", "en",
     turn("BYD: market value over net profit, what multiple?", "7595.54/326.19 = 23.29", "en", required_tools=FUND,
          required_facts=[fact("div", f(BYD, "total_mv"), f(BYD, "net_profit"), dp=2)], required_entity=BYD))

# ---------------------------------------------------------------- (l) two questions in one message
task("r11h_twoq_zh_01", "two_questions", "zh",
     turn("宁德时代市值是美的的几倍？营收呢？", "2.20; 0.92", "zh", required_tools=FUND,
          required_facts=[fact("div", f(CATL, "total_mv"), f(MIDEA, "total_mv"), dp=2),
                          fact("div", f(CATL, "revenue"), f(MIDEA, "revenue"), dp=2)],
          required_entities=[CATL, MIDEA]))
task("r11h_twoq_en_01", "two_questions", "en",
     turn("How much higher is Zijin's ROE than ICBC's, and what about P/E?", "27.91-8.68 = 19.23; 11.71-7.89 = 3.82",
          "en", required_tools=FUND,
          required_facts=[fact("sub", f(ZIJIN, "roe"), f(ICBC, "roe"), dp=2),
                          fact("sub", f(ZIJIN, "pe_ttm"), f(ICBC, "pe_ttm"), dp=2)],
          required_entities=[ZIJIN, ICBC]))

# ---------------------------------------------------------------- (m) comparative verbs outside the lexicon
task("r11h_verb_zh_01", "verb_comparative", "zh",
     turn("美的集团去年净利润比比亚迪多挣出多少", "439.45-326.19 = 113.26亿", "zh", required_tools=FUND,
          required_facts=[fact("sub", f(MIDEA, "net_profit"), f(BYD, "net_profit"), dp=-6)],
          required_entities=[MIDEA, BYD]))
task("r11h_verb_en_01", "verb_comparative", "en",
     turn("By what margin does CMB's P/E exceed Ping An Bank's?", "6.86-5.17 = 1.69", "en", required_tools=FUND,
          required_facts=[fact("sub", f(CMB, "pe_ttm"), f(PAB, "pe_ttm"), dp=2)], required_entities=[CMB, PAB]))
task("r11h_verb_en_02", "verb_comparative", "en",
     turn("How much does ICBC's revenue eclipse CMB's by?", "8382.70-3375.32 = 5007.38亿 (500.74 bn)", "en",
          required_tools=FUND, required_facts=[fact("sub", f(ICBC, "revenue"), f(CMB, "revenue"), dp=-6)],
          required_entities=[ICBC, CMB]))

# ---------------------------------------------------------------- (n) a stated past date
task("r11h_date_zh_01", "dated_close", "zh",
     dated(turn("招商银行在2025年最后一个交易日收盘多少", "prior year-end close 42.1 (2025-12-31)", "zh", required_tools=PRICE,
                required_facts=[V(p(CMB, "prior_year_end.close"), 2)], required_entity=CMB),
           p(CMB, "prior_year_end.date"), "2025-12-31"))
task("r11h_date_en_01", "dated_close", "en",
     dated(turn("What did CATL close at on 2025-12-31?", "367.26", "en", required_tools=PRICE,
                required_facts=[V(p(CATL, "prior_year_end.close"), 2)], required_entity=CATL),
           p(CATL, "prior_year_end.date"), "2025-12-31"))
task("r11h_date_zh_02", "dated_close", "zh",
     dated(turn("比亚迪2026年6月26日的收盘价", "78.2 (the 52-week low bar)", "zh", required_tools=PRICE,
                required_facts=[V(p(BYD, "range_52w.low"), 2)], required_entity=BYD),
           p(BYD, "range_52w.low_date"), "2026-06-26"))
task("r11h_date_zh_03", "dated_close", "zh",
     after_data(turn("工商银行2026年10月9日收盘价是多少", "after the last bar (2026-09-30): state the day is not in the data",
                     "zh", required_tools=PRICE, must_state_missing=True, required_entity=ICBC),
                f"price_{ICBC}", "2026-10-09"))

# ---------------------------------------------------------------- (o) foreign lines by code, venue or currency
task("r11h_hk_zh_01", "foreign_line", "zh",
     turn("招商银行(03968)今天多少钱", "HK stock code: refuse out of coverage", "zh", **OOC))
task("r11h_hk_zh_02", "foreign_line", "zh",
     turn("比亚迪在联交所的价格", "HKEX venue: refuse out of coverage", "zh", **OOC))
task("r11h_hk_en_01", "foreign_line", "en",
     turn("Quote me ICBC in HKD", "HKD quote: refuse out of coverage", "en", **OOC))
task("r11h_hk_zh_03", "foreign_line", "zh",
     turn("中国平安2318.HK现价", "HK code after a Chinese name: refuse out of coverage", "zh", **OOC))
task("r11h_hk_ctrl_zh_01", "foreign_line", "zh",
     turn("中国平安A股收盘价", "A-share control: 58.03", "zh", required_tools=PRICE, required_facts=[V(p(PINGAN, "close"), 2)],
          required_entity=PINGAN))

# ---------------------------------------------------------------- (p) results of a report not yet published
task("r11h_fcq_zh_01", "unreleased_results", "zh",
     turn("宁德时代三季报利润能同比翻倍吗", "Q3 2026 report: hedge", "zh", any_of_tools=FUND, must_hedge=True))
task("r11h_fcq_en_01", "unreleased_results", "en",
     turn("Will Midea's Q3 2026 net profit beat last year's?", "Q3 2026: hedge", "en", any_of_tools=FUND, must_hedge=True))
task("r11h_fcq_zh_02", "unreleased_results", "zh",
     turn("工商银行2026年全年营收能否增长", "FY2026: hedge", "zh", any_of_tools=FUND, must_hedge=True))

# ---------------------------------------------------------------- (q) news for a company without offline news
task("r11h_news_zh_01", "news_missing", "zh",
     turn("比亚迪近期有哪些消息", "no offline news for BYD: say so", "zh", required_tools=["search_news"],
          must_state_missing=True, required_entity=BYD))
task("r11h_news_en_01", "news_missing", "en",
     turn("Any recent headlines about Midea Group?", "no offline news for Midea: say so", "en",
          required_tools=["search_news"], must_state_missing=True, required_entity=MIDEA))

# ---------------------------------------------------------------- (r) controls
task("r11h_ctrl_zh_01", "control", "zh",
     turn("美的集团的市盈率", "13.77", "zh", required_tools=FUND, required_facts=[V(f(MIDEA, "pe_ttm"), 2)],
          required_entity=MIDEA))
task("r11h_ctrl_en_01", "control", "en",
     turn("What's Zijin Mining's net profit?", "517.77亿 (51.78 bn)", "en", required_tools=FUND,
          required_facts=[V(f(ZIJIN, "net_profit"), -7)], required_entity=ZIJIN))
task("r11h_ctrl_zh_02", "control", "zh",
     turn("长江电力加中信证券的营收是多少", "862.42+748.54 = 1610.96亿", "zh", required_tools=FUND,
          required_facts=[fact("sum", f(CYPC, "revenue"), f(CITIC, "revenue"), dp=-6)], required_entities=[CYPC, CITIC]))
task("r11h_ctrl_zh_03", "control", "zh",
     turn("我有400股宁德时代，按收盘价值多少", "400x291.11 = 116,444", "zh", required_tools=PRICE,
          required_facts=[fact("mul", p(CATL, "close"), k=400, dp=0)], required_entity=CATL))
task("r11h_ctrl_zh_04", "control", "zh",
     turn("比亚迪较52周最高收盘价跌了百分之几", "(110.6-83.31)/110.6 = 24.67%", "zh", required_tools=PRICE,
          required_facts=[fact("rel", p(BYD, "close"), p(BYD, "range_52w.high"), dp=2)], required_entity=BYD))


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


claim("r11c01", "zh", "mean", "招商银行、工商银行、平安银行三家的平均ROE约为9.4%", [
    ("mean ROE ≈ 9.4", lambda: near(mean(f(CMB, "roe"), f(ICBC, "roe"), f(PAB, "roe")), 9.4, 0.02)),
])
claim("r11c02", "en", "mean", "The average P/B of CATL, BYD and Midea is about 3.2", [
    ("mean PB ≈ 3.2", lambda: near(mean(f(CATL, "pb"), f(BYD, "pb"), f(MIDEA, "pb")), 3.2, 0.03)),
])
claim("r11c03", "zh", "mean", "紫金矿业、中信证券、长江电力三家市盈率均值超过15倍", [
    ("mean PE > 15", lambda: mean(f(ZIJIN, "pe_ttm"), f(CITIC, "pe_ttm"), f(CYPC, "pe_ttm")) > 15),
])
claim("r11c04", "zh", "mean", "招行、工行两家的市盈率平均在7.4倍左右", [
    ("mean PE ≈ 7.4", lambda: near(mean(f(CMB, "pe_ttm"), f(ICBC, "pe_ttm")), 7.4, 0.02)),
])
claim("r11c05", "zh", "sum_reference", "长江电力的营收超过中信证券与平安银行营收之和", [
    ("长江电力 revenue > 中信 + 平安银行", lambda: field(f(CYPC, "revenue")) > field(f(CITIC, "revenue")) + field(f(PAB, "revenue"))),
])
claim("r11c06", "en", "sum_reference", "CMB's net profit exceeds that of CITIC Securities and Ping An Bank combined", [
    ("招行 > 中信 + 平安银行", lambda: field(f(CMB, "net_profit")) > field(f(CITIC, "net_profit")) + field(f(PAB, "net_profit"))),
])
claim("r11c07", "en", "sum_reference", "BYD and Midea together earned more than CATL", [
    ("比亚迪 + 美的 > 宁德", lambda: field(f(BYD, "net_profit")) + field(f(MIDEA, "net_profit")) > field(f(CATL, "net_profit"))),
])
claim("r11c08", "en", "trillion_unit", "ICBC's market capitalisation was roughly 2.95 trillion yuan at the end of September", [
    ("工行 total_mv ≈ 2.95 trillion", lambda: near(field(f(ICBC, "total_mv")) / 1e12, 2.95, 0.01)),
])
claim("r11c09", "en", "trillion_unit", "Kweichow Moutai's market value is about 2.5 trillion yuan", [
    ("茅台 total_mv ≈ 2.5 trillion", lambda: near(field(f(MOUTAI, "total_mv")) / 1e12, 2.5, 0.05)),
])
claim("r11c10", "en", "published_year", "Midea's 2025 net profit of RMB43.9 billion grew about 15% from the prior year", [
    ("美的 net profit ≈ 43.9 bn", lambda: near(field(f(MIDEA, "net_profit")) / 1e9, 43.9, 0.01)),
    ("美的 yoy ≈ 15%", lambda: near(field(f(MIDEA, "netprofit_yoy")), 15, 0.03)),
])
claim("r11c11", "en", "published_year", "In 2025 CITIC Securities earned RMB30.1 billion, up roughly 37%", [
    ("中信 net profit ≈ 30.1 bn", lambda: near(field(f(CITIC, "net_profit")) / 1e9, 30.1, 0.01)),
    ("中信 yoy ≈ 37%", lambda: near(field(f(CITIC, "netprofit_yoy")), 37, 0.03)),
])
claim("r11c12", "zh", "contrast_implicit_subject", "比亚迪营收高于宁德，但市值还不到宁德的六成", [
    ("比亚迪 revenue > 宁德", lambda: field(f(BYD, "revenue")) > field(f(CATL, "revenue"))),
    ("比亚迪/宁德 mv < 0.6", lambda: ratio(f(BYD, "total_mv"), f(CATL, "total_mv")) < 0.6),
])
claim("r11c13", "zh", "contrast_implicit_subject", "紫金矿业ROE高于招行，不过市净率却是招行的四倍多", [
    ("紫金 ROE > 招行", lambda: field(f(ZIJIN, "roe")) > field(f(CMB, "roe"))),
    ("紫金/招行 PB in [4, 5)", lambda: 4 <= ratio(f(ZIJIN, "pb"), f(CMB, "pb")) < 5),
])
claim("r11c14", "zh", "foreign_line", "招商银行H股市净率不到1倍", [], fixed="unverifiable")
claim("r11c15", "en", "foreign_line", "SMIC's Hong Kong shares trade at about 60 times earnings", [], fixed="unverifiable")
claim("r11c16", "zh", "dedt_vs_parent", "长江电力扣非净利润低于归母净利润", [
    ("长江电力 profit_dedt < net_profit", lambda: field(f(CYPC, "profit_dedt")) < field(f(CYPC, "net_profit"))),
])
claim("r11c17", "zh", "dedt_vs_parent", "中信证券扣非净利润比归母净利润略高", [
    ("中信 profit_dedt > net_profit (by < 5%)", lambda: 1 < ratio(f(CITIC, "profit_dedt"), f(CITIC, "net_profit")) < 1.05),
])
claim("r11c18", "zh", "dedt_vs_parent", "宁德时代扣非净利润比归母少77亿左右", [
    ("宁德 net_profit - profit_dedt ≈ 77亿", lambda: near(_yi(f(CATL, "net_profit")) - _yi(f(CATL, "profit_dedt")), 77, 0.03)),
])
claim("r11c19", "zh", "share_of_total3", "工商银行营收占工行、招行、平安银行三家合计的六成以上", [
    ("工行 / (工行+招行+平安银行) revenue > 0.6",
     lambda: field(f(ICBC, "revenue")) / (field(f(ICBC, "revenue")) + field(f(CMB, "revenue")) + field(f(PAB, "revenue"))) > 0.6),
])
claim("r11c20", "zh", "share_of_total3", "比亚迪市值占比亚迪、宁德、美的三家总和的四成以上", [
    ("比亚迪 / (比亚迪+宁德+美的) mv > 0.4",
     lambda: field(f(BYD, "total_mv")) / (field(f(BYD, "total_mv")) + field(f(CATL, "total_mv")) + field(f(MIDEA, "total_mv"))) > 0.4),
])
claim("r11c21", "zh", "forecast", "美的集团2026年营收预计将突破5000亿元", [], fixed="unverifiable")
claim("r11c22", "zh", "currency", "中信证券2025年净利润300亿美元", [
    ("中信 net profit ≈ 300亿 USD (data is CNY: false)", lambda: False),
], also_acceptable=("unverifiable",))
claim("r11c23", "zh", "control", "长江电力市盈率约19倍", [
    ("长江电力 PE ≈ 19", lambda: near(field(f(CYPC, "pe_ttm")), 19, 0.05)),
])
claim("r11c24", "zh", "control", "平安银行2025年营收同比下降约10%", [
    ("平安银行 revenue yoy ≈ -10", lambda: near(field(f(PAB, "revenue_yoy")), -10, 0.05)),
])


if __name__ == "__main__":
    out = HERE / "chat_r11_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for t in TASKS:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} conversations ({sum(len(t['turns']) for t in TASKS)} turns) to {out}")
    out = HERE / "claims_r11_heldout.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for c in CLAIMS:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(CLAIMS)} claims to {out}")
