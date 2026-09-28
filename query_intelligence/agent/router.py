"""Route a query to refusal, clarification, the fixed workflow, or the agent loop.

The decision uses classical NLU output plus a few explicit lexical markers, and always returns
the reasons that fired, so routing stays explainable.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["refuse", "clarify", "workflow", "agent"]
Mode = Literal["auto", "workflow", "agent"]

_LISTED_TYPES = {"stock", "etf", "fund", "index"}
_COMPLEX_STYLES = {"why", "compare", "forecast"}
_COMPLEX_INTENTS = {"market_explanation", "macro_policy_impact", "peer_compare"}
_MULTI_HOP_MARKERS = re.compile(
    r"结合|同时|并且|以及.*(影响|变化)|对比|相比|比较|分别|还是|哪个|影响|传导|联动|为什么|原因|归因|"
    r"意味着|关联|关系|是否匹配|综合来看|综合|"
    r"\bcompare|\bversus\b|\bvs\.?\b|\bimpact\b|\baffect|\beffect\b|\brelat(?:ion|ed|es)|\bcorrelat|"
    r"\blinked\b|\bwhy\b|\bcombined\b|\btogether with\b|\band then\b",
    re.IGNORECASE,
)
# Judgment and timing questions need valuation, fundamentals and news plus hedging: never a single lookup.
_JUDGMENT_MARKERS = re.compile(
    r"抄底|能不能买|能买吗|值得买|值不值得|要不要|该不该|会涨|会跌|能涨|还能涨|涨吗|跌吗|见底|高估|低估|买点|卖点|"
    r"止盈|止损|逃顶|上车|还能拿|拿得住|适合定投|适合买|值得持有|长期持有|"
    r"\bshould i\b|\bworth (?:buying|it)\b|\bwill\b.{0,40}\b(?:rise|fall|go up|go down|drop|rebound)\b|"
    r"\bgood time to\b|\bovervalued\b|\bundervalued\b|\bbottom(?:ed)?\b",
    re.IGNORECASE,
)
_WHY_MARKERS = re.compile(
    r"\bdr(?:ove|ives|iving)\b|\bdriver[s]?\b|\bbehind\b|\bwhat caused\b|\breasons? for\b|\bexplain\b",
    re.IGNORECASE,
)
_FOLLOW_UP_MARKERS = re.compile(r"^(那|那么|它|这只|这个|该股|那它|and |what about |how about )", re.IGNORECASE)
# "利率" must not fire inside 毛利率 / 净利率 (company margins, not interest rates).
_MACRO_ANCHOR = re.compile(
    r"cpi|ppi|pmi|gdp|m2|lpr|社融|(?<![毛净])利率|国债|降息|降准|通胀|通缩|通货紧缩|货币供应|宏观|"
    r"bond yield|interest rate|inflation|deflation|money supply|\bcgb\b|government bond|treasury yield|"
    r"\b10[- ]?(?:year|yr|y)\b.{0,20}\byield",
    re.IGNORECASE,
)
_FINANCE_ANCHOR = re.compile(
    r"股票|股价|个股|该股|这只票|那只票|基金|etf|lof|指数|估值|走势|行情|市盈率|市净率|收盘|涨跌|上涨|下跌|大涨|大跌|"
    r"涨停|跌停|业绩|财报|分红|满仓|加仓|减仓|清仓|仓位|会涨|会跌|能涨|涨吗|跌吗|抄底|买入|卖出|能买|值得买|"
    r"(?<![A-Za-z])(?:P/?E|P/?B|ROE)(?![A-Za-z])|股息|不良率|营收|净利|同行|涨了|跌了|费率|股价|"
    r"净资产收益率|毛利率|负债率|市值|公告|年报|季报|现金流|"
    r"\bstocks?\b|\bshares?\b|\bfunds?\b|valuation|overvalued|undervalued|earnings|share price|dividend|"
    r"\bbuy\b|\bsell\b|\binvest(?:ing|ment)?\b|\bdrop\b|\bgo up\b|\brise\b|\bfall\b|"
    r"\brevenue\b|\bnet (?:profit|income)\b|\bmargins?\b|market cap|\bprice\b|\bannouncements?\b",
    re.IGNORECASE,
)
_DANGLING_REFERENCE = re.compile(
    r"(?<!其)它|这只|这支|这个基金|这个标的|这个指数|这个股票|这家|那家|该股|该公司|该基金|那只|那支|"
    r"\bit\b|\bits\b|\b(?:this|that) (?:stock|fund|company|one|etf|index|bank)\b|"
    # plural references ("这两家哪个更值得关注") and dangling why follow-ups ("为什么会这样") in a session
    # without earlier targets: a clarification, never an off-topic refusal
    r"这两家|这两只|这两个|两家公司|两者|二者|它们|\bboth (?:of them|companies|stocks)\b|\bthese two\b|\bthe two\b",
    re.IGNORECASE,
)
# Dangling "why" follow-ups that name neither a target nor an aspect: "为什么会这样", "怎么回事", "why did that
# happen?". Only whole short questions qualify ("大盘今天怎么回事" names the market and is not dangling).
_DANGLING_WHY_ZH = re.compile(
    r"^(?:那|那么|所以|但|可)?(?:这|那)?(?:是)?(?:为什么|为何|怎么|咋)(?:会|能|就)?"
    r"(?:这样|如此|这么\S{0,3}|那样|回事|了)?(?:呢|啊|呀)?[？?。!！]*$|"
    r"^(?:那|那么)?(?:这|那)?(?:是)?(?:什么原因|啥原因)(?:呢|啊|导致的)?[？?。!！]*$|"
    r"^(?:那|那么)?(?:背后的)?原因(?:是什么|是啥|呢|何在)[？?。!！]*$"
)
_DANGLING_WHY_EN = re.compile(
    r"^(?:and |so |but |ok,? )?(?:why(?: is| was| did| does| do| has| would)?(?: that| this)?"
    r"(?: happen(?:ing|ed)?| so| the case)?|how come|what caused (?:that|this)|"
    r"what(?:'s| is| was) behind (?:that|this)|what drove (?:that|this)|"
    r"what(?:'s| is) the reason(?: for (?:that|this))?)\s*[?.!]*$",
    re.IGNORECASE,
)


def is_dangling_why(query: str) -> bool:
    text = (query or "").strip()
    return bool(_DANGLING_WHY_ZH.match(text) or _DANGLING_WHY_EN.match(text))


# Requests for a non-research task. They are refused even when they mention a stock or finance words
# ("你能帮我写个Python爬虫抓股价吗"): FinSight researches securities, it does not write code, translate, or book
# travel. Only explicit task phrasings count, so "天气转暖对白酒消费有影响吗" is not a weather request.
_OFF_TOPIC_TASKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "coding",
        re.compile(
            r"python|java(?:script)?|c\+\+|爬虫|爬取|写(?:一)?(?:个|段|份)?(?:程序|脚本|代码|函数|接口)|编程|"
            r"\bscrap(?:e|er|ing)\b|\b(?:write|give me|generate)\b.{0,20}\b(?:code|script|program|function)\b|"
            r"\bdebug\b",
            re.IGNORECASE,
        ),
    ),
    ("translation", re.compile(r"翻译|译成|\btranslate\b|\binto (?:french|english|german|spanish|japanese)\b", re.I)),
    (
        "travel",
        re.compile(
            r"机票|火车票|高铁票|订.{0,10}?(?:票|酒店|民宿)|酒店预订|\bbook (?:a |me )?(?:flight|hotel|ticket)", re.I
        ),
    ),
    ("weather", re.compile(r"天气(?:怎么样|如何|预报)|\bweather (?:like|today|tomorrow|forecast)\b", re.I)),
    (
        "writing",
        re.compile(
            r"写(?:一)?(?:首|篇).{0,12}?(?:诗|作文|文章|小说)|\bwrite (?:me )?(?:a |an )?(?:poem|essay|story)\b", re.I
        ),
    ),
    ("entertainment", re.compile(r"讲个笑话|\btell me a joke\b|推荐(?:一部|几部)?电影|\brecommend a movie\b", re.I)),
)


def off_topic_request(query: str) -> str | None:
    """The kind of non-research task the message asks for (``coding``, ``translation``, ...), or ``None``."""
    for label, pattern in _OFF_TOPIC_TASKS:
        if pattern.search(query or ""):
            return label
    return None


# Words that only make sense about a security, a sector or the economy. Inside a conversation that already has
# a target they mark an entity-less message ("为什么涨？", "增速是多少？", "What's the 3-day return?") as a follow-up.
_FOLLOW_UP_CUE = re.compile(
    r"涨|跌|反弹|回调|走势|趋势|行情|表现|收盘|开盘|最高|最低|成交|均线|MA\s*\d+|RSI|MACD|布林|站上|跌破|支撑|"
    r"增速|增长|同比|环比|营收|收入|利润|赚|亏|毛利|净利|估值|贵|便宜|高估|低估|市盈|市净|分红|股息|市值|"
    r"舆情|情绪|正面|负面|利好|利空|消息|新闻|公告|财报|业绩|牛市|熊市|信号|抄底|买入|卖出|买吗|卖吗|能买|该买|"
    r"加仓|减仓|仓位|全仓|满仓|持有|机会|风险|波动|收益|回报|说明|意味|预示|代表|反映|怎么看|为什么|原因|影响|通缩|通胀|"
    r"比较|相比|对比|哪家|哪个|哪只|上车|下车|入场|离场|建仓|"
    r"\b(?:up|down|rise|rose|risen|fall|fell|drop(?:ped)?|rally|gain(?:ed|s)?|loss|returns?|perform(?:ance)?|"
    r"close[sd]?|closing|open(?:ed|ing)?|high|low|volume|turnover|trend(?:ing)?|moving average|ma\d+|rsi|macd|"
    r"volatility|growth|revenue|profit|earnings|margin|valuation|cheap(?:er|est)?|expensive|p/?e|p/?b|roe|"
    r"dividend|sentiment|news|announcements?|filings?|bull(?:ish)?|bear(?:ish)?|signal|buy(?:ing)?|sell(?:ing)?|"
    r"hold(?:ing)?|position|opportunity|risk|why|mean|imply|inflation|deflation|yield|rates?|compare[sd]?|"
    r"which)\b",
    re.IGNORECASE,
)


def has_follow_up_cue(query: str) -> bool:
    return bool(_FOLLOW_UP_CUE.search(query or ""))


def has_macro_content(query: str) -> bool:
    return bool(_MACRO_ANCHOR.search(query))


def has_finance_content(query: str) -> bool:
    return bool(_FINANCE_ANCHOR.search(query) or _MACRO_ANCHOR.search(query))


def apply_finance_overrides(nlu_result: dict[str, Any], query: str) -> tuple[dict[str, Any], list[str]]:
    """Correct classical-NLU out-of-scope false positives with explicit finance anchors.

    Returns the (possibly patched) NLU result and the reasons for any change.
    """
    flags = set(nlu_result.get("risk_flags") or [])
    product = (nlu_result.get("product_type") or {}).get("label")
    if "out_of_scope_query" not in flags and product != "out_of_scope":
        return nlu_result, []
    patched = dict(nlu_result)
    patched["risk_flags"] = [flag for flag in nlu_result.get("risk_flags") or [] if flag != "out_of_scope_query"]
    if _MACRO_ANCHOR.search(query):
        patched["product_type"] = {"label": "macro", "score": 0.5}
        patched["source_plan"] = ["macro_sql", "news"]
        return patched, ["override:out_of_scope_with_macro_anchor"]
    if _FINANCE_ANCHOR.search(query):
        patched["product_type"] = {"label": "unknown", "score": 0.5}
        patched["missing_slots"] = sorted({*(nlu_result.get("missing_slots") or []), "missing_entity"})
        return patched, ["override:out_of_scope_with_finance_anchor"]
    if (_DANGLING_REFERENCE.search(query) or is_dangling_why(query)) and len(query) <= 40:
        # A short follow-up about "it" is a conversation turn without context, not an off-topic request.
        patched["product_type"] = {"label": "unknown", "score": 0.5}
        patched["missing_slots"] = sorted({*(nlu_result.get("missing_slots") or []), "missing_entity"})
        return patched, ["override:out_of_scope_dangling_reference"]
    return nlu_result, []


_DEFINITION = re.compile(
    r"什么是|是什么|什么意思|含义|定义|概念|怎么算|如何计算|计算公式|区别|"
    r"\bwhat (?:is|are) (?:a|an|the)?\s*(?:p/?e|p/?b|roe|price|dividend)|\bdefin|\bmeaning\b|\bexplain\b",
    re.IGNORECASE,
)
_CONCEPT_TYPES = {"sector", "financial_metric", "macro_indicator", "policy"}
_COMPANY_METRIC = re.compile(
    r"市盈率|市净率|净资产收益率|营收|营业收入|净利润|毛利率|股息率|市值|(?<![A-Za-z])(?:P/?E|P/?B|ROE)(?![A-Za-z])|"
    r"\brevenue\b|\bnet (?:profit|income)\b|\bgross margin\b|\bdividend yield\b|\bmarket cap|\bearnings\b",
    re.IGNORECASE,
)


def drop_fuzzy_concepts(nlu_result: dict[str, Any], query: str) -> tuple[dict[str, Any], list[str]]:
    """Drop fuzzy-matched concept entities whose name is not in the question.

    Fuzzy alias matching is useful for company names with typos, but for short concept names it produces
    false hits ("那家公司最近有公告吗" -> sector 有色金属), which would ground a question that names no target.
    The same holds for fuzzy company matches in a question that refers back with a pronoun.
    """
    kept, dropped = [], []
    # "它值得长期持有吗" / "这只股票适合长期持有吗" point back at a target; a fuzzy company match inside them
    # ("值得" -> 值得买, "长期" -> 长江投资) is noise, so the reference is resolved from the session or clarified.
    dangling = bool(_DANGLING_REFERENCE.search(query))
    for entity in nlu_result.get("entities") or []:
        name = str(entity.get("canonical_name") or "")
        fuzzy = "fuzzy" in str(entity.get("match_type") or "")
        concept = entity.get("entity_type") in _CONCEPT_TYPES
        listed_noise = dangling and entity.get("entity_type") in _LISTED_TYPES
        if fuzzy and (concept or listed_noise) and name and name.lower() not in query.lower():
            dropped.append(name)
        else:
            kept.append(entity)
    if not dropped:
        return nlu_result, []
    return {**nlu_result, "entities": kept}, [f"dropped_fuzzy_concept:{name}" for name in dropped]


class RouteDecision(BaseModel):
    route: Route
    reasons: list[str] = Field(default_factory=list)
    complexity_score: int = 0
    features: dict[str, Any] = Field(default_factory=dict)


def entity_types_of(entities: list[dict[str, Any]]) -> set[str]:
    return {str(entity.get("entity_type")) for entity in entities if entity.get("entity_type")}


def decide_route(nlu_result: dict[str, Any], *, mode: Mode = "auto", query: str | None = None) -> RouteDecision:
    risk_flags = set(nlu_result.get("risk_flags") or [])
    entities = nlu_result.get("entities") or []
    listed = {
        entity.get("symbol")
        for entity in entities
        if entity.get("symbol") and entity.get("entity_type") in _LISTED_TYPES
    }
    text = query or str(nlu_result.get("raw_query") or nlu_result.get("normalized_query") or "")

    if "out_of_scope_query" in risk_flags or (nlu_result.get("product_type") or {}).get("label") == "out_of_scope":
        return RouteDecision(route="refuse", reasons=["nlu:out_of_scope_query"])
    missing = set(nlu_result.get("missing_slots") or [])
    # A metric alone ("市净率是多少") names no target: only listed, macro, policy or sector entities count.
    targeted = listed or entity_types_of(entities) & {"macro_indicator", "policy", "sector"}
    if "missing_entity" in missing and not targeted:
        return RouteDecision(route="clarify", reasons=["nlu:missing_entity"])
    if "clarification_required" in risk_flags and not listed and not entities:
        return RouteDecision(route="clarify", reasons=["nlu:clarification_required"])

    if not targeted and (_DANGLING_REFERENCE.search(text) or is_dangling_why(text)):
        return RouteDecision(route="clarify", reasons=["dangling_reference"])
    if not targeted and "financial_metric" in entity_types_of(entities) and not _DEFINITION.search(text):
        # "市净率是多少": a company metric with no company ("什么是市净率" is a concept question).
        return RouteDecision(route="clarify", reasons=["metric_without_target"])
    if not targeted and _FOLLOW_UP_MARKERS.search(text.strip()) and _COMPANY_METRIC.search(text):
        # "What about the P/E?" opening a conversation: an elliptical metric question with nothing to refer to.
        return RouteDecision(route="clarify", reasons=["metric_without_target", "ellipsis_without_antecedent"])

    reasons: list[str] = []
    style = str(nlu_result.get("question_style") or "")
    intents = {
        item.get("label") for item in nlu_result.get("intent_labels") or [] if float(item.get("score", 1)) >= 0.5
    }
    entity_types = {entity.get("entity_type") for entity in entities}
    comparison_targets = nlu_result.get("comparison_targets") or []

    if len(listed) >= 2:
        reasons.append(f"multi_entity:{len(listed)}")
    if len(comparison_targets) >= 2:
        reasons.append("comparison_targets")
    multi_hop = bool(_MULTI_HOP_MARKERS.search(text))
    anchored_to_market = bool(listed) or "sector" in entity_types
    if style in _COMPLEX_STYLES:
        # The style classifier alone is noisy (比亚迪 reads as a comparison): with a single target it needs
        # lexical support before a question counts as complex.
        supported = len(listed) >= 2 or multi_hop or bool(_WHY_MARKERS.search(text)) or style == "forecast"
        if supported:
            reasons.append(f"question_style:{style}")
    for intent in sorted(intents & _COMPLEX_INTENTS):
        # A macro question counts as macro-to-market only when it names a market target.
        if intent == "macro_policy_impact" and not (anchored_to_market or multi_hop):
            continue
        reasons.append(f"intent:{intent}")
    if len(intents) >= 3:
        reasons.append(f"multi_intent:{len(intents)}")
    if entity_types & {"macro_indicator", "policy"} and (listed or "sector" in entity_types):
        reasons.append("cross_domain:macro_to_market")
    if multi_hop:
        reasons.append("lexical:multi_hop_marker")
    if _JUDGMENT_MARKERS.search(text):
        reasons.append("lexical:judgment_or_timing")
    if _WHY_MARKERS.search(text):
        reasons.append("lexical:why")
    if _FOLLOW_UP_MARKERS.search(text.strip()):
        reasons.append("lexical:follow_up")

    score = len(reasons)
    features = {
        "listed_entities": len(listed),
        "question_style": style,
        "intents": sorted(label for label in intents if label),
        "entity_types": sorted(label for label in entity_types if label),
    }
    if mode == "workflow":
        return RouteDecision(
            route="workflow", reasons=["mode:workflow", *reasons], complexity_score=score, features=features
        )
    if mode == "agent":
        return RouteDecision(route="agent", reasons=["mode:agent", *reasons], complexity_score=score, features=features)
    route: Route = "agent" if score >= 1 else "workflow"
    if not reasons:
        reasons.append("simple:single_lookup")
    return RouteDecision(route=route, reasons=reasons, complexity_score=score, features=features)
