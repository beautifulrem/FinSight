"""Compliance guard for agent answers.

Reuses the judgment/causal softening rules from ``query_intelligence/answer_guards.py`` and the market-freshness
and disclaimer conventions from ``query_intelligence/chatbot.py``, and additionally removes direct
trading instructions that a model might still produce: buy/sell calls, price targets, investment ratings
("买入评级", "rated outperform") and position sizing ("八成仓位", "逢低加仓"), including ratings quoted from
third-party documents, because relaying them reads as a recommendation.
"""

from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

from ..chatbot import (
    DEFAULT_RISK_DISCLAIMER_EN,
    DEFAULT_RISK_DISCLAIMER_ZH,
    _asks_for_current_market_data,
    _is_known_non_trading_day,
    detect_query_language,
)
from ..text_safety import fold
from .coverage import without_holding_value
from .evidence import AgentEvidence
from .router import _JUDGMENT_MARKERS as _ROUTER_JUDGMENT
from .router import FAIR_VALUE_MARKERS

_ROOT = Path(__file__).resolve().parents[2]

_TRADING_ZH = re.compile(
    r"(?:建议|推荐|可以|应该|应当|不妨|宜|可考虑|考虑|赶紧|立即|果断)\s*(?:逢低|逢高|适当|分批|立即|果断|继续)?\s*"
    r"(?:买入|卖出|加仓|减仓|清仓|满仓|全仓|抄底|建仓|止损|止盈|持有|增持|减持|入场|离场|上车)"
    r"|强烈推荐|强烈建议|目标价|梭哈|满仓|全仓买入"
    # Investment ratings and position sizing are advice too, whoever is quoted as their author.
    r"|(?:强烈)?(?:买入|增持|推荐|跑赢大市|优于大市|跑赢行业)(?:」|”|\"|')?\s*评级|评级(?:为|上调至|维持)?\s*(?:强烈)?(?:买入|增持|推荐)"
    r"|(?:[一二三四五六七八九十]|\d{1,2})\s*成仓位?|仓位(?:可|应|宜|提高|提升|加到|降到|降至|控制在)|逢低(?:加仓|买入|吸纳|布局)"
    # a call addressed to readers with adverbs in between ("建议投资者一次性建仓并长期持有", "请投资者尽快卖出持仓")
    r"|(?:建议|推荐|应该|应当|不妨|可考虑|赶紧|立即|果断|尽快|务必|必须|请)\s*(?:投资者|读者|大家|用户|股民|持有人|您|你)?\s*"
    r"(?:一次性|逢低|逢高|适当|分批|立即|果断|继续|长期|坚定|积极|尽快|马上|全部|提前)+\s*"
    r"(?:买入|卖出|加仓|减仓|清仓|满仓|全仓|抄底|建仓|止损|止盈|持有|增持|减持|入场|离场|上车|抛售|出货|退出)"
    r"|(?:尽快|务必|必须|赶紧)(?:卖出|清仓|离场|抛售|出货|买入)|黄金坑|满仓(?:杀入|买入|干)|止损位"
    # one number presented as the fair value ("合理估值约为1500元", "每股内在价值1320元"): an opinion, like a
    # target price
    r"|(?:合理|内在|真实)(?:的)?(?:估值|价值|价位|价格|股价)"
    r"(?:区间|中枢|应该|应当|应|大概|大约|估计|约|为|是|在|看|有|:|：|\s){0,4}\d|每股(?:合理|内在)价值"
)
_TRADING_EN = re.compile(
    r"\b(?:you should|we recommend|i recommend|i suggest|consider|it is a good time to|now is the time to)\s+"
    r"(?:buy|sell|short|add|accumulate|trim|dump|hold|exit|enter)\w*\b"
    r"|\b(?:strong buy|strong sell|price target|target price|go all[- ]in)\b"
    r"|\b(?:outperform|overweight|underweight|underperform|buy|sell|accumulate|market perform)\s+rating\b"
    r"|\brated\s+(?:a\s+)?(?:strong\s+)?(?:buy|sell|outperform|overweight|underweight|accumulate)\b"
    r"|\b(?:position size|increase (?:your|the) position|raise (?:your|the) position|allocate \d+%)"
    # "BUY PING AN NOW" (after NFKC folding), "sell Wuliangye before Friday", "holders must exit by 30 April"
    r"|\bbuy\s+(?:[a-z0-9.&'-]+\s+){0,3}?now\b|\b(?:sell|dump|exit)\s+(?:[a-z0-9.&'-]+\s+){0,3}?(?:now|immediately|before\s+\w+)\b"
    r"|\b(?:holders|investors|shareholders|you|readers|users|everyone)\s+(?:must|should)\s+"
    r"(?:exit|sell|dump|liquidate|get out)\b|\bgo all[- ]in\b|\bmust[- ]buy\b|\bfull position\b"
    # "a fair value of CNY 1,500", "intrinsic value is about 1320", "worth about CNY 1,500 a share"
    r"|\b(?:fair|intrinsic|true) (?:value|price)\b[^.;]{0,30}?\b(?:is|of|at|around|about|near)\b\s*"
    r"(?:about |around |roughly )?(?:cny|rmb|us\$|\$|¥)?\s*\d"
    r"|\bworth (?:about |around |roughly |approximately )?(?:cny|rmb|us\$|\$|¥)\s*\d",
    re.IGNORECASE,
)
# Lexical triggers make hedging independent of the NLU question-style label. Besides buy/sell and timing words
# they cover valuation judgements ("贵还是便宜", "which is cheaper"), market calls ("牛市信号", "is it trending
# up?"), guarantees ("一定会涨", "guarantee stocks go up") and position sizing ("全仓…行不行").
_JUDGMENT_TRIGGER = re.compile(
    r"能买|能不能买|值得买|要不要|该不该|会涨|会跌|能涨|能跌|涨多少|跌多少|必涨|必跌|目标价|买点|卖点|买入点|满仓|梭哈|"
    r"加仓|清仓|止损|止盈|还能拿|值得拿|持有吗|翻倍|哪个更好|选哪个|抄底|上车|全仓|行不行|贵还是便宜|贵不贵|便宜|"
    r"高估|低估|牛市|熊市|见顶|见底|一定会|一定涨|必然|肯定会|稳赚|保证|机会|涨到|跌到|值不值|走强|走弱|"
    r"\bshould i\b|\bbuy\b|\bsell\b|price target|target price|strong buy|go all[- ]in|\bdouble\b|"
    r"when to (?:buy|sell)|worth buying|\bwill\b.*\b(?:rise|fall|go up|go down|double|rally|drop|rebound)\b|"
    r"\bbuying opportunit|\bopportunit(?:y|ies)\b|\bcheap(?:er|est)?\b|\bexpensive\b|\bover(?:valued|priced)\b|"
    r"\bunder(?:valued|priced)\b|\bbull(?:ish)? market\b|\bbear(?:ish)? market\b|\bguarantee|\btrending\b|"
    r"\b(?:up|down)trend\b|\bgood time to\b",
    re.IGNORECASE,
)
_CAUSAL_TRIGGER = re.compile(
    r"为什么|原因|因素|影响|导致|意味着|关系|传导|友好|有利|不利|利好|利空|说明|预示|信号|反映|体现|"
    r"\bwhy\b|impact|affect|cause|mean for|driver|good for|bad for|favorable|\bmeans?\b|\bsignal|"
    r"\bindicat|\bsuggest|\breflect|priced in",
    re.IGNORECASE,
)
_JUDGMENT_PREFIX_ZH = "基于当前证据只能做条件性判断，不能据此给出确定的买入、卖出、持有建议或价格预测。"
_JUDGMENT_PREFIX_EN = (
    "Based on the current evidence this can only be a conditional assessment, not a buy, sell, or hold "
    "recommendation or a price forecast."
)
_CAUSAL_CAVEAT_ZH = "以上证据只能提示可能的影响因素，不能据此确定因果关系。"
_CAUSAL_CAVEAT_EN = "This evidence only points to possible factors; it does not establish cause and effect."
_HEDGE_MARKERS = ("条件性", "不能据此", "可能", "possible", "conditional", "does not establish", "cannot")

# A fair-value question ("合理估值是多少", "What is it worth?"): the evidence has market prices and multiples, which
# are stated, but no valuation model; the answer says so instead of letting a price read as the fair value.
_FAIR_VALUE_ZH = "FinSight 不给出合理估值：下文的价格、PE、PB 和行业对比是市场数据和估值参考，不是对合理价值的判断。"
_FAIR_VALUE_EN = (
    "FinSight does not give a fair value: the prices, P/E, P/B and industry comparison below are market data and "
    "valuation references, not a judgement of what the stock is worth."
)
_FAIR_VALUE_LIMITATION_ZH = "证据中没有可据以确定合理估值的估值模型或一致预期，不能给出合理价格"
_FAIR_VALUE_LIMITATION_EN = "The evidence has no valuation model or consensus estimate from which a fair value follows"

_NEUTRAL_ZH = "是否交易取决于个人风险承受能力、投资期限和持仓情况，以上仅为证据梳理，不构成买卖建议。"
_NEUTRAL_EN = (
    "Whether to trade depends on your risk tolerance, horizon, and existing positions; the above is an evidence "
    "summary, not a trading recommendation."
)


def _guards() -> ModuleType:
    from .. import answer_guards

    return answer_guards


_LINGUA_LOCK = threading.Lock()
_lingua_detector: Any = None


def _detector() -> Any:
    """Lazily built lingua detector restricted to English and major European languages (about 5 s once)."""
    global _lingua_detector
    with _LINGUA_LOCK:
        if _lingua_detector is None:
            try:
                from lingua import Language, LanguageDetectorBuilder
            except ImportError:
                _lingua_detector = False
            else:
                _lingua_detector = LanguageDetectorBuilder.from_languages(
                    Language.ENGLISH,
                    Language.FRENCH,
                    Language.GERMAN,
                    Language.SPANISH,
                    Language.PORTUGUESE,
                    Language.ITALIAN,
                ).build()
        return _lingua_detector


_ACRONYM_OR_TICKER = re.compile(r"(?<![A-Za-z])(?:[A-Z]{1,5}(?:/[A-Z]{1,3})?|\d{6}\.[A-Z]{2})(?![A-Za-z])")


def language_violation(answer_text: str, query: str, *, language: str | None = None) -> bool:
    """True when the answer is not in the language of the question (e.g. hijacked by a poisoned document).

    Chinese questions need a CJK-dominant answer. English answers are checked with lingua on the text
    stripped of citations, Chinese names and numbers, and only when enough Latin text remains to be reliable.
    """
    text = str(answer_text or "")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if (language or detect_query_language(query)) == "zh":
        return latin >= 40 and cjk < max(2, latin * 0.2)
    if cjk >= max(40, latin):
        return True
    # (round 12, H7) a short Chinese answer to an English question ("贵州茅台（600519.SH）的市盈率 PE(TTM) 为 24.6 倍
    # [fundamental_600519.SH]。" for "And Moutai?") has fewer than 40 Chinese characters; counted without citations,
    # tickers and acronyms, Chinese still outweighs the Latin text. Chinese names in an English answer do not.
    body = _ACRONYM_OR_TICKER.sub(" ", re.sub(r"\[[^\]]+\]", " ", text))
    body_cjk = len(re.findall(r"[一-鿿]", body))
    if body_cjk >= 8 and body_cjk >= len(re.findall(r"[A-Za-z]", body)):
        return True
    cleaned = re.sub(r"\[[^\]]+\]|[\u4e00-\u9fff]+|[\d.,%()（）:：/-]+", " ", text)
    if len(re.findall(r"[A-Za-z]", cleaned)) < 60:
        return False
    detector = _detector()
    if not detector:
        return False
    values = {item.language.name: item.value for item in detector.compute_language_confidence_values(cleaned)}
    top = max(values, key=values.get)
    return top != "ENGLISH" and values[top] >= 0.8 and values.get("ENGLISH", 0.0) < 0.2


def contains_trading_instruction(text: str) -> bool:
    folded = fold(text)  # full-width ("ＢＵＹ ＮＯＷ"), zero-width and homoglyph spellings match too
    return bool(_TRADING_ZH.search(folded) or _TRADING_EN.search(folded))


def apply_compliance(
    answer: dict[str, Any],
    *,
    query: str,
    nlu_result: dict[str, Any],
    tool_failures: list[str] | None = None,
    market_evidence: list[AgentEvidence] | None = None,
    today: date | None = None,
    language: str | None = None,
    effective_query: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(guarded_answer, notes)``; notes name the rules that changed the answer.

    ``language`` overrides detection from ``query`` (the graph passes the language of the user's own words).
    ``effective_query`` is the question after follow-up resolution: judgment and causal wording is looked for in
    both, so "五粮液" answering "这个能买吗？", or "为什么涨？" resolved to the index, is still hedged.
    """
    guards = _guards()
    zh = (language or detect_query_language(query)) == "zh"
    pseudo_retrieval = {"warnings": list(tool_failures or []), "coverage": {}}
    guarded = dict(answer)
    notes: list[str] = []

    text = str(guarded.get("answer") or "")
    softened = guards._soften_answer_text(text, nlu_result, pseudo_retrieval, None, query=query, zh=zh)
    points = [str(point) for point in guarded.get("key_points") or [] if str(point).strip()]
    softened_points = guards._soften_key_points(points, nlu_result, query=query, zh=zh)
    if softened != text or softened_points != points:
        notes.append("softened_judgment_or_causal_language")

    softened, removed_answer = _strip_trading_sentences(softened, zh=zh)
    cleaned_points: list[str] = []
    removed_points = 0
    for point in softened_points:
        if contains_trading_instruction(point):
            removed_points += 1
            continue
        cleaned_points.append(point)
    if removed_answer or removed_points:
        notes.append("removed_trading_instruction")

    # Guaranteed returns, stock-tip solicitation and private contact details: removed whoever wrote them.
    softened, removed_promotion = guards.strip_prohibited_promotion(softened, zh=zh)
    promotion_points = [point for point in cleaned_points if guards.contains_prohibited_promotion(point)]
    cleaned_points = [point for point in cleaned_points if point not in promotion_points]
    if removed_promotion or promotion_points:
        notes.append("removed_prohibited_promotion")

    if guards._needs_conditional_answer_guard(nlu_result, pseudo_retrieval, None, query):
        prefix = guards._conditional_answer_prefix(nlu_result, zh=zh, query=query).strip()
        if prefix and prefix not in softened:
            softened = f"{prefix}{'' if zh else ' '}{softened}".strip()
            notes.append("conditional_prefix")

    # (round 11, G5) the value of a stated holding ("我有1000股…值多少钱") is arithmetic on the close, not a fair value
    asked = f"{without_holding_value(query)} {without_holding_value(effective_query or '')}"
    if (_JUDGMENT_TRIGGER.search(asked) or _ROUTER_JUDGMENT.search(asked)) and "conditional_prefix" not in notes:
        prefix = _JUDGMENT_PREFIX_ZH if zh else _JUDGMENT_PREFIX_EN
        if not any(marker in softened for marker in ("条件性判断", "conditional assessment", "证据不足以直接判断")):
            softened = f"{prefix}{'' if zh else ' '}{softened}".strip()
            notes.append("conditional_prefix")
    elif _CAUSAL_TRIGGER.search(asked) and not any(marker in softened.lower() for marker in _HEDGE_MARKERS):
        caveat = _CAUSAL_CAVEAT_ZH if zh else _CAUSAL_CAVEAT_EN
        softened = f"{softened}{'' if zh else ' '}{caveat}".strip()
        notes.append("causal_caveat")

    fair_value = bool(FAIR_VALUE_MARKERS.search(asked))
    if fair_value:
        # after the conditional prefix, before the evidence: the price that follows is not the answer to "worth"
        note = _FAIR_VALUE_ZH if zh else _FAIR_VALUE_EN
        if note not in softened:
            prefixes = (
                _JUDGMENT_PREFIX_ZH if zh else _JUDGMENT_PREFIX_EN,
                guards._conditional_answer_prefix(nlu_result, zh=zh, query=query).strip(),
            )
            lead = next((prefix for prefix in prefixes if prefix and softened.startswith(prefix)), "")
            rest = softened[len(lead) :].lstrip()
            softened = ("" if zh else " ").join(part for part in (lead, note, rest) if part)
        notes.append("fair_value_hedge")

    limitations = [str(item) for item in guarded.get("limitations") or [] if str(item).strip()]
    if fair_value:
        limitations = guards._append_unique(
            limitations, [_FAIR_VALUE_LIMITATION_ZH if zh else _FAIR_VALUE_LIMITATION_EN]
        )
    limitations = guards._append_unique(
        limitations, guards._guardrail_limitations(pseudo_retrieval, nlu_result, zh=zh, query=query)
    )
    if "conditional_prefix" in notes:
        limitations = guards._append_unique(
            limitations, ["问题包含投资建议或预测属性" if zh else "The query has advice or forecast-like risk"]
        )

    freshness_point = _freshness_note(query, market_evidence or [], zh=zh, today=today or date.today())
    if freshness_point:
        cleaned_points = [freshness_point, *[point for point in cleaned_points if point != freshness_point]]
        limitations = guards._append_unique(limitations, [freshness_point])
        notes.append("market_freshness")

    guarded["answer"] = softened
    guarded["key_points"] = cleaned_points
    guarded["limitations"] = limitations
    disclaimer = str(guarded.get("risk_disclaimer") or "").strip()
    default_disclaimer = DEFAULT_RISK_DISCLAIMER_ZH if zh else DEFAULT_RISK_DISCLAIMER_EN
    if not disclaimer or contains_trading_instruction(disclaimer) or guards.contains_prohibited_promotion(disclaimer):
        guarded["risk_disclaimer"] = default_disclaimer
    return guarded, notes


def _strip_trading_sentences(text: str, *, zh: bool) -> tuple[str, int]:
    # Split after sentence ends but keep the whitespace (on the next sentence), so rejoining English text
    # does not glue sentences together ("[price_x].Industry").
    sentences = [part for part in re.split(r"(?<=[。！？!?；;])|(?<=\.)(?=\s)", text) if part]
    kept: list[str] = []
    removed = 0
    for sentence in sentences:
        if contains_trading_instruction(sentence):
            removed += 1
            continue
        kept.append(sentence)
    if removed:
        neutral = _NEUTRAL_ZH if zh else _NEUTRAL_EN
        kept.append(neutral if zh else f" {neutral}")
    return "".join(kept).strip(), removed


def _price_basis_note(market_evidence: list[AgentEvidence], *, zh: bool, latest: str | None) -> str | None:
    """What a 今天 price answer is based on, when ``get_price_history(intraday=true)`` ran.

    An intraday quote is labelled as intraday (not a close); a daily close served to an intraday request says
    why (outside trading hours, no live source, or the real-time source failed). A non-trading day is left to
    the generic note below.
    """
    payloads = [item.payload or {} for item in market_evidence]
    quotes = [p["intraday"] for p in payloads if p.get("price_basis") == "intraday" and p.get("intraday")]
    if quotes:
        stamp = max(str(quote.get("quote_time") or "") for quote in quotes)[:19].replace("T", " ")
        return (
            f"以上为盘中实时行情（截至 {stamp} 北京时间），不是收盘价，收盘前仍会变化。"
            if zh
            else f"Prices are intraday quotes (as of {stamp} Beijing time), not closing prices; "
            "they change until the close."
        )
    reasons = [str(p.get("basis_reason")) for p in payloads if p.get("basis_reason")]
    sessions = {str(p.get("market_session")) for p in payloads if p.get("market_session")}
    if not reasons or sessions == {"non_trading_day"}:
        return None
    day = latest or "-"
    if reasons[0] == "outside_trading_hours":
        return (
            f"当前为非交易时段，以下为最近交易日收盘价（{day}），不是盘中实时价格。"
            if zh
            else f"The market is closed now; the price below is the latest daily close ({day}), not an intraday quote."
        )
    if reasons[0] == "intraday_unavailable":
        return (
            f"未接入盘中实时行情（离线数据），以下为最近交易日收盘价（{day}），不能据此判断今天的盘中涨跌。"
            if zh
            else "No real-time source is configured (offline data); the price below is the latest daily close "
            f"({day}), so today's intraday move cannot be determined."
        )
    return (
        f"盘中实时行情获取失败，以下为最近交易日收盘价（{day}），不能据此判断今天的盘中涨跌。"
        if zh
        else f"The real-time quote could not be retrieved; the price below is the latest daily close ({day}), "
        "so today's intraday move cannot be determined."
    )


def _freshness_note(query: str, market_evidence: list[AgentEvidence], *, zh: bool, today: date) -> str | None:
    if not _asks_for_current_market_data(query):
        return None
    today_text = today.isoformat()
    dates = sorted({str(item.as_of)[:10] for item in market_evidence if item.as_of}, reverse=True)
    basis_note = _price_basis_note(market_evidence, zh=zh, latest=dates[0] if dates else None)
    if basis_note:
        return basis_note
    if _is_known_non_trading_day(today):
        latest = f"，最新可用交易日行情日期为 {dates[0]}" if dates and zh else ""
        latest_en = f"; the latest available trading-day quote is from {dates[0]}" if dates and not zh else ""
        return (
            f"今天（{today_text}）不是 A 股常规交易日，没有当日涨跌行情{latest}。"
            if zh
            else f"Today ({today_text}) is not a regular A-share trading day{latest_en}."
        )
    if not dates:
        return (
            f"未获取到今日（{today_text}）实时行情，不能判断今天的涨跌。"
            if zh
            else f"Today's ({today_text}) real-time quote was not retrieved, so today's move cannot be determined."
        )
    if dates[0] != today_text:
        return (
            f"最新可用行情日期为 {dates[0]}，不是今日（{today_text}）实时行情。"
            if zh
            else f"The latest available market date is {dates[0]}, not today's ({today_text}) real-time quote."
        )
    return None
