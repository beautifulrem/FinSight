"""Answer guardrails shared by the agent layer and the offline answer generator.

Softening of judgment and causal wording, conditional-answer prefixes, guardrail limitations and safe
follow-up questions. Moved here from ``scripts/llm_response.py`` so the package does not import from
``scripts/`` (which required editing ``sys.path`` at runtime); that script re-exports these names.
"""

from __future__ import annotations

import re
from typing import Any

def strip_history(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_history(val) for key, val in value.items() if key != "history"}
    if isinstance(value, list):
        return [strip_history(item) for item in value]
    return value


def _short_text(value: Any, limit: int = 280) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _nlu_product_label(nlu_result: dict[str, Any] | None) -> str:
    if not isinstance(nlu_result, dict):
        return ""
    product_type = nlu_result.get("product_type")
    if isinstance(product_type, dict):
        return str(product_type.get("label") or "").strip().lower()
    return str(product_type or "").strip().lower()


def _nlu_risk_flags(nlu_result: dict[str, Any] | None) -> list[str]:
    if not isinstance(nlu_result, dict):
        return []
    flags = nlu_result.get("risk_flags")
    if not isinstance(flags, list):
        return []
    return [str(flag).strip() for flag in flags if str(flag).strip()]


def _retrieval_warnings(retrieval_result: dict[str, Any] | None) -> list[str]:
    if not isinstance(retrieval_result, dict):
        return []
    return _clean_warnings(retrieval_result.get("warnings"))


def _is_out_of_scope_context(
    nlu_result: dict[str, Any] | None,
    retrieval_result: dict[str, Any] | None = None,
) -> bool:
    flags = set(_nlu_risk_flags(nlu_result))
    warnings = set(_retrieval_warnings(retrieval_result))
    return (
        _nlu_product_label(nlu_result) == "out_of_scope"
        or "out_of_scope_query" in flags
        or "out_of_scope_query" in warnings
    )


def _out_of_scope_answer_response(*, zh: bool) -> dict[str, Any]:
    if zh:
        return {
            "model_status": "deterministic_guardrail",
            "model_name": "",
            "answer": "这个问题不属于金融问答范围，我不能基于当前系统回答天气、生活服务或其他非金融内容。可以改问金融、市场、经济、资产、产品或行业相关问题。",
            "key_points": [
                "当前查询被识别为非金融问题。",
                "检索结果没有可用的金融证据。",
                "请改问金融、市场、经济或相关证据分析问题。",
            ],
            "evidence_used": [],
            "limitations": ["out_of_scope_query", "查询内容不属于金融范畴", "无相关金融证据可用"],
            "risk_disclaimer": "如果改问金融问题，回答仍仅基于给定证据生成，不构成投资建议、买卖建议或确定性涨跌预测。",
        }
    return {
        "model_status": "deterministic_guardrail",
        "model_name": "",
        "answer": "This question is outside the financial QA scope. I cannot answer weather, lifestyle, or other non-financial topics with this system. Please ask about finance, markets, economics, assets, products, or sectors.",
        "key_points": [
            "The query was classified as out of scope.",
            "No financial evidence was retrieved.",
            "Please reframe the question as a finance, market, economic, or evidence-analysis query.",
        ],
        "evidence_used": [],
        "limitations": ["out_of_scope_query", "The query is outside the financial domain", "No financial evidence is available"],
        "risk_disclaimer": "If you ask a financial question, the answer will still be based only on provided evidence and is not investment, trading, or price-forecast advice.",
    }


def _out_of_scope_next_question_response(*, zh: bool) -> dict[str, Any]:
    questions = (
        [
            "我可以问哪些金融、市场或经济相关问题？",
            "如果要分析一个具体问题，需要补充哪些证据或数据？",
            "如何判断当前证据是否足以支持结论？",
        ]
        if zh
        else [
            "What financial, market, or economic questions can I ask?",
            "What evidence or data should I provide for a specific analysis question?",
            "How can I tell whether the current evidence is enough to support a conclusion?",
        ]
    )
    return {
        "model_status": "deterministic_guardrail",
        "model_name": "",
        "predictions": [
            {"question": question, "score": round(0.92 - index * 0.06, 4), "reason": "finance_reentry_followup"}
            for index, question in enumerate(questions)
        ],
    }


def _retrieval_confidence(retrieval_result: dict[str, Any] | None) -> float | None:
    if not isinstance(retrieval_result, dict):
        return None
    value = retrieval_result.get("retrieval_confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_low_confidence(retrieval_result: dict[str, Any] | None) -> bool:
    confidence = _retrieval_confidence(retrieval_result)
    return confidence is not None and confidence < 0.5


def _has_partial_or_missing_evidence(
    retrieval_result: dict[str, Any] | None,
    statistical_result: dict[str, Any] | None = None,
) -> bool:
    if _is_low_confidence(retrieval_result) or bool(_retrieval_warnings(retrieval_result)):
        return True
    if isinstance(statistical_result, dict):
        overall = statistical_result.get("overall_statistical_summary")
        if isinstance(overall, dict):
            if overall.get("data_sufficiency") in {"partial", "low"} or overall.get("should_abstain"):
                return True
    if isinstance(retrieval_result, dict):
        coverage = retrieval_result.get("coverage")
        if isinstance(coverage, dict) and any(value is False for value in coverage.values()):
            return True
    return False


def _nlu_question_style(nlu_result: dict[str, Any] | None) -> str:
    if not isinstance(nlu_result, dict):
        return ""
    return str(nlu_result.get("question_style") or "").strip().lower()


def _is_advice_or_forecast_context(nlu_result: dict[str, Any] | None) -> bool:
    style = _nlu_question_style(nlu_result)
    flags = set(_nlu_risk_flags(nlu_result))
    return style in {"advice", "forecast"} or "investment_advice_like" in flags


def _is_comparison_judgment_context(nlu_result: dict[str, Any] | None, query: str = "") -> bool:
    if _nlu_question_style(nlu_result) != "compare":
        return False
    lowered = str(query or "").lower()
    markers = ("哪个好", "哪只好", "哪个更好", "更适合", "更稳", "better", "which is better", "which one")
    return any(marker in lowered for marker in markers)


def _is_judgment_context(nlu_result: dict[str, Any] | None, query: str = "") -> bool:
    return _is_advice_or_forecast_context(nlu_result) or _is_comparison_judgment_context(nlu_result, query)


def _needs_conditional_answer_guard(
    nlu_result: dict[str, Any] | None,
    retrieval_result: dict[str, Any] | None,
    statistical_result: dict[str, Any] | None = None,
    query: str = "",
) -> bool:
    if _is_judgment_context(nlu_result, query):
        return True
    style = _nlu_question_style(nlu_result)
    return style == "why" and _has_partial_or_missing_evidence(retrieval_result, statistical_result)


def _conditional_answer_prefix(nlu_result: dict[str, Any] | None, *, zh: bool, query: str = "") -> str:
    if _is_comparison_judgment_context(nlu_result, query):
        return (
            "当前证据不足以直接判断哪个更好，应分维度比较风险、估值、盈利质量和适配条件。"
            if zh
            else "The current evidence is not enough to decide which one is better; compare risk, valuation, earnings quality, and suitability by dimension. "
        )
    if _is_advice_or_forecast_context(nlu_result):
        return (
            "基于当前证据只能做条件性判断，不能据此给出确定的买入、卖出或持有建议。"
            if zh
            else "Based on the current evidence, this can only be a conditional assessment, not a definitive buy, sell, or hold recommendation. "
        )
    return (
        "现有证据不足以把结果归因于单一原因，以下只能作为可能解释。"
        if zh
        else "The available evidence is not enough to attribute the result to a single proven cause; the following should be treated as possible explanations. "
    )


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    seen = {value for value in values}
    for item in additions:
        text = str(item).strip()
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _replace_preserving_case(text: str, source: str, replacement: str) -> str:
    """Replace ``source`` case-insensitively on word boundaries; a capitalised match keeps its capital."""

    def substitute(match: re.Match[str]) -> str:
        found = match.group(0)
        if found[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return re.sub(rf"\b{re.escape(source)}\b", substitute, text, flags=re.IGNORECASE)


def _soften_answer_text(
    text: str,
    nlu_result: dict[str, Any] | None,
    retrieval_result: dict[str, Any] | None,
    statistical_result: dict[str, Any] | None,
    *,
    query: str,
    zh: bool,
) -> str:
    if not text:
        return text
    softened = text
    if zh:
        if (
            _nlu_question_style(nlu_result) == "why"
            or _is_judgment_context(nlu_result, query)
        ) and _has_partial_or_missing_evidence(retrieval_result, statistical_result):
            replacements = (
                ("主要原因是", "可能相关的因素包括"),
                ("核心原因是", "可能相关的因素包括"),
                ("根本原因是", "可能相关的因素包括"),
                ("主要受", "可能受"),
                ("导致", "可能影响"),
                ("证明", "提示"),
                ("形成拖累", "可能形成压力"),
                ("拖累", "可能形成压力"),
            )
            for source, replacement in replacements:
                softened = softened.replace(source, replacement)
        if _is_judgment_context(nlu_result, query):
            judgment_replacements = (
                ("仍然值得继续持有", "是否继续持有需要结合持仓成本、风险承受能力和后续证据判断"),
                ("值得继续持有", "是否继续持有需要结合持仓成本、风险承受能力和后续证据判断"),
                ("短期持有需谨慎", "短期是否持有需要结合持仓成本、风险承受能力和后续证据判断"),
                ("更适合长期持有", "长期适配性仍需结合投资期限和风险承受能力评估"),
                ("适合长期持有", "长期适配性仍需结合投资期限和风险承受能力评估"),
                ("建议买入", "不能据此直接建议买入"),
                ("建议卖出", "不能据此直接建议卖出"),
                ("可以买入", "不能据此直接买入"),
                ("应买入", "不能据此直接买入"),
                ("应卖出", "不能据此直接卖出"),
            )
            for source, replacement in judgment_replacements:
                softened = softened.replace(source, replacement)
            if _is_comparison_judgment_context(nlu_result, query):
                softened = re.sub(r"[^，。；！？]*更好[，,。；;]?", "", softened).strip()
                if not softened:
                    softened = "需要补充实时行情、资金流、公告和估值口径后再做分维度比较。"
        return softened

    if _nlu_question_style(nlu_result) == "why" and _has_partial_or_missing_evidence(retrieval_result, statistical_result):
        replacements = (
            ("the main reason is", "possible related factors include"),
            ("mainly caused by", "possibly related to"),
            ("caused by", "possibly affected by"),
            ("proves that", "suggests that"),
        )
        # replace case-insensitively but leave the rest of the answer untouched: lower-casing the whole
        # text broke company names and evidence ids ([fundamental_600519.SH] -> [fundamental_600519.sh])
        for source, replacement in replacements:
            softened = _replace_preserving_case(softened, source, replacement)
    if _is_judgment_context(nlu_result, query):
        softened = re.sub(r"\b(is|are) better\b", "may compare more favorably on selected dimensions", softened, flags=re.IGNORECASE)
        softened = re.sub(r"\bshould (buy|sell|hold)\b", "should not treat this as a direct trading action", softened, flags=re.IGNORECASE)
    return softened


def _soften_key_points(points: list[str], nlu_result: dict[str, Any] | None, *, query: str, zh: bool) -> list[str]:
    softened: list[str] = []
    for point in points:
        text = point
        if zh and _is_comparison_judgment_context(nlu_result, query):
            text = re.sub(r"[^，。；！？]*更好[，,。；;]?", "需分维度比较", text).strip()
        if zh and _is_judgment_context(nlu_result, query):
            text = text.replace("适合长期持有", "长期适配性需结合风险承受能力评估")
        if text:
            softened.append(text)
    return softened


def _contains_direct_trading_action(question: str, *, zh: bool) -> bool:
    lowered = question.lower()
    if zh:
        markers = ("买入", "卖出", "继续持有", "长期持有", "值得拿", "还能拿", "什么时候卖", "什么时候买", "上车", "止盈", "止损")
    else:
        markers = ("buy", "sell", "hold", "entry point", "exit point", "stop loss", "take profit")
    return any(marker in lowered for marker in markers)


def _safe_judgment_followups(*, zh: bool) -> list[dict[str, Any]]:
    questions = (
        [
            "需要补充哪些证据或数据才能评估这个问题？",
            "当前判断最需要关注哪些风险和限制？",
            "不同风险承受能力下应如何理解当前证据？",
        ]
        if zh
        else [
            "What evidence or data is still needed to evaluate this question?",
            "What risks and limitations matter most for the current assessment?",
            "How should the current evidence be interpreted under different risk tolerances?",
        ]
    )
    return [
        {"question": question, "score": round(0.88 - index * 0.05, 4), "reason": "evidence_boundary_followup"}
        for index, question in enumerate(questions)
    ]


def _safe_causal_followups(*, zh: bool) -> list[dict[str, Any]]:
    questions = (
        [
            "哪些因素只是相关信息，哪些需要进一步验证？",
            "还需要哪些实时数据来验证这些解释？",
            "哪些证据能区分相关性和因果性？",
        ]
        if zh
        else [
            "Which factors are only related signals and which need more validation?",
            "What real-time data is still needed to validate these explanations?",
            "What evidence would distinguish correlation from causation?",
        ]
    )
    return [
        {"question": question, "score": round(0.88 - index * 0.05, 4), "reason": "causal_evidence_followup"}
        for index, question in enumerate(questions)
    ]


def _contains_strong_causal_claim(question: str, *, zh: bool) -> bool:
    lowered = question.lower()
    if zh:
        markers = ("导致", "拖累", "主要原因", "核心原因", "根本原因")
    else:
        markers = ("caused by", "main reason", "key reason", "drove", "dragged")
    return any(marker in lowered for marker in markers)


def _sanitize_next_questions_for_context(
    predictions: list[dict[str, Any]],
    *,
    query: str,
    nlu_result: dict[str, Any] | None,
    retrieval_result: dict[str, Any] | None = None,
    statistical_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    zh = _is_zh(query)
    style = _nlu_question_style(nlu_result)
    should_soften_causal = style == "why" and _has_partial_or_missing_evidence(retrieval_result, statistical_result)
    if not _is_judgment_context(nlu_result, query) and not should_soften_causal:
        return predictions

    safe_fillers = _safe_causal_followups(zh=zh) if should_soften_causal else _safe_judgment_followups(zh=zh)
    sanitized: list[dict[str, Any]] = []
    for item in predictions:
        question = str(item.get("question") or "")
        if _is_judgment_context(nlu_result, query) and _contains_direct_trading_action(question, zh=zh):
            continue
        if should_soften_causal and _contains_strong_causal_claim(question, zh=zh):
            continue
        sanitized.append(item)

    existing_questions = {item["question"] for item in sanitized}
    for filler in safe_fillers:
        if len(sanitized) >= 3:
            break
        if filler["question"] not in existing_questions:
            sanitized.append(filler)
            existing_questions.add(filler["question"])
    return sanitized[:3]


def _missing_evidence_note(nlu_result: dict[str, Any] | None, query: str, *, zh: bool) -> str | None:
    style = _nlu_question_style(nlu_result)
    if style == "why" or _is_judgment_context(nlu_result, query):
        return (
            "缺少实时成交、资金流或公告等验证"
            if zh
            else "Real-time trading, fund-flow, or announcement evidence is missing"
        )
    return None


def _guardrail_limitations(
    retrieval_result: dict[str, Any] | None,
    nlu_result: dict[str, Any] | None,
    statistical_result: dict[str, Any] | None = None,
    *,
    zh: bool,
    query: str = "",
) -> list[str]:
    limitations = _retrieval_warnings(retrieval_result)
    if _is_low_confidence(retrieval_result):
        limitations.append("检索置信度较低" if zh else "Retrieval confidence is low")
    if isinstance(statistical_result, dict):
        overall = statistical_result.get("overall_statistical_summary")
        if isinstance(overall, dict) and overall.get("data_sufficiency") in {"partial", "low"}:
            limitations.append("数据覆盖不完整" if zh else "Data coverage is incomplete")
        if isinstance(overall, dict) and overall.get("should_abstain"):
            limitations.append("统计结果提示应谨慎回答" if zh else "The statistical result suggests caution or abstention")
    if _has_partial_or_missing_evidence(retrieval_result, statistical_result):
        missing_note = _missing_evidence_note(nlu_result, query, zh=zh)
        if missing_note:
            limitations.append(missing_note)
    if _needs_conditional_answer_guard(nlu_result, retrieval_result, statistical_result, query):
        if _is_advice_or_forecast_context(nlu_result):
            limitations.append("问题包含投资建议或预测属性" if zh else "The query has advice or forecast-like risk")
        elif _is_comparison_judgment_context(nlu_result, query):
            limitations.append("当前证据不足以直接判断优劣" if zh else "The evidence is insufficient to rank the options directly")
        else:
            limitations.append("因果解释证据置信度有限" if zh else "Evidence confidence is limited for causal explanation")
    return list(dict.fromkeys(limitations))


def _question_style_from_payload(payload: dict[str, Any]) -> str:
    nlu = payload.get("nlu_result") if isinstance(payload.get("nlu_result"), dict) else {}
    style = str(nlu.get("question_style") or "").strip().lower()
    if style in {"fact", "why", "advice", "compare", "forecast"}:
        return style
    query = str(payload.get("query") or "").lower()
    if any(term in query for term in ("compare", "which", "vs", "versus", "哪个", "对比", "相比")):
        return "compare"
    if any(term in query for term in ("why", "driven", "cause", "impact", "为什么", "原因", "影响", "异动")):
        return "why"
    if any(term in query for term in ("suitable", "should i", "advice", "适合", "能不能买", "定投")):
        return "advice"
    if any(term in query for term in ("forecast", "upside", "downside", "上涨", "下跌", "还能", "压力")):
        return "forecast"
    return "fact"


def _clean_warnings(warnings: Any) -> list[str]:
    if not isinstance(warnings, list):
        return []
    cleaned = []
    for warning in warnings:
        text = str(warning)
        if "synthetic" in text.lower():
            continue
        cleaned.append(text)
    return cleaned
