"""Answer assembly for /chat: normalization, template fallback, freshness guard, evidence sources."""

from __future__ import annotations

from datetime import date
from typing import Any

from .language import _default_risk_disclaimer, detect_query_language
from .llm_client import DeepSeekClient, _evidence_ids


def build_chatbot_response(
    *,
    query: str,
    pipeline_result: dict[str, Any],
    deepseek_client: DeepSeekClient,
    progress: Any | None = None,
) -> dict[str, Any]:
    record = {
        "status": "ok",
        "query": query,
        "nlu_result": pipeline_result["nlu_result"],
        "retrieval_result": pipeline_result["retrieval_result"],
    }
    try:
        if progress:
            progress("DeepSeek: sending compact evidence for response polishing...")
        answer = deepseek_client.generate(record)
        if progress:
            progress("DeepSeek: response received and normalized.")
        llm_status = {"provider": "deepseek", "model": deepseek_client.model, "status": "ok", "error": None}
    except Exception as exc:
        if progress:
            progress(f"DeepSeek: unavailable; using structured-summary fallback. reason={exc}")
        answer = template_answer(record, fallback_reason=str(exc))
        llm_status = {"provider": "deepseek", "model": deepseek_client.model, "status": "fallback", "error": str(exc)}
    if progress:
        progress("Step 3/3: applying market freshness guard and formatting evidence sources...")
    answer = apply_market_freshness_guard(answer, record)
    evidence_sources = build_evidence_sources(record, answer.get("evidence_used") or [])
    if progress:
        progress(f"Step 3/3 complete: evidence_sources={len(evidence_sources)}")
    return {
        **answer,
        "evidence_sources": evidence_sources,
        "llm": llm_status,
        "nlu_result": pipeline_result["nlu_result"],
        "retrieval_result": pipeline_result["retrieval_result"],
    }


def _template_key_points(
    *,
    language: str,
    analysis_summary: dict[str, Any],
    structured_data: list[Any],
    documents: list[Any],
    warnings: list[Any],
) -> list[str]:
    key_points: list[str] = []
    if analysis_summary:
        readiness = analysis_summary.get("data_readiness") or {}
        ready_labels = [label for label, ready in readiness.items() if isinstance(ready, bool) and ready]
        if language == "en":
            coverage = ", ".join(ready_labels) if ready_labels else "basic evidence"
            key_points.append(f"Generated a structured analysis summary covering: {coverage}.")
        else:
            key_points.append(
                f"已生成结构化分析摘要，覆盖：{', '.join(ready_labels) if ready_labels else '基础证据'}。"
            )
    if structured_data:
        if language == "en":
            key_points.append(f"Loaded {len(structured_data)} structured data item(s).")
        else:
            key_points.append(f"已读取 {len(structured_data)} 条结构化数据。")
    if documents:
        source_types = sorted({str(doc.get("source_type") or "document") for doc in documents if isinstance(doc, dict)})
        if language == "en":
            source_text = ", ".join(source_types) if source_types else "documents"
            key_points.append(
                f"Retrieved {len(documents)} text evidence item(s), including source types: {source_text}."
            )
        else:
            key_points.append(f"已检索 {len(documents)} 条文本证据，来源类型包括：{', '.join(source_types)}。")
    if warnings:
        if language == "en":
            key_points.append("Data warnings are present; inspect retrieval_result.warnings for details.")
        else:
            key_points.append(f"数据提示：{'; '.join(str(item) for item in warnings[:3])}。")
    if not key_points:
        if language == "en":
            key_points.append(
                "Available evidence is limited; consider adding a clearer target, time range, or data source."
            )
        else:
            key_points.append("当前可用证据有限，建议补充更明确的标的、时间范围或数据源。")
    return key_points


def template_answer(record: dict[str, Any], *, fallback_reason: str | None = None) -> dict[str, Any]:
    query = str(record.get("query") or "")
    retrieval = record.get("retrieval_result") or {}
    documents = retrieval.get("documents") or []
    structured_data = retrieval.get("structured_data") or []
    warnings = retrieval.get("warnings") or []
    analysis_summary = retrieval.get("analysis_summary") or {}

    language = detect_query_language(query)
    key_points = _template_key_points(
        language=language,
        analysis_summary=analysis_summary,
        structured_data=structured_data,
        documents=documents,
        warnings=warnings,
    )

    if language == "en":
        prefix = "Model refinement failed; the following is a structured summary."
        if fallback_reason and "API key" in fallback_reason:
            prefix = "DeepSeek API is not configured; the following is a structured summary."
        answer = (
            f'{prefix}\n\nFor "{query}", the system completed financial question understanding '
            f"and evidence retrieval. {' '.join(key_points)}"
        )
    else:
        prefix = "模型润色失败，以下为结构化摘要。"
        if fallback_reason and "API key" in fallback_reason:
            prefix = "DeepSeek API 未配置，以下为结构化摘要。"
        answer = f"针对“{query}”，系统已完成金融问题理解和证据检索。{''.join(key_points)}"
        answer = f"{prefix}\n\n{answer}"
    return {
        "answer": answer,
        "key_points": key_points,
        "risk_disclaimer": _default_risk_disclaimer(record),
        "evidence_used": sorted(_evidence_ids(record))[:8],
    }


def apply_market_freshness_guard(answer: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    query = str(record.get("query") or (record.get("nlu_result") or {}).get("raw_query") or "")
    if not _asks_for_current_market_data(query):
        return answer

    language = detect_query_language(query)
    today_date = date.today()
    today = today_date.isoformat()
    market_item = _first_market_item(record)
    if _is_known_non_trading_day(today_date):
        return _non_trading_day_market_answer(
            answer=answer,
            market_item=market_item,
            language=language,
            today=today,
        )
    if not market_item:
        guarded = dict(answer)
        if language == "en":
            guarded["answer"] = (
                f"Unable to retrieve today's ({today}) real-time market data, so I cannot determine "
                "whether it rose or fell today. The current response can only provide background "
                "based on non-real-time evidence."
            )
            key_point = f"Today's ({today}) real-time quote was not retrieved."
        else:
            guarded["answer"] = (
                f"未获取到今日（{today}）实时行情，因此不能判断今天是否上涨或下跌。"
                "当前回复仅能基于非实时证据做背景说明。"
            )
            key_point = f"今日（{today}）实时行情获取失败。"
        guarded["key_points"] = _prepend_unique(
            answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
            key_point,
        )
        guarded["evidence_used"] = []
        return guarded

    payload = market_item.get("payload") if isinstance(market_item.get("payload"), dict) else {}
    trade_date = str(payload.get("trade_date") or market_item.get("as_of") or "").strip()
    if not trade_date or trade_date[:10] == today:
        return answer

    symbol = payload.get("symbol") or market_item.get("evidence_id") or "该标的"
    close = payload.get("close") if payload.get("close") is not None else payload.get("price")
    pct_change = payload.get("pct_change_1d")
    guarded = dict(answer)
    if language == "en":
        close_text = f"; latest available price/close is {close}" if close is not None else ""
        pct_text = f"; percent change is {pct_change}%" if pct_change is not None else ""
        guarded["answer"] = (
            f"Unable to retrieve today's ({today}) real-time quote; the latest available market date "
            f"is {trade_date[:10]}. {symbol}{close_text}{pct_text}. Therefore, I cannot use this data "
            f"to determine whether it rose or fell today ({today})."
        )
        key_point = f"Today's quote was not retrieved; the latest available market date is {trade_date[:10]}."
    else:
        close_text = f"，最新可用价格/收盘价为 {close}" if close is not None else ""
        pct_text = f"，涨跌幅为 {pct_change}%" if pct_change is not None else ""
        guarded["answer"] = (
            f"未获取到今日（{today}）实时行情；系统最新可用行情日期是 {trade_date[:10]}。"
            f"{symbol}{close_text}{pct_text}。"
            f"因此不能据此判断今天（{today}）是否上涨或下跌。"
        )
        key_point = f"今日行情未获取成功，最新可用行情日期为 {trade_date[:10]}。"
    guarded["key_points"] = _prepend_unique(
        answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
        key_point,
    )
    guarded["evidence_used"] = [str(market_item.get("evidence_id"))] if market_item.get("evidence_id") else []
    return guarded


def _non_trading_day_market_answer(
    *,
    answer: dict[str, Any],
    market_item: dict[str, Any] | None,
    language: str,
    today: str,
) -> dict[str, Any]:
    guarded = dict(answer)
    if not market_item:
        if language == "en":
            guarded["answer"] = (
                f"Today ({today}) is not a regular A-share trading day, so no same-day market move "
                "is expected. I also could not retrieve a recent trading-day quote for this request."
            )
            key_point = f"Today ({today}) is not a regular A-share trading day."
        else:
            guarded["answer"] = (
                f"今天（{today}）不是 A 股常规交易日，因此没有当日涨跌行情。本次请求也未获取到最近交易日行情。"
            )
            key_point = f"今天（{today}）不是 A 股常规交易日。"
        guarded["key_points"] = _prepend_unique(
            answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
            key_point,
        )
        guarded["evidence_used"] = []
        return guarded

    payload = market_item.get("payload") if isinstance(market_item.get("payload"), dict) else {}
    trade_date = str(payload.get("trade_date") or market_item.get("as_of") or "").strip()
    symbol = payload.get("symbol") or market_item.get("evidence_id") or ("the target" if language == "en" else "该标的")
    close = payload.get("close") if payload.get("close") is not None else payload.get("price")
    pct_change = payload.get("pct_change_1d")
    if language == "en":
        date_text = f" The latest available trading-day quote is from {trade_date[:10]}." if trade_date else ""
        close_text = f" Latest available price/close: {close}." if close is not None else ""
        pct_text = f" Change on that trading day: {pct_change}%." if pct_change is not None else ""
        guarded["answer"] = (
            f"Today ({today}) is not a regular A-share trading day, so there is no same-day trading move. "
            f"{symbol}.{date_text}{close_text}{pct_text}"
        )
        key_point = (
            f"Today ({today}) is not a regular A-share trading day; using the latest available trading-day quote."
        )
    else:
        date_text = f"最新可用交易日行情日期为 {trade_date[:10]}。" if trade_date else ""
        close_text = f"最新可用价格/收盘价为 {close}。" if close is not None else ""
        pct_text = f"该交易日涨跌幅为 {pct_change}%。" if pct_change is not None else ""
        guarded["answer"] = (
            f"今天（{today}）不是 A 股常规交易日，因此没有当日涨跌行情。{symbol}。{date_text}{close_text}{pct_text}"
        )
        key_point = f"今天（{today}）不是 A 股常规交易日，已使用最新可用交易日行情。"
    guarded["key_points"] = _prepend_unique(
        answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
        key_point,
    )
    guarded["evidence_used"] = [str(market_item.get("evidence_id"))] if market_item.get("evidence_id") else []
    return guarded


def build_evidence_sources(
    record: dict[str, Any], evidence_used: list[str] | None = None, *, limit: int = 8
) -> list[dict[str, Any]]:
    retrieval = record.get("retrieval_result") or {}
    items = (retrieval.get("documents") or []) + (retrieval.get("structured_data") or [])
    by_id = {str(item.get("evidence_id") or ""): item for item in items if str(item.get("evidence_id") or "").strip()}
    selected: list[dict[str, Any]] = []
    for evidence_id in evidence_used or []:
        item = by_id.get(str(evidence_id))
        if item:
            selected.append(item)
    if not selected:
        selected = items[:limit]
    return [_source_display_item(item) for item in selected[:limit]]


def _asks_for_current_market_data(query: str) -> bool:
    lower_query = query.lower()
    return any(term in query for term in ("今天", "今日", "现在", "当前", "实时", "最新")) or any(
        term in lower_query
        for term in (
            "today",
            "current",
            "right now",
            "now",
            "real-time",
            "realtime",
            "latest",
        )
    )


def _is_known_non_trading_day(day: date) -> bool:
    return day.weekday() >= 5 or (day.month, day.day) in {
        (1, 1),
        (5, 1),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
    }


def _first_market_item(record: dict[str, Any]) -> dict[str, Any] | None:
    retrieval = record.get("retrieval_result") or {}
    for item in retrieval.get("structured_data") or []:
        if item.get("source_type") == "market_api":
            return item
    return None


def _prepend_unique(items: list[Any], item: str) -> list[str]:
    normalized = [str(value) for value in items if str(value).strip()]
    return [item] + [value for value in normalized if value != item]


def _source_display_item(item: dict[str, Any]) -> dict[str, Any]:
    source_name = item.get("source_name") or item.get("provider") or item.get("source_type")
    title = item.get("title") or item.get("summary") or item.get("evidence_id") or source_name
    source_url = item.get("source_url")
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    if not source_url:
        source_url = payload.get("source_url") or payload.get("url")
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_name": source_name,
        "title": title,
        "source_url": source_url,
    }
