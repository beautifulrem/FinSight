from __future__ import annotations

import copy
import html
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "app_config.json"
DEFAULT_RISK_DISCLAIMER_ZH = "以上内容仅基于系统检索到的证据生成，不构成投资建议或确定性买卖结论。"
DEFAULT_RISK_DISCLAIMER_EN = (
    "This answer is based only on evidence retrieved by the system and is not investment advice "
    "or a deterministic buy/sell conclusion."
)
DEFAULT_RISK_DISCLAIMER = DEFAULT_RISK_DISCLAIMER_ZH

DEFAULT_CHATBOT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "ui": {
        "title": "FinSight Financial Research Assistant",
        "input_placeholder": "Ask a financial question, e.g. What do you think about Ping An Insurance (601318.SH)?",
        "submit_text": "Submit",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "chat_path": "/chat/completions",
        "model": "deepseek-v4-flash",
        "api_key": "",
        "timeout_seconds": 60,
        "thinking_type": "enabled",
        "reasoning_effort": "high",
        "max_tokens": 8192,
    },
    "live_data": {
        "enabled": True,
    },
}


class DeepSeekError(RuntimeError):
    pass


def load_chatbot_config(
    config_path: str | Path | None = None,
    *,
    load_env_file: bool = True,
) -> dict[str, Any]:
    if load_env_file:
        _load_dotenv(ROOT / ".env")

    path = Path(os.getenv("FINANCIAL_CHATBOT_CONFIG") or config_path or DEFAULT_CONFIG_PATH)
    config = copy.deepcopy(DEFAULT_CHATBOT_CONFIG)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"chatbot config must be a JSON object: {path}")
        _deep_merge(config, loaded)

    _apply_env_overrides(config)
    _coerce_config_types(config)
    return config


def apply_live_data_env(config: dict[str, Any]) -> None:
    enabled = bool((config.get("live_data") or {}).get("enabled", True))
    value = "1" if enabled else "0"
    for name in (
        "QI_USE_LIVE_MARKET",
        "QI_USE_LIVE_NEWS",
        "QI_USE_LIVE_ANNOUNCEMENT",
        "QI_USE_LIVE_MACRO",
    ):
        os.environ.setdefault(name, value)


STATIC_DIR = ROOT / "query_intelligence" / "web" / "static"


def render_index_html(config: dict[str, Any]) -> str:
    """Render the browser page from ``web/static/index.html`` (CSS/JS are served from ``/static``)."""
    ui = config.get("ui") or {}
    values = {
        "title": str(ui.get("title") or DEFAULT_CHATBOT_CONFIG["ui"]["title"]),
        "placeholder": str(ui.get("input_placeholder") or DEFAULT_CHATBOT_CONFIG["ui"]["input_placeholder"]),
        "submit_text": str(ui.get("submit_text") or DEFAULT_CHATBOT_CONFIG["ui"]["submit_text"]),
    }
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", html.escape(value))
    return page


class DeepSeekClient:
    def __init__(self, config: dict[str, Any], *, http_client: Any | None = None) -> None:
        deepseek = config.get("deepseek") or {}
        self.base_url = str(deepseek.get("base_url") or DEFAULT_CHATBOT_CONFIG["deepseek"]["base_url"]).rstrip("/")
        self.chat_path = str(deepseek.get("chat_path") or DEFAULT_CHATBOT_CONFIG["deepseek"]["chat_path"])
        self.model = str(deepseek.get("model") or DEFAULT_CHATBOT_CONFIG["deepseek"]["model"])
        self.api_key = str(deepseek.get("api_key") or "")
        self.timeout_seconds = int(deepseek.get("timeout_seconds") or DEFAULT_CHATBOT_CONFIG["deepseek"]["timeout_seconds"])
        self.thinking_type = str(deepseek.get("thinking_type") or "").strip()
        self.reasoning_effort = str(deepseek.get("reasoning_effort") or "").strip()
        max_tokens = deepseek.get("max_tokens")
        self.max_tokens = int(max_tokens) if max_tokens not in {None, ""} else None
        self.http_client = http_client

    def generate(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self._has_api_key():
            raise DeepSeekError("DeepSeek API key is not configured")

        payload = compact_evidence_payload(record)
        query = str(payload.get("query") or "")
        response = self._post_chat_completion(make_answer_messages(payload))
        content = response["choices"][0]["message"]["content"]
        parsed = _parse_json_object(content)
        if not answer_matches_language(parsed, query):
            response = self._post_chat_completion(make_answer_language_repair_messages(query, parsed))
            content = response["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            if not answer_matches_language(parsed, query):
                raise DeepSeekError("DeepSeek response language did not match the query language")
        return normalize_llm_answer(parsed, record, model=self.model)

    def _post_chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url}{self.chat_path if self.chat_path.startswith('/') else '/' + self.chat_path}"
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self.thinking_type:
            body["thinking"] = {"type": self.thinking_type}
        if self.reasoning_effort and self.thinking_type != "disabled":
            body["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.thinking_type != "enabled":
            body["temperature"] = 0.2
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.http_client is None
        try:
            result = client.post(url, headers=headers, json=body)
            result.raise_for_status()
            data = result.json()
        except Exception as exc:  # noqa: BLE001
            raise DeepSeekError(f"DeepSeek API request failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if not isinstance(data, dict) or not data.get("choices"):
            raise DeepSeekError("DeepSeek API response is missing choices")
        return data

    def _has_api_key(self) -> bool:
        stripped = self.api_key.strip()
        return bool(stripped and stripped.lower() not in {"your_deepseek_api_key_here", "changeme"})


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
    except Exception as exc:  # noqa: BLE001
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


def compact_evidence_payload(record: dict[str, Any]) -> dict[str, Any]:
    retrieval = record.get("retrieval_result") or {}
    documents = retrieval.get("documents") or []
    structured_data = retrieval.get("structured_data") or []
    query = record.get("query") or (record.get("nlu_result") or {}).get("raw_query")
    response_language = detect_query_language(str(query or ""))
    return {
        "query": query,
        "response_language": response_language,
        "nlu_result": {
            "question_style": (record.get("nlu_result") or {}).get("question_style"),
            "product_type": (record.get("nlu_result") or {}).get("product_type"),
            "intent_labels": (record.get("nlu_result") or {}).get("intent_labels"),
            "topic_labels": (record.get("nlu_result") or {}).get("topic_labels"),
            "entities": (record.get("nlu_result") or {}).get("entities"),
            "risk_flags": (record.get("nlu_result") or {}).get("risk_flags"),
        },
        "retrieval_result": {
            "retrieval_confidence": retrieval.get("retrieval_confidence"),
            "warnings": retrieval.get("warnings") or [],
            "coverage": retrieval.get("coverage") or {},
            "analysis_summary": retrieval.get("analysis_summary") or {},
            "structured_data": [_compact_structured_item(item) for item in structured_data[:10]],
            "documents": [_compact_document(item) for item in documents[:8]],
        },
        "output_contract": {
            "answer": "string",
            "key_points": ["string"],
            "risk_disclaimer": "string",
            "evidence_used": ["evidence_id"],
        },
    }


def make_answer_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    query = str(payload.get("query") or "")
    target_language = _target_language_name(detect_query_language(query))
    return [
        {
            "role": "system",
            "content": (
                "You are the response-polishing layer for a financial chatbot. "
                "Answer only from the JSON evidence provided by the user. Do not invent market prices, "
                "financial data, news, macro facts, statistics, or investment conclusions. "
                "Return exactly one strict JSON object with these keys: "
                "answer, key_points, risk_disclaimer, evidence_used. "
                "Preserve evidence_used IDs exactly and do not add IDs that are absent from the evidence. "
                f"All natural-language strings in answer, key_points, and risk_disclaimer must be in {target_language}. "
                "If the target language is Chinese, use Simplified Chinese. If it is English, write fluent English "
                "even when company names or source names in the evidence are Chinese."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def answer_matches_language(output: dict[str, Any], query: str) -> bool:
    expected_language = detect_query_language(query)
    text_values: list[str] = []
    for key in ("answer", "risk_disclaimer"):
        value = str(output.get(key) or "").strip()
        if value:
            text_values.append(value)
    for key in ("key_points", "limitations"):
        values = output.get(key)
        if isinstance(values, list):
            text_values.extend(str(item).strip() for item in values if str(item).strip())
    return all(_text_matches_language(value, expected_language) for value in text_values)


def make_answer_language_repair_messages(query: str, output: dict[str, Any]) -> list[dict[str, str]]:
    target_language = _target_language_name(detect_query_language(query))
    return [
        {
            "role": "system",
            "content": (
                "You rewrite answer-generation JSON into the user's language. "
                "Return only one valid JSON object with exactly these keys: "
                "answer, key_points, risk_disclaimer, evidence_used. "
                "Preserve the financial meaning, risk caution, and evidence_used IDs. "
                f"All natural-language strings must be in {target_language}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "target_language": target_language,
                    "current_output": output,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def detect_query_language(text: str) -> str:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk_count and cjk_count >= max(2, latin_count * 0.4):
        return "zh"
    if latin_count:
        return "en"
    if cjk_count:
        return "zh"
    return "zh"


def _target_language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"


def _default_risk_disclaimer(record: dict[str, Any]) -> str:
    query = str(record.get("query") or (record.get("nlu_result") or {}).get("raw_query") or "")
    return DEFAULT_RISK_DISCLAIMER_EN if detect_query_language(query) == "en" else DEFAULT_RISK_DISCLAIMER_ZH


def _text_matches_language(text: str, expected_language: str) -> bool:
    signal = _language_signal(text)
    if signal in {"neutral", "mixed"}:
        return True
    if expected_language == "en":
        return signal != "zh"
    return signal != "en"


def _language_signal(text: str) -> str:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk_count == 0 and latin_count < 4:
        return "neutral"
    if cjk_count >= max(2, int(latin_count * 0.2)):
        return "zh"
    if latin_count >= max(4, cjk_count * 4):
        return "en"
    return "mixed"


def normalize_llm_answer(output: dict[str, Any], record: dict[str, Any], *, model: str) -> dict[str, Any]:
    answer = str(output.get("answer") or "").strip()
    if not answer:
        raise DeepSeekError(f"{model} returned an empty answer")
    key_points_raw = output.get("key_points") or []
    key_points = [str(item).strip() for item in key_points_raw if str(item).strip()] if isinstance(key_points_raw, list) else []
    evidence_used_raw = output.get("evidence_used") or []
    allowed_ids = _evidence_ids(record)
    evidence_used = [
        str(item).strip()
        for item in evidence_used_raw
        if str(item).strip() and (not allowed_ids or str(item).strip() in allowed_ids)
    ]
    return {
        "answer": answer,
        "key_points": key_points[:6],
        "risk_disclaimer": str(output.get("risk_disclaimer") or _default_risk_disclaimer(record)).strip(),
        "evidence_used": evidence_used[:8],
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
            key_points.append(f"已生成结构化分析摘要，覆盖：{', '.join(ready_labels) if ready_labels else '基础证据'}。")
    if structured_data:
        if language == "en":
            key_points.append(f"Loaded {len(structured_data)} structured data item(s).")
        else:
            key_points.append(f"已读取 {len(structured_data)} 条结构化数据。")
    if documents:
        source_types = sorted({str(doc.get("source_type") or "document") for doc in documents if isinstance(doc, dict)})
        if language == "en":
            source_text = ", ".join(source_types) if source_types else "documents"
            key_points.append(f"Retrieved {len(documents)} text evidence item(s), including source types: {source_text}.")
        else:
            key_points.append(f"已检索 {len(documents)} 条文本证据，来源类型包括：{', '.join(source_types)}。")
    if warnings:
        if language == "en":
            key_points.append("Data warnings are present; inspect retrieval_result.warnings for details.")
        else:
            key_points.append(f"数据提示：{'; '.join(str(item) for item in warnings[:3])}。")
    if not key_points:
        if language == "en":
            key_points.append("Available evidence is limited; consider adding a clearer target, time range, or data source.")
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
                f"今天（{today}）不是 A 股常规交易日，因此没有当日涨跌行情。"
                "本次请求也未获取到最近交易日行情。"
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
        key_point = f"Today ({today}) is not a regular A-share trading day; using the latest available trading-day quote."
    else:
        date_text = f"最新可用交易日行情日期为 {trade_date[:10]}。" if trade_date else ""
        close_text = f"最新可用价格/收盘价为 {close}。" if close is not None else ""
        pct_text = f"该交易日涨跌幅为 {pct_change}%。" if pct_change is not None else ""
        guarded["answer"] = (
            f"今天（{today}）不是 A 股常规交易日，因此没有当日涨跌行情。"
            f"{symbol}。{date_text}{close_text}{pct_text}"
        )
        key_point = f"今天（{today}）不是 A 股常规交易日，已使用最新可用交易日行情。"
    guarded["key_points"] = _prepend_unique(
        answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
        key_point,
    )
    guarded["evidence_used"] = [str(market_item.get("evidence_id"))] if market_item.get("evidence_id") else []
    return guarded


def build_evidence_sources(record: dict[str, Any], evidence_used: list[str] | None = None, *, limit: int = 8) -> list[dict[str, Any]]:
    retrieval = record.get("retrieval_result") or {}
    items = (retrieval.get("documents") or []) + (retrieval.get("structured_data") or [])
    by_id = {
        str(item.get("evidence_id") or ""): item
        for item in items
        if str(item.get("evidence_id") or "").strip()
    }
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


def _compact_document(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_name": item.get("source_name"),
        "title": item.get("title"),
        "summary": item.get("summary"),
        "text_excerpt": item.get("text_excerpt"),
        "publish_time": item.get("publish_time"),
        "source_url": item.get("source_url"),
    }


def _compact_structured_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") or {}
    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if key not in {"history", "raw", "rows"} and not str(key).startswith("_debug")
        }
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_name": item.get("source_name"),
        "provider": item.get("provider"),
        "as_of": item.get("as_of"),
        "quality_flags": item.get("quality_flags") or [],
        "payload": payload,
    }


def _evidence_ids(record: dict[str, Any]) -> set[str]:
    retrieval = record.get("retrieval_result") or {}
    ids: set[str] = set()
    for item in (retrieval.get("documents") or []) + (retrieval.get("structured_data") or []):
        evidence_id = str(item.get("evidence_id") or "").strip()
        if evidence_id:
            ids.add(evidence_id)
    return ids


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise DeepSeekError("DeepSeek response did not contain a JSON object") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise DeepSeekError(f"DeepSeek response JSON parse failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DeepSeekError("DeepSeek response JSON must be an object")
    return parsed


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _apply_env_overrides(config: dict[str, Any]) -> None:
    mappings = {
        "CHATBOT_HOST": ("server", "host"),
        "CHATBOT_PORT": ("server", "port"),
        "CHATBOT_TITLE": ("ui", "title"),
        "CHATBOT_INPUT_PLACEHOLDER": ("ui", "input_placeholder"),
        "CHATBOT_SUBMIT_TEXT": ("ui", "submit_text"),
        "DEEPSEEK_BASE_URL": ("deepseek", "base_url"),
        "DEEPSEEK_CHAT_PATH": ("deepseek", "chat_path"),
        "DEEPSEEK_MODEL": ("deepseek", "model"),
        "DEEPSEEK_API_KEY": ("deepseek", "api_key"),
        "DEEPSEEK_TIMEOUT_SECONDS": ("deepseek", "timeout_seconds"),
        "DEEPSEEK_THINKING_TYPE": ("deepseek", "thinking_type"),
        "DEEPSEEK_REASONING_EFFORT": ("deepseek", "reasoning_effort"),
        "DEEPSEEK_MAX_TOKENS": ("deepseek", "max_tokens"),
        "CHATBOT_LIVE_DATA": ("live_data", "enabled"),
    }
    for env_name, path in mappings.items():
        if env_name in os.environ:
            section, key = path
            config.setdefault(section, {})[key] = os.environ[env_name]


def _coerce_config_types(config: dict[str, Any]) -> None:
    config["server"]["port"] = int(config["server"]["port"])
    config["deepseek"]["timeout_seconds"] = int(config["deepseek"]["timeout_seconds"])
    max_tokens = config["deepseek"].get("max_tokens")
    config["deepseek"]["max_tokens"] = int(max_tokens) if max_tokens not in {None, ""} else None
    config["live_data"]["enabled"] = _parse_bool(config["live_data"].get("enabled", True))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
