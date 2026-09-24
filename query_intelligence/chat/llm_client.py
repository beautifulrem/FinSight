"""DeepSeek (OpenAI-compatible) client that rewrites compact evidence into a JSON answer."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .config import DEFAULT_CHATBOT_CONFIG
from .language import _default_risk_disclaimer, _target_language_name, answer_matches_language, detect_query_language


class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, config: dict[str, Any], *, http_client: Any | None = None) -> None:
        deepseek = config.get("deepseek") or {}
        self.base_url = str(deepseek.get("base_url") or DEFAULT_CHATBOT_CONFIG["deepseek"]["base_url"]).rstrip("/")
        self.chat_path = str(deepseek.get("chat_path") or DEFAULT_CHATBOT_CONFIG["deepseek"]["chat_path"])
        self.model = str(deepseek.get("model") or DEFAULT_CHATBOT_CONFIG["deepseek"]["model"])
        self.api_key = str(deepseek.get("api_key") or "")
        self.timeout_seconds = int(
            deepseek.get("timeout_seconds") or DEFAULT_CHATBOT_CONFIG["deepseek"]["timeout_seconds"]
        )
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
        except Exception as exc:
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
                "All natural-language strings in answer, key_points, and risk_disclaimer "
                f"must be in {target_language}. "
                "If the target language is Chinese, use Simplified Chinese. If it is English, write fluent English "
                "even when company names or source names in the evidence are Chinese."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


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


def normalize_llm_answer(output: dict[str, Any], record: dict[str, Any], *, model: str) -> dict[str, Any]:
    answer = str(output.get("answer") or "").strip()
    if not answer:
        raise DeepSeekError(f"{model} returned an empty answer")
    key_points_raw = output.get("key_points") or []
    key_points = (
        [str(item).strip() for item in key_points_raw if str(item).strip()] if isinstance(key_points_raw, list) else []
    )
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


def _evidence_ids(record: dict[str, Any]) -> set[str]:
    retrieval = record.get("retrieval_result") or {}
    ids: set[str] = set()
    for item in (retrieval.get("documents") or []) + (retrieval.get("structured_data") or []):
        evidence_id = str(item.get("evidence_id") or "").strip()
        if evidence_id:
            ids.add(evidence_id)
    return ids
