from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..label_maps import autolabel_intents, autolabel_question_style, autolabel_topics, map_tnews_product_type
from .intent_autolabel import build_autolabeled_classification_row


def _load_records(raw: Any) -> list[Mapping[str, Any]]:
    if isinstance(raw, Path):
        return _load_records_from_path(raw)
    if isinstance(raw, str):
        path = Path(raw)
        if path.exists():
            return _load_records_from_path(path)
        return [{"text": raw}]
    if isinstance(raw, Mapping):
        for key in ("records", "data", "items", "examples", "samples", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _load_records_from_path(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson", ".json"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
            return _load_records(payload)
        rows: list[Mapping[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                rows.append(item)
        return rows
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            return [row for row in reader]
    return []


def _load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_text(record: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_present(record: Mapping[str, Any], fields: tuple[str, ...]) -> Any | None:
    for field in fields:
        if field in record and record.get(field) is not None:
            return record.get(field)
    return None


def _as_label_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        value = str(label).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _extend_available_labels(row: dict[str, Any], labels: list[str]) -> None:
    available = list(row.get("available_labels") or [])
    for label in labels:
        if label not in available:
            available.append(label)
    row["available_labels"] = available


def _attach_source_supervision(
    row: dict[str, Any],
    *,
    expected_document_sources: list[str] | None = None,
    expected_structured_sources: list[str] | None = None,
) -> dict[str, Any]:
    doc_sources = _dedupe_labels(expected_document_sources or [])
    structured_sources = _dedupe_labels(expected_structured_sources or [])
    if doc_sources:
        row["expected_document_sources"] = doc_sources
    if structured_sources:
        row["expected_structured_sources"] = structured_sources
    if doc_sources or structured_sources:
        _extend_available_labels(row, ["expected_document_sources", "expected_structured_sources"])
    return row


def _build_explicit_classification_row(
    text: str,
    *,
    source_id: str,
    source_row_id: Any | None = None,
    product_type: str = "unknown",
    raw_label: Any | None = None,
    question_style: str | None = None,
    base_query: str | None = None,
    base_question_style: str | None = None,
    available_labels: list[str] | None = None,
) -> dict[str, Any]:
    row = build_autolabeled_classification_row(
        text,
        source_id=source_id,
        source_row_id=source_row_id,
        product_type=product_type,
        raw_label=raw_label,
        available_labels=available_labels or ["product_type", "intent_labels", "topic_labels", "question_style"],
    )
    row["intent_labels"] = _dedupe_labels(autolabel_intents(text))
    row["topic_labels"] = _dedupe_labels(autolabel_topics(text))
    row["question_style"] = question_style or autolabel_question_style(text)
    row["product_type"] = product_type
    if base_query:
        row["base_query"] = base_query
    if base_question_style:
        row["base_question_style"] = base_question_style
    return row


def _extract_query_head(text: str) -> str:
    content = str(text or "").strip()
    if not content:
        return ""

    normalized = content.replace("\r\n", "\n")
    split_markers = ("原标题：", "来源：", "Source:", "\n")
    for marker in split_markers:
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0].strip()
    for punctuation in ("？", "?", "。", "!"):
        if punctuation in normalized:
            head, _, _ = normalized.partition(punctuation)
            if len(head.strip()) >= 4:
                return f"{head.strip()}{punctuation}"
    return normalized[:256].strip()


def _conversation_prompt(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, Mapping):
                continue
            speaker = str(item.get("from") or "").lower()
            if speaker in {"human", "user"}:
                value = str(item.get("value") or "").strip()
                if value:
                    return value
    return ""


def _product_type_from_text(text: str) -> str:
    if not text:
        return "unknown"
    lowered = text.lower()
    if any(token in text for token in ("ETF", "etf", "LOF", "lof")):
        return "etf"
    if any(token in text for token in ("基金", "公募", "私募", "基金经理", "申赎", "定投")) or "fund" in lowered:
        return "fund"
    if any(token in text for token in ("指数", "中证", "沪深300", "上证50", "创业板指")) or "index" in lowered:
        return "index"
    if any(token in text for token in ("股票", "股价", "证券", "港股", "A股", "a股", "美股", "个股")) or "stock" in lowered:
        return "stock"
    if any(token in text for token in ("宏观", "央行", "利率", "GDP", "CPI", "PMI", "财政", "货币", "降准", "降息", "经济")) or any(
        token in lowered for token in ("macro", "economy", "inflation", "interest rate", "gdp", "cpi", "pmi")
    ):
        return "macro"
    if any(token in text for token in ("财经", "金融", "市场", "券商", "行业", "板块", "公司", "业绩", "财报", "研报", "评级", "大盘")) or any(
        token in lowered for token in ("market", "finance", "sector", "industry", "earnings")
    ):
        return "generic_market"
    return "unknown"


def _product_type_from_record(record: Mapping[str, Any]) -> str:
    label_id = _as_label_id(_first_present(record, ("label", "label_id", "label_desc")))
    if label_id is not None:
        return map_tnews_product_type(label_id)
    text = " ".join(
        part
        for part in (
            _first_text(record, ("title", "text", "content", "sentence", "instruction", "question", "input")),
            str(_first_present(record, ("label_desc", "category", "label")) or ""),
        )
        if part
    )
    inferred = _product_type_from_text(text)
    if inferred != "unknown":
        return inferred
    label_text = str(_first_present(record, ("label_desc", "category", "label")) or "").strip().lower()
    if not label_text:
        return "unknown"
    if any(token in label_text for token in ("股票", "stock")):
        return "stock"
    if any(token in label_text for token in ("财经", "finance", "market")):
        return "generic_market"
    if any(token in label_text for token in ("房", "地产", "macro")):
        return "macro"
    return "unknown"


def _source_labels_for_product_type(product_type: str) -> tuple[list[str], list[str]]:
    if product_type == "stock":
        return ["news", "announcement", "research_note"], ["market_api", "fundamental_sql"]
    if product_type == "etf":
        return ["research_note", "faq", "product_doc"], ["market_api"]
    if product_type == "fund":
        return ["research_note", "faq", "product_doc"], ["fundamental_sql"]
    if product_type == "index":
        return ["news", "research_note"], ["market_api", "macro_sql"]
    if product_type == "generic_market":
        return ["news", "research_note"], ["macro_sql", "fundamental_sql"]
    if product_type == "macro":
        return ["news", "announcement", "research_note"], ["macro_sql"]
    return [], []


def _source_labels_from_text(text: str, product_type: str, raw_label: Any | None = None) -> tuple[list[str], list[str]]:
    combined = " ".join(part for part in (text, str(raw_label or "")) if part).strip()
    doc_sources, structured_sources = _source_labels_for_product_type(product_type)

    if any(token in combined for token in ("政策", "法规", "法条", "条例", "合规", "监管", "司法", "行政", "法律")):
        doc_sources.extend(["faq", "product_doc", "announcement"])
        structured_sources.append("macro_sql")
    if any(token in combined for token in ("问答", "知识", "百科", "常识", "Q&A", "QA", "答疑")):
        doc_sources.extend(["faq", "product_doc"])
    if any(token in combined for token in ("新闻", "消息", "公告", "资讯", "快讯", "动态")):
        doc_sources.extend(["news", "announcement"])
    if any(token in combined for token in ("news", "headline", "update", "filing", "disclosure")):
        doc_sources.extend(["news", "announcement"])
    if any(token in combined for token in ("研报", "研究报告", "评级", "目标价", "调研")):
        doc_sources.extend(["research_note", "news"])
        structured_sources.append("fundamental_sql")
    if any(token in combined.lower() for token in ("research report", "broker", "rating", "target price")):
        doc_sources.extend(["research_note", "news"])
        structured_sources.append("fundamental_sql")
    if any(token in combined for token in ("财报", "业绩", "营收", "利润", "净利润", "毛利", "ROE", "估值", "市盈率", "市净率")):
        structured_sources.append("fundamental_sql")
    if any(token in combined.lower() for token in ("earnings", "revenue", "profit", "valuation", "pe", "pb")):
        structured_sources.append("fundamental_sql")
    if any(token in combined for token in ("行业", "板块", "产业链", "上下游", "龙头", "细分")):
        structured_sources.append("industry_sql")
        doc_sources.extend(["news", "research_note"])
    if any(token in combined.lower() for token in ("industry", "sector", "supply chain")):
        structured_sources.append("industry_sql")
        doc_sources.extend(["news", "research_note"])
    if any(token in combined for token in ("宏观", "央行", "利率", "GDP", "CPI", "PMI", "财政", "货币", "降准", "降息")):
        structured_sources.append("macro_sql")
        doc_sources.extend(["news", "research_note"])
    if any(token in combined.lower() for token in ("macro", "inflation", "interest rate", "gdp", "cpi", "policy")):
        structured_sources.append("macro_sql")
        doc_sources.extend(["news", "research_note"])
    if any(token in combined for token in ("ETF", "LOF", "基金", "费率", "申赎", "定投")) or any(
        token in combined.lower() for token in ("etf", "lof", "fund", "redemption", "subscription")
    ):
        doc_sources.extend(["faq", "product_doc", "research_note"])
    if any(token in combined for token in ("指数", "大盘", "沪深300")) or any(token in combined.lower() for token in ("index", "benchmark")):
        structured_sources.extend(["market_api", "macro_sql"])

    return _dedupe_labels(doc_sources), _dedupe_labels(structured_sources)


def adapt_tnews_rows(raw: Any, source_id: str = "tnews") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        text = _first_text(record, ("sentence", "text", "content", "title"))
        if not text:
            continue
        label_id = _as_label_id(_first_present(record, ("label", "label_id")))
        product_type = map_tnews_product_type(label_id) if label_id is not None else _product_type_from_record(record)
        row = build_autolabeled_classification_row(
            text,
            source_id=source_id,
            source_row_id=record.get("id") or record.get("uid") or record.get("sample_id"),
            product_type=product_type,
            raw_label=label_id if label_id is not None else _first_present(record, ("label", "label_desc")),
            available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
        )
        doc_sources, structured_sources = _source_labels_from_text(text, product_type, row["raw_label"])
        rows.append(
            _attach_source_supervision(
                row,
                expected_document_sources=doc_sources,
                expected_structured_sources=structured_sources,
            )
        )
    return rows


def adapt_thucnews_rows(raw: Any, source_id: str = "thucnews") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Path):
        if raw.is_file() and raw.suffix.lower() == ".7z":
            return rows
        if raw.is_dir():
            for path in sorted(raw.rglob("*.txt")):
                category = next((part for part in path.parts if part in {"财经", "股票"}), "")
                if not category:
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError:
                    continue
                if not content:
                    continue
                first_line, _, rest = content.partition("\n")
                text = f"{first_line.strip()}\n{rest.strip()}".strip()
                product_type = "stock" if category == "股票" else "generic_market"
                row = build_autolabeled_classification_row(
                    text,
                    source_id=source_id,
                    source_row_id=str(path),
                    product_type=product_type,
                    raw_label=category,
                    available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
                )
                doc_sources, structured_sources = _source_labels_from_text(text, product_type, category)
                rows.append(
                    _attach_source_supervision(
                        row,
                        expected_document_sources=doc_sources,
                        expected_structured_sources=structured_sources,
                    )
                )
            return rows
    for record in _load_records(raw):
        text = _first_text(record, ("title", "text", "content", "sentence"))
        if not text:
            continue
        product_type = _product_type_from_record(record)
        row = build_autolabeled_classification_row(
            text,
            source_id=source_id,
            source_row_id=_first_present(record, ("id", "uid", "sample_id")),
            product_type=product_type,
            raw_label=_first_present(record, ("label", "category", "label_desc")),
            available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
        )
        doc_sources, structured_sources = _source_labels_from_text(text, product_type, row["raw_label"])
        rows.append(
            _attach_source_supervision(
                row,
                expected_document_sources=doc_sources,
                expected_structured_sources=structured_sources,
            )
        )
    return rows


def adapt_finnl_rows(raw: Any, source_id: str = "finnl") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Path):
        if raw.is_dir():
            for path in sorted(raw.rglob("*finnl*.json")):
                rows.extend(adapt_finnl_rows(path, source_id=source_id))
            return rows
        payload = _load_json_payload(raw) if raw.suffix.lower() == ".json" else raw
    else:
        payload = raw

    if isinstance(payload, list) and payload and isinstance(payload[0], Sequence) and not isinstance(payload[0], (str, bytes, bytearray)):
        for index, record in enumerate(payload):
            if not isinstance(record, Sequence) or len(record) < 1:
                continue
            text = str(record[0]).strip()
            raw_label = str(record[1]).strip() if len(record) > 1 else ""
            if not text:
                continue
            row = build_autolabeled_classification_row(
                text,
                source_id=source_id,
                source_row_id=index,
                product_type=_product_type_from_text(text),
                raw_label=raw_label,
                available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
            )
            doc_sources, structured_sources = _source_labels_from_text(text, row["product_type"], raw_label)
            rows.append(
                _attach_source_supervision(
                    row,
                    expected_document_sources=doc_sources,
                    expected_structured_sources=structured_sources,
                )
            )
        return rows

    for record in _load_records(payload):
        text = _first_text(record, ("title", "text", "content", "sentence"))
        if not text:
            continue
        product_type = _product_type_from_record(record)
        raw_label = _first_present(record, ("label", "category", "label_desc"))
        row = build_autolabeled_classification_row(
            text,
            source_id=source_id,
            source_row_id=_first_present(record, ("id", "uid", "sample_id")),
            product_type=product_type,
            raw_label=raw_label,
            available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
        )
        doc_sources, structured_sources = _source_labels_from_text(text, product_type, raw_label)
        rows.append(
            _attach_source_supervision(
                row,
                expected_document_sources=doc_sources,
                expected_structured_sources=structured_sources,
            )
        )
    return rows


def adapt_smp2017_rows(raw: Any, source_id: str = "smp2017") -> list[dict[str, Any]]:
    if isinstance(raw, Path) and raw.is_dir():
        rows: list[dict[str, Any]] = []
        for path in sorted(raw.rglob("*.txt")):
            domain = path.stem.split("_")[-1]
            product_type = "stock" if domain == "stock" else ""
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    text = line.strip()
                    if not text:
                        continue
                    rows.append(
                        build_autolabeled_classification_row(
                            text,
                            source_id=source_id,
                            source_row_id=f"{path.name}:{index}",
                            product_type=product_type,
                            raw_label=domain,
                            available_labels=["intent_labels", "topic_labels", "question_style", "product_type"],
                        )
                    )
        return rows
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        text = _first_text(record, ("query", "text", "content", "title", "sentence"))
        if not text:
            continue
        row = build_autolabeled_classification_row(
            text,
            source_id=source_id,
            source_row_id=_first_present(record, ("id", "uid", "sample_id")),
            product_type="",
            raw_label=_first_present(record, ("label", "category", "label_desc")),
            available_labels=["intent_labels", "topic_labels", "question_style"],
        )
        rows.append(row)
    return rows


def adapt_cflue_rows(raw: Any, source_id: str = "cflue") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        text = " ".join(
            part
            for part in (
                _first_text(record, ("question", "instruction", "text", "input", "title")),
                _first_text(record, ("output", "answer")),
                str(_first_present(record, ("task", "sub_task")) or ""),
            )
            if part
        ).strip()
        if not text:
            continue
        product_type = _product_type_from_text(text)
        raw_label = _first_present(record, ("output", "answer", "sub_task", "task"))
        doc_sources, structured_sources = _source_labels_from_text(text, product_type, raw_label)
        rows.append(
            _attach_source_supervision(
                build_autolabeled_classification_row(
                    text,
                    source_id=source_id,
                    source_row_id=_first_present(record, ("id", "name", "task", "sub_task")),
                    product_type=product_type,
                    raw_label=raw_label,
                    available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
                ),
                expected_document_sources=doc_sources,
                expected_structured_sources=structured_sources,
            )
        )
    return rows


def adapt_mxode_finance_rows(raw: Any, source_id: str = "mxode_finance") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        query = _extract_query_head(_first_text(record, ("prompt", "instruction", "question", "text")))
        if not query:
            continue
        product_type = _product_type_from_text(query)
        row = _build_explicit_classification_row(
            query,
            source_id=source_id,
            source_row_id=_first_present(record, ("id",)),
            product_type=product_type,
            raw_label=_first_present(record, ("subset", "source")),
            available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
        )
        doc_sources, structured_sources = _source_labels_from_text(query, product_type, row["raw_label"])
        rows.append(
            _attach_source_supervision(
                row,
                expected_document_sources=doc_sources,
                expected_structured_sources=structured_sources,
            )
        )
    return rows


def adapt_baai_finance_instruction_rows(raw: Any, source_id: str = "baai_finance_instruction") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        query = _extract_query_head(_conversation_prompt(record) or _first_text(record, ("instruction", "text", "input")))
        if not query:
            continue
        product_type = _product_type_from_text(query)
        row = _build_explicit_classification_row(
            query,
            source_id=source_id,
            source_row_id=_first_present(record, ("id",)),
            product_type=product_type,
            raw_label=_first_present(record, ("source", "lang")),
            available_labels=["product_type", "intent_labels", "topic_labels", "question_style"],
        )
        doc_sources, structured_sources = _source_labels_from_text(query, product_type, row["raw_label"])
        rows.append(
            _attach_source_supervision(
                row,
                expected_document_sources=doc_sources,
                expected_structured_sources=structured_sources,
            )
        )
    return rows


def adapt_qrecc_rows(raw: Any, source_id: str = "qrecc") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        base_query = _first_text(record, ("Question", "question"))
        rewritten_query = _first_text(record, ("Truth_rewrite", "rewrite"))
        query = _extract_query_head(rewritten_query or base_query)
        if not query:
            continue
        product_type = _product_type_from_text(query)
        row = _build_explicit_classification_row(
            query,
            source_id=source_id,
            source_row_id=_first_present(record, ("Conversation_no", "Turn_no")),
            product_type=product_type,
            raw_label=_first_present(record, ("Conversation_source",)),
            question_style=autolabel_question_style(query),
            base_query=base_query,
            base_question_style=autolabel_question_style(base_query) if base_query else None,
            available_labels=["question_style"],
        )
        rows.append(row)
    return rows


def adapt_risawoz_rows(raw: Any, source_id: str = "risawoz") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        dialogue = record.get("dialogue")
        if not isinstance(dialogue, list):
            continue
        dialogue_id = _first_present(record, ("dialogue_id", "id"))
        for turn in dialogue:
            if not isinstance(turn, Mapping):
                continue
            query = _extract_query_head(_first_text(turn, ("user_utterance",)))
            if not query:
                continue
            row = _build_explicit_classification_row(
                query,
                source_id=source_id,
                source_row_id=f"{dialogue_id}:{_first_present(turn, ('turn_id',))}",
                product_type="unknown",
                raw_label=_first_present(turn, ("turn_domain",)),
                question_style=autolabel_question_style(query),
                available_labels=["question_style"],
            )
            rows.append(row)
    return rows


def adapt_naturalconv_rows(raw: Any, source_id: str = "naturalconv") -> list[dict[str, Any]]:
    if isinstance(raw, Path) and raw.is_dir():
        path = raw / "dialog_release.json"
        if not path.exists():
            return []
        payload = _load_json_payload(path)
    else:
        payload = raw

    rows: list[dict[str, Any]] = []
    for record in _load_records(payload):
        content = record.get("content")
        if not isinstance(content, list):
            continue
        dialog_id = _first_present(record, ("dialog_id", "id"))
        for index, utterance in enumerate(content):
            text = str(utterance).strip()
            if not text:
                continue
            rows.append(
                {
                    "sample_family": "classification",
                    "query": text,
                    "text": text,
                    "product_type": "unknown",
                    "intent_labels": [],
                    "topic_labels": [],
                    "question_style": "",
                    "sentiment_label": None,
                    "split_lock_key": f"{source_id}:{dialog_id}:{index}",
                    "source_id": source_id,
                    "source_row_id": f"{dialog_id}:{index}",
                    "raw_label": "chat",
                    "available_labels": ["out_of_scope_only"],
                }
            )
    return rows


def adapt_dailydialog_rows(raw: Any, source_id: str = "dailydialog") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Path) and raw.is_dir():
        for path in sorted(raw.rglob("dialogues_*.txt")):
            if any(marker in path.name for marker in ("dialogues_act_", "dialogues_emotion_")):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_index, line in enumerate(lines):
                utterances = [item.strip() for item in line.split("__eou__") if item.strip()]
                for turn_index, utterance in enumerate(utterances):
                    rows.append(
                        {
                            "sample_family": "classification",
                            "query": utterance,
                            "text": utterance,
                            "product_type": "unknown",
                            "intent_labels": [],
                            "topic_labels": [],
                            "question_style": "",
                            "sentiment_label": None,
                            "split_lock_key": f"{source_id}:{path.stem}:{line_index}:{turn_index}",
                            "source_id": source_id,
                            "source_row_id": f"{path.stem}:{line_index}:{turn_index}",
                            "raw_label": "chat",
                            "available_labels": ["out_of_scope_only"],
                        }
                    )
        return rows
    return rows
