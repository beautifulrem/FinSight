from __future__ import annotations

import csv
import json
import hashlib
import heapq
import re
from pathlib import Path
from typing import Any

from .training_manifest import load_training_manifest

DEFAULT_AVAILABLE_LABELS = (
    "product_type",
    "intent_labels",
    "topic_labels",
    "expected_document_sources",
    "expected_structured_sources",
    "question_style",
    "sentiment_label",
)
MAX_TRAINING_QUERY_CHARS = 256
FINANCE_PRODUCT_TYPES = {"stock", "etf", "fund", "bond", "crypto", "index", "future", "option", "commodity"}
PRIORITY_TRAINING_SOURCE_IDS = {"curated_boundary_cases"}
SOURCE_PLAN_PRIORITY = (
    "market_api",
    "news",
    "industry_sql",
    "fundamental_sql",
    "announcement",
    "research_note",
    "faq",
    "product_doc",
    "macro_sql",
)
MAX_SOURCE_PLAN_SUPERVISION_QUERIES = 25000
MAX_CLARIFICATION_SUPERVISION_ROWS = 200000
MAX_OUT_OF_SCOPE_SUPERVISION_ROWS = 200000
MAX_OUT_OF_SCOPE_SYNTHETIC_NOISE_ROWS = 60000
MAX_OUT_OF_SCOPE_HARD_NEGATIVE_ROWS = 40000
MAX_DIALOGUE_OOD_BOOST_ROWS = 50000
CURATED_OOD_NEGATIVE_QUERIES = [
    "你好",
    "你好？你是谁？",
    "你是谁",
    "你能做什么",
    "hello",
    "hello, who are you?",
    "who are you",
    "what can you do",
    "translate this sentence into chinese",
    "帮我翻译这段英文",
    "今天天气怎么样",
    "上海今天下雨吗",
    "write a python bubble sort",
    "帮我写个 Python 冒泡排序",
    "讲个笑话",
    "who won the nba game tonight",
    "法国大革命是什么",
    "胸痛急救怎么做",
    "巴黎三日游怎么安排",
    "意面怎么做",
    "什么是线性代数",
    "你好，我想买一辆汽油驱动的SUV，给个意见吧！",
    "你好，我想购买一台台式机，预算在5000元以内，可以帮忙做下推荐吗？",
    "你好，我来苏州办事，顺便想逛逛景点，有没有消费便宜的水乡古镇推荐一下。",
    "医院电话多少，我了解一下相关病情，早做准备。",
    "他们的人均消费是多少呀？",
    "电话呢，我提前订个座位。",
    "预算小于15万。",
    "啊数据库的建立撒娇的你离开",
    "Did it sell well?",
    "What songs were off of Fastball's album All the Pain Money Can Buy?",
    "医院电话多少",
    "你好，我想买一辆上汽大众的MPV，1个人坐，能推荐一辆吗？",
    "你好，我想购买一台台式机，预算在5000元以内，可以帮忙做下推荐吗？",
    "I would like to deposit a check at the bank counter.",
    "What is the difference between the board of directors and the CEO?",
    "The insurance premium can be paid through the current account.",
    "Please send me the invoice and bill of lading.",
    "Could you say something about your educational background?",
    "I want to buy three tickets. What is the entrance fee?",
    "Can I have the roll of film developed here?",
    "Your clock never works. Perhaps you should buy a new one.",
    "I want to pick up my parcels.",
    "I'm the general manager of the company.",
    "Thank you very much. See you then.",
    "Good news! Congratulations!",
]
CURATED_OOD_NEGATIVE_REPEAT = 20
CURATED_FINANCE_POSITIVE_QUERIES = [
    "这个能买吗？",
    "这只该不该止损？",
    "后面还值得拿吗？",
    "现在适合买入吗？",
    "该止损还是继续拿？",
    "Should I cut the loss here?",
    "Would you still hold it?",
    "Is this still worth buying?",
    "ETF和LOF有什么区别？",
    "公募基金费率包含哪些部分？",
    "白酒板块最近为什么跌？",
    "中国平安最近为什么跌？",
    "证券ETF适合长期拿吗？",
    "创业板ETF波动大吗？",
    "上证指数今天为什么跌？",
    "宽基指数最近能买吗？",
    "白酒行业今天为什么跌？",
    "中国平安还值得持有吗？",
    "公募基金申赎规则是什么？",
    "What is the difference between ETF and LOF?",
    "Why did Moutai fall today?",
    "Is Ping An still worth holding?",
    "你觉得中国平安怎么样",
    "你怎么看贵州茅台",
    "中华汽车怎么样",
    "你觉得中华汽车怎么样",
    "Does macro policy affect liquor?",
    "Any recent announcements from Kweichow Moutai?",
]
CURATED_FINANCE_POSITIVE_REPEAT = 20


def _split_labels(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.replace(",", "|").split("|") if item.strip()]


def _is_manifest_path(path: Path) -> bool:
    return path.name == "manifest.json"


def _resolve_dataset_path(path: str | Path, manifest_key: str) -> Path:
    dataset_path = Path(path)
    if _is_manifest_path(dataset_path):
        manifest = load_training_manifest(dataset_path)
        return getattr(manifest, manifest_key)
    return dataset_path


def _resolve_optional_dataset_path(path: str | Path, manifest_key: str) -> Path | None:
    dataset_path = Path(path)
    if _is_manifest_path(dataset_path):
        manifest = load_training_manifest(dataset_path)
        optional_path = getattr(manifest, manifest_key, None)
        if optional_path is None:
            return None
        return optional_path if optional_path.exists() else None
    return dataset_path if dataset_path.exists() else None


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_classification_row(item: dict) -> dict:
    row = dict(item)
    row["query"] = str(item.get("query") or "").strip()[:MAX_TRAINING_QUERY_CHARS]
    row["product_type"] = str(item.get("product_type") or "").strip()
    row["intent_labels"] = _split_labels("|".join(str(value) for value in item.get("intent_labels", []))) if isinstance(item.get("intent_labels"), list) else _split_labels(item.get("intent_labels"))
    row["topic_labels"] = _split_labels("|".join(str(value) for value in item.get("topic_labels", []))) if isinstance(item.get("topic_labels"), list) else _split_labels(item.get("topic_labels"))
    row["expected_document_sources"] = _split_labels("|".join(str(value) for value in item.get("expected_document_sources", []))) if isinstance(item.get("expected_document_sources"), list) else _split_labels(item.get("expected_document_sources"))
    row["expected_structured_sources"] = _split_labels("|".join(str(value) for value in item.get("expected_structured_sources", []))) if isinstance(item.get("expected_structured_sources"), list) else _split_labels(item.get("expected_structured_sources"))
    row["primary_symbol"] = str(item.get("primary_symbol") or "").strip()
    row["comparison_symbol"] = str(item.get("comparison_symbol") or "").strip()
    row["question_style"] = str(item.get("question_style") or "").strip()
    row["sentiment_label"] = str(item.get("sentiment_label") or "").strip()
    available_labels = item.get("available_labels")
    if isinstance(available_labels, list):
        row["available_labels"] = [str(label).strip() for label in available_labels if str(label).strip()]
    elif isinstance(available_labels, str):
        row["available_labels"] = _split_labels(available_labels)
    else:
        row["available_labels"] = list(DEFAULT_AVAILABLE_LABELS)
    return row


def _normalize_jsonl_row(item: dict) -> dict:
    row = dict(item)
    if "query" in row:
        row["query"] = str(row.get("query") or "").strip()
    return row


def _load_optional_jsonl_rows(path: str | Path, manifest_key: str) -> list[dict]:
    dataset_path = _resolve_optional_dataset_path(path, manifest_key)
    if dataset_path is None:
        return []
    if dataset_path.suffix.lower() != ".jsonl":
        return []
    return [_normalize_jsonl_row(item) for item in _load_jsonl_rows(dataset_path)]


def load_training_rows(path: str | Path) -> list[dict]:
    dataset_path = _resolve_dataset_path(path, "classification_path")
    if dataset_path.suffix.lower() == ".jsonl":
        return [_normalize_classification_row(item) for item in _load_jsonl_rows(dataset_path)]

    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = dict(row)
            parsed["query"] = row["query"][:MAX_TRAINING_QUERY_CHARS]
            parsed["product_type"] = row.get("product_type", "unknown")
            parsed["intent_labels"] = _split_labels(row.get("intent_labels"))
            parsed["topic_labels"] = _split_labels(row.get("topic_labels"))
            parsed["expected_document_sources"] = _split_labels(row.get("expected_document_sources"))
            parsed["expected_structured_sources"] = _split_labels(row.get("expected_structured_sources"))
            parsed["primary_symbol"] = (row.get("primary_symbol") or "").strip()
            parsed["comparison_symbol"] = (row.get("comparison_symbol") or "").strip()
            parsed["question_style"] = row.get("question_style", "").strip()
            parsed["sentiment_label"] = row.get("sentiment_label", "").strip()
            parsed["available_labels"] = list(DEFAULT_AVAILABLE_LABELS)
            rows.append(parsed)
        return rows


def iter_training_rows(path: str | Path):
    dataset_path = _resolve_dataset_path(path, "classification_path")
    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield _normalize_classification_row(json.loads(line))
        return

    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["query"] = row["query"]
            parsed["product_type"] = row.get("product_type", "unknown")
            parsed["intent_labels"] = _split_labels(row.get("intent_labels"))
            parsed["topic_labels"] = _split_labels(row.get("topic_labels"))
            parsed["expected_document_sources"] = _split_labels(row.get("expected_document_sources"))
            parsed["expected_structured_sources"] = _split_labels(row.get("expected_structured_sources"))
            parsed["primary_symbol"] = (row.get("primary_symbol") or "").strip()
            parsed["comparison_symbol"] = (row.get("comparison_symbol") or "").strip()
            parsed["question_style"] = row.get("question_style", "").strip()
            parsed["sentiment_label"] = row.get("sentiment_label", "").strip()
            parsed["available_labels"] = list(DEFAULT_AVAILABLE_LABELS)
            yield parsed


def load_qrel_rows(path: str | Path) -> list[dict]:
    dataset_path = _resolve_dataset_path(path, "qrels_path")
    if dataset_path.suffix.lower() == ".jsonl":
        return [_normalize_jsonl_row(item) for item in _load_jsonl_rows(dataset_path)]

    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_entity_annotation_rows(path: str | Path) -> list[dict]:
    dataset_path = _resolve_dataset_path(path, "entity_annotations_path")
    if dataset_path.suffix.lower() == ".jsonl":
        return [_normalize_jsonl_row(item) for item in _load_jsonl_rows(dataset_path)]

    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def row_supports_label(row: dict, label_field: str) -> bool:
    available_labels = row.get("available_labels")
    if isinstance(available_labels, list) and available_labels:
        return label_field in available_labels

    value = row.get(label_field)
    if isinstance(value, list):
        return bool(value)
    if value is None:
        return False
    return str(value).strip() != ""


def filter_rows_for_label(rows: list[dict], label_field: str) -> list[dict]:
    return [row for row in rows if row_supports_label(row, label_field)]


def sample_training_rows_for_label(path: str | Path, label_field: str, limit: int) -> list[dict]:
    heap: list[tuple[int, int, dict]] = []
    priority_rows: list[dict] = []
    serial = 0
    for row in iter_training_rows(path):
        if not row_supports_label(row, label_field):
            continue
        if _is_priority_training_row(row):
            priority_rows.append(row)
            continue
        key = _stable_row_hash(row)
        serial += 1
        remaining_limit = max(limit - len(priority_rows), 0)
        if len(heap) < remaining_limit:
            heapq.heappush(heap, (-key, serial, row))
            continue
        if remaining_limit <= 0:
            continue
        if key < -heap[0][0]:
            heapq.heapreplace(heap, (-key, serial, row))
    if len(priority_rows) >= limit:
        rows = _sample_records(priority_rows, limit)
        rows.sort(key=_stable_row_hash)
        return rows
    rows = [*priority_rows, *[item[2] for item in heap]]
    rows.sort(key=_stable_row_hash)
    return rows


def sample_training_rows_per_value(path: str | Path, label_field: str, per_value_cap: int) -> list[dict]:
    buckets: dict[str, list[tuple[int, int, dict]]] = {}
    priority_rows: list[dict] = []
    serial = 0
    for row in iter_training_rows(path):
        if not row_supports_label(row, label_field):
            continue
        if _is_priority_training_row(row):
            priority_rows.append(row)
            continue
        value = str(row.get(label_field) or "").strip()
        if not value:
            continue
        bucket = buckets.setdefault(value, [])
        key = _stable_row_hash(row)
        serial += 1
        if len(bucket) < per_value_cap:
            heapq.heappush(bucket, (-key, serial, row))
            continue
        if key < -bucket[0][0]:
            heapq.heapreplace(bucket, (-key, serial, row))
    rows: list[dict] = []
    for bucket in buckets.values():
        rows.extend(item[2] for item in bucket)
    rows.extend(priority_rows)
    rows.sort(key=_stable_row_hash)
    return rows


def _is_priority_training_row(row: dict) -> bool:
    return str(row.get("source_id") or "") in PRIORITY_TRAINING_SOURCE_IDS


def load_source_plan_supervision_rows(path: str | Path) -> list[dict]:
    return _load_optional_jsonl_rows(path, "source_plan_supervision_path")


def load_clarification_supervision_rows(path: str | Path) -> list[dict]:
    return _load_optional_jsonl_rows(path, "clarification_supervision_path")


def load_out_of_scope_supervision_rows(path: str | Path) -> list[dict]:
    return _load_optional_jsonl_rows(path, "out_of_scope_supervision_path")


def load_typo_supervision_rows(path: str | Path) -> list[dict]:
    return _load_optional_jsonl_rows(path, "typo_supervision_path")


def build_source_plan_supervision_rows_from_records(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    source_plan_records = _sample_records(
        (
            row
            for row in _iter_classification_rows(records)
            if row_supports_label(row, "expected_document_sources")
            or row_supports_label(row, "expected_structured_sources")
        ),
        MAX_SOURCE_PLAN_SUPERVISION_QUERIES,
    )
    for row in source_plan_records:
        expected_sources = _ordered_unique(_as_str_list(row.get("expected_document_sources")) + _as_str_list(row.get("expected_structured_sources")))
        if not expected_sources:
            continue
        query = str(row.get("query") or "")
        intent_labels = _as_str_list(row.get("intent_labels"))
        topic_labels = _as_str_list(row.get("topic_labels"))
        time_scope = _detect_time_scope(query)
        base_row = _copy_public_row(row)
        base_row.update(
            {
                "query": query,
                "product_type": str(row.get("product_type") or "unknown"),
                "intent_labels": intent_labels,
                "topic_labels": topic_labels,
                "time_scope": time_scope,
                "available_labels": _as_str_list(row.get("available_labels")) or list(DEFAULT_AVAILABLE_LABELS),
                "rule_plan": list(expected_sources),
                "expected_sources": expected_sources,
            }
        )
        for source in SOURCE_PLAN_PRIORITY:
            rows.append({**base_row, "source": source, "label": int(source in expected_sources)})
    return rows


def build_clarification_supervision_rows_from_records(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in _iter_classification_rows(records):
        if not _is_finance_row(row):
            continue
        query = str(row.get("query") or "")
        if not query:
            continue
        intent_labels = _as_str_list(row.get("intent_labels"))
        topic_labels = _as_str_list(row.get("topic_labels"))
        base_row = _copy_public_row(row)
        explicit_needs_clarification = row.get("needs_clarification")
        base_row.update(
            {
                "query": query,
                "product_type": str(row.get("product_type") or "unknown"),
                "intent_labels": intent_labels,
                "topic_labels": topic_labels,
                "time_scope": _detect_time_scope(query),
                "entity_count": _estimate_entity_count(row),
                "comparison_target_count": _estimate_comparison_target_count(query, intent_labels, topic_labels),
                "mask_type": "identity",
                "needs_clarification": _as_bool(explicit_needs_clarification) if explicit_needs_clarification is not None else False,
            }
        )
        rows.append(base_row)

        if _as_bool(explicit_needs_clarification):
            continue
        masked_query = _mask_clarification_query(row)
        if masked_query and masked_query != query:
            masked_row = dict(base_row)
            masked_row.update(
                {
                    "query": masked_query,
                    "base_query": query,
                    "mask_type": "underspecified",
                    "needs_clarification": True,
                }
            )
            rows.append(masked_row)
        if len(rows) >= MAX_CLARIFICATION_SUPERVISION_ROWS:
            break
    return rows


def build_out_of_scope_supervision_rows_from_records(records: list[dict]) -> list[dict]:
    finance_records: list[dict] = []
    general_records: list[dict] = []
    for row in _iter_classification_rows(records):
        query = str(row.get("query") or "")
        if not query:
            continue
        if _is_finance_row(row):
            finance_records.append(row)
        else:
            general_records.append(row)

    finance_limit = MAX_OUT_OF_SCOPE_SUPERVISION_ROWS // 2
    general_limit = MAX_OUT_OF_SCOPE_SUPERVISION_ROWS - finance_limit
    sampled_records = [
        *_sample_records(finance_records, finance_limit),
        *_sample_records_balanced_by_source(general_records, general_limit),
    ]

    rows: list[dict] = []
    for row in sampled_records:
        query = str(row.get("query") or "")
        is_finance = _is_finance_row(row)
        domain = "finance" if is_finance else "general"
        rows.append(
            {
                **_copy_public_row(row),
                "query": query,
                "product_type": str(row.get("product_type") or "unknown"),
                "domain": domain,
                "label": 0 if is_finance else 1,
                "source_family": "classification",
                "is_in_domain": is_finance,
            }
        )
    rows.extend(_boost_hard_ood_negative_rows(rows, MAX_OUT_OF_SCOPE_HARD_NEGATIVE_ROWS))
    rows.extend(_boost_dialogue_ood_negative_rows(rows, MAX_DIALOGUE_OOD_BOOST_ROWS))
    rows.extend(_build_out_of_scope_noise_rows(sampled_records, MAX_OUT_OF_SCOPE_SYNTHETIC_NOISE_ROWS))
    rows.extend(_build_curated_ood_seed_rows())
    rows.extend(_build_curated_finance_seed_rows())
    return rows


def build_typo_supervision_rows_from_records(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    alias_rows = [row for row in records if _normalize_family(row.get("sample_family")) == "alias"]
    for row in alias_rows:
        mention = _first_nonempty(row.get("alias_text"), row.get("mention"), row.get("normalized_alias"))
        alias = _first_nonempty(row.get("normalized_alias"), row.get("canonical_name"), row.get("alias_text"))
        if not mention or not alias:
            continue
        rows.append(
            {
                **_copy_public_row(row),
                "query": mention,
                "mention": mention,
                "alias": alias,
                "canonical_name": str(row.get("canonical_name") or ""),
                "label": 1,
                "heuristic_score": 1.0,
                "pair_type": "alias_match",
            }
        )
    return rows


def _iter_classification_rows(records: list[dict]):
    for row in records:
        if _normalize_family(row.get("sample_family")) == "classification":
            yield row


def _sample_records(rows: Any, limit: int) -> list[dict]:
    heap: list[tuple[int, int, dict]] = []
    serial = 0
    for row in rows:
        key = _stable_row_hash(row)
        serial += 1
        if len(heap) < limit:
            heapq.heappush(heap, (-key, serial, row))
            continue
        if key < -heap[0][0]:
            heapq.heapreplace(heap, (-key, serial, row))
    sampled = [item[2] for item in heap]
    sampled.sort(key=_stable_row_hash)
    return sampled


def _sample_records_balanced_by_source(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return list(rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "unknown")
        grouped.setdefault(source_id, []).append(row)
    if not grouped:
        return _sample_records(rows, limit)

    per_source_limit = max(limit // len(grouped), 1)
    sampled: list[dict] = []
    for source_id in sorted(grouped):
        sampled.extend(_sample_records(grouped[source_id], per_source_limit))
    if len(sampled) >= limit:
        sampled = _sample_records(sampled, limit)
        sampled.sort(key=_stable_row_hash)
        return sampled

    seen_ids = {id(row) for row in sampled}
    remainder = [row for row in rows if id(row) not in seen_ids]
    sampled.extend(_sample_records(remainder, limit - len(sampled)))
    sampled.sort(key=_stable_row_hash)
    return sampled[:limit]


def _copy_public_row(row: dict) -> dict:
    copied = dict(row)
    copied.pop("text", None)
    copied.pop("body", None)
    return copied


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return _split_labels(value)
    return [str(value).strip()] if str(value).strip() else []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _detect_time_scope(query: str) -> str:
    if any(term in query for term in ("今天", "今日", "当日", "现在", "当前")):
        return "today"
    if any(term in query for term in ("最近一周", "近一周", "一周", "过去7天")):
        return "recent_1w"
    if any(term in query for term in ("最近", "近几天", "这几天", "近期")):
        return "recent"
    if any(term in query for term in ("明天", "后天", "未来", "接下来", "后面")):
        return "future"
    return "unspecified"


def _is_finance_row(row: dict) -> bool:
    product_type = str(row.get("product_type") or "").strip().lower()
    if product_type in FINANCE_PRODUCT_TYPES:
        return True
    query = str(row.get("query") or "")
    return any(term in query for term in ("股票", "基金", "ETF", "LOF", "债券", "券商", "港股", "A股", "茅台", "平安", "白酒", "大盘"))


def _estimate_entity_count(row: dict) -> int:
    count = 0
    for field in ("primary_symbol", "comparison_symbol", "symbol", "canonical_name"):
        value = str(row.get(field) or "").strip()
        if value:
            count += 1
    return min(count, 2)


def _estimate_comparison_target_count(query: str, intent_labels: list[str], topic_labels: list[str]) -> int:
    if "peer_compare" in intent_labels or "comparison" in topic_labels:
        return 2
    if any(term in query for term in ("比", "区别", "不同", "哪个", "谁更", "一回事")):
        return 2
    return 0


def _mask_clarification_query(row: dict) -> str | None:
    query = str(row.get("query") or "").strip()
    if not query:
        return None
    primary = _first_nonempty(row.get("primary_symbol"), row.get("canonical_name"))
    comparison = _first_nonempty(row.get("comparison_symbol"))
    masked = query
    if primary and primary in masked:
        replacement = "A标的" if comparison and comparison in masked else "这个标的"
        masked = masked.replace(primary, replacement)
    if comparison and comparison in masked:
        masked = masked.replace(comparison, "B标的")
    if masked != query:
        return masked
    if any(term in query for term in ("这个", "这只", "这家", "它", "该标的")):
        return None
    if any(term in query for term in ("股票", "ETF", "基金", "LOF", "债券")):
        return query.replace("股票", "这个标的").replace("ETF", "这个标的").replace("基金", "这个标的").replace("LOF", "这个标的").replace("债券", "这个标的")
    return None


def _build_out_of_scope_noise_rows(records: list[dict], limit: int) -> list[dict]:
    general_queries = [
        str(row.get("query") or "").strip()
        for row in records
        if not _is_finance_row(row) and str(row.get("query") or "").strip()
    ]
    if len(general_queries) < 2 or limit <= 0:
        return []

    cleaned_queries = [_normalize_noise_query(query) for query in general_queries]
    cleaned_queries = [query for query in cleaned_queries if len(query) >= 4]
    rows: list[dict] = []
    fillers = ["啊", "额", "嗯", "唉"]
    for index in range(len(cleaned_queries) - 1):
        left = cleaned_queries[index]
        right = cleaned_queries[index + 1]
        mixed = _mix_noise_fragments(left, right)
        if mixed:
            rows.append(
                {
                    "query": mixed,
                    "product_type": "unknown",
                    "domain": "general",
                    "label": 1,
                    "source_family": "synthetic_noise",
                    "is_in_domain": False,
                    "source_id": "synthetic_ood_noise",
                }
            )
        if len(rows) >= limit:
            break
        filler = fillers[index % len(fillers)]
        prefixed = f"{filler}{mixed}" if mixed else ""
        if prefixed and prefixed != mixed:
            rows.append(
                {
                    "query": prefixed,
                    "product_type": "unknown",
                    "domain": "general",
                    "label": 1,
                    "source_family": "synthetic_noise",
                    "is_in_domain": False,
                    "source_id": "synthetic_ood_noise",
                }
            )
        if len(rows) >= limit:
            break
    return rows[:limit]


def _build_curated_ood_seed_rows() -> list[dict]:
    rows: list[dict] = []
    for index, query in enumerate(CURATED_OOD_NEGATIVE_QUERIES):
        for repeat in range(CURATED_OOD_NEGATIVE_REPEAT):
            rows.append(
                {
                    "query": query,
                    "product_type": "unknown",
                    "domain": "general",
                    "label": 1,
                    "source_family": "curated_seed_negative",
                    "is_in_domain": False,
                    "source_id": "curated_ood_seed",
                    "source_row_id": f"curated:{index}:{repeat}",
                }
            )
    return rows


def _build_curated_finance_seed_rows() -> list[dict]:
    rows: list[dict] = []
    for index, query in enumerate(CURATED_FINANCE_POSITIVE_QUERIES):
        for repeat in range(CURATED_FINANCE_POSITIVE_REPEAT):
            rows.append(
                {
                    "query": query,
                    "product_type": "unknown",
                    "domain": "finance",
                    "label": 0,
                    "source_family": "curated_seed_positive",
                    "is_in_domain": True,
                    "source_id": "curated_finance_seed",
                    "source_row_id": f"finance:{index}:{repeat}",
                }
            )
    return rows


def _boost_hard_ood_negative_rows(rows: list[dict], limit: int) -> list[dict]:
    hard_negatives = [
        row
        for row in rows
        if int(row.get("label") or 0) == 1 and _is_hard_ood_negative_query(str(row.get("query") or ""))
    ]
    sampled = _sample_records(hard_negatives, limit)
    boosted: list[dict] = []
    for row in sampled:
        cloned = dict(row)
        cloned["source_family"] = "hard_negative_boost"
        cloned["source_id"] = f"{row.get('source_id') or 'general'}_hard_negative"
        boosted.append(cloned)
    return boosted


def _boost_dialogue_ood_negative_rows(rows: list[dict], limit: int) -> list[dict]:
    candidates = []
    for row in rows:
        if int(row.get("label") or 0) != 1:
            continue
        if str(row.get("source_id") or "") not in {"dailydialog", "naturalconv", "risawoz", "qrecc"}:
            continue
        query = str(row.get("query") or "")
        subcategory = _dialogue_ood_subcategory(query)
        if not subcategory:
            continue
        candidates.append((subcategory, row))
    if not candidates:
        return []

    grouped: dict[str, list[dict]] = {}
    for subcategory, row in candidates:
        grouped.setdefault(subcategory, []).append(row)

    per_group = max(limit // len(grouped), 1)
    boosted: list[dict] = []
    for subcategory in sorted(grouped):
        sampled = _sample_records(grouped[subcategory], per_group)
        for row in sampled:
            cloned = dict(row)
            cloned["source_family"] = "dialogue_ood_boost"
            cloned["source_id"] = f"dialogue_ood_{subcategory}"
            boosted.append(cloned)
    return boosted[:limit]


def _normalize_noise_query(query: str) -> str:
    query = str(query or "").strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"[^\w\u4e00-\u9fff]+", " ", query)
    return query.strip()


def _is_hard_ood_negative_query(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return False
    finance_like_terms = [
        "价格",
        "多少",
        "预算",
        "推荐",
        "买",
        "卖",
        "持有",
        "消费",
        "电话",
        "配置",
        "高配",
        "低配",
        "一星级",
        "二星级",
        "price",
        "sell",
        "buy",
        "worth",
        "budget",
        "recommend",
        "hold",
        "best-selling",
        "charge",
    ]
    if any(term in lowered for term in finance_like_terms):
        return True
    short_followup_patterns = [
        "可以吗",
        "有吗",
        "多少呢",
        "电话呢",
        "好啊",
        "行啊",
        "再见",
        "拜拜",
        "真的吗",
    ]
    if len(lowered) <= 16 and any(term in lowered for term in short_followup_patterns):
        return True
    if len(lowered.split()) <= 5 and any(term in lowered for term in ["did it", "was it", "what happened", "how much"]):
        return True
    return False


def _dialogue_ood_subcategory(query: str) -> str:
    lowered = query.lower().strip()
    if not lowered:
        return ""
    if len(lowered) <= 20 and any(term in lowered for term in ["thank", "thanks", "ok", "okay", "sorry", "great", "bye", "再见", "谢谢", "好的", "可以"]):
        return "short_followup"
    if any(term in lowered for term in ["dollar", "$", "yuan", "ticket", "tip", "price", "charge", "fee", "one hundred", "人均消费", "多少", "预算"]):
        return "transactional_amount"
    if any(term in lowered for term in ["hotel", "room", "airport", "flight", "bus", "taxi", "seat", "station", "reservation", "check in", "酒店", "机票", "公交", "出租车"]):
        return "travel_transport"
    if any(term in lowered for term in ["buy", "store", "engine", "service", "guarantee", "woolen vest", "parcels", "cd", "car", "shop", "deposit a check", "汽车", "电脑", "推荐一款"]):
        return "service_purchase"
    if any(term in lowered for term in ["educational background", "working hours", "manager", "contract", "claim", "office", "公司", "合同", "经理", "工作时间"]):
        return "business_dialogue"
    return ""


def _mix_noise_fragments(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if " " in left or " " in right:
        left_parts = [item for item in left.split(" ") if item]
        right_parts = [item for item in right.split(" ") if item]
        if not left_parts or not right_parts:
            return ""
        left_cut = max(1, len(left_parts) // 2)
        right_cut = max(1, len(right_parts) // 2)
        return " ".join(left_parts[:left_cut] + right_parts[-right_cut:])
    left_cut = max(2, len(left) // 2)
    right_cut = max(2, len(right) // 2)
    return f"{left[:left_cut]}{right[-right_cut:]}"


def _normalize_family(value: object) -> str:
    family = str(value or "")
    if family == "entity_annotation":
        return "entity"
    if family == "alias_catalog":
        return "alias"
    return family


def _stable_row_hash(row: dict) -> int:
    key = str(row.get("split_lock_key") or row.get("query") or row.get("source_row_id") or "")
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
