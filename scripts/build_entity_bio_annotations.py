from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "data"


QUERY_TEMPLATES = [
    "{m}今天为什么跌？",
    "{m}最近为什么涨？",
    "{m}后面还值得拿吗？",
    "{m}现在能买吗？",
    "{m}风险高不高？",
    "{m}基本面怎么样？",
    "{m}估值高不高？",
    "最近有哪些{m}新闻？",
    "{m}最近涨跌怎么样？",
]

PAIR_TEMPLATES = [
    "{a}和{b}哪个好？",
    "{a}和{b}哪个更稳？",
    "{a}和{b}哪个更值得拿？",
]

TRIPLE_TEMPLATES = [
    "{a}、{b}和{c}哪个更稳？",
    "{a}、{b}和{c}哪个好？",
]


def build_spans(query: str, mentions: list[str]) -> list[dict]:
    spans: list[dict] = []
    occupied: list[tuple[int, int]] = []
    for mention in sorted(mentions, key=len, reverse=True):
        start = query.find(mention)
        if start == -1:
            continue
        end = start + len(mention)
        if any(not (end <= s or start >= e) for s, e in occupied):
            continue
        occupied.append((start, end))
        spans.append({"text": mention, "start": start, "end": end, "label": "ENT"})
    spans.sort(key=lambda item: item["start"])
    return spans


def load_aliases() -> list[str]:
    aliases = []
    with (DATA_DIR / "alias_table.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            alias = row["alias_text"].strip()
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


def load_supervised_queries() -> list[str]:
    path = DATA_DIR / "query_labels.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [row["query"] for row in csv.DictReader(handle)]


def generate_annotations() -> list[dict]:
    aliases = load_aliases()
    rows: list[dict] = []

    for mention in aliases:
        for template in QUERY_TEMPLATES:
            query = template.format(m=mention)
            rows.append({"query": query, "spans": build_spans(query, [mention])})

    for idx in range(0, len(aliases) - 1, 2):
        a = aliases[idx]
        b = aliases[idx + 1]
        for template in PAIR_TEMPLATES:
            query = template.format(a=a, b=b)
            rows.append({"query": query, "spans": build_spans(query, [a, b])})

    triples = [
        ("中国平安", "平安银行", "茅台"),
        ("证券ETF", "创业板ETF", "沪深300ETF"),
        ("上证指数", "沪深300", "大盘"),
        ("五粮液", "中国平安", "平安银行"),
    ]
    for a, b, c in triples:
        for template in TRIPLE_TEMPLATES:
            query = template.format(a=a, b=b, c=c)
            rows.append({"query": query, "spans": build_spans(query, [a, b, c])})

    for query in load_supervised_queries():
        mentions = [alias for alias in aliases if alias and alias in query]
        if mentions:
            rows.append({"query": query, "spans": build_spans(query, mentions)})

    deduped = []
    seen = set()
    for row in rows:
        if row["query"] in seen:
            continue
        seen.add(row["query"])
        deduped.append(row)
    return deduped[:280]


def main() -> None:
    output_path = DATA_DIR / "entity_bio_annotations.jsonl"
    rows = generate_annotations()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"annotations": len(rows), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
