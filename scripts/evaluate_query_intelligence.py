from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.config import Settings
from query_intelligence.data_loader import load_entities
from query_intelligence.nlu.pipeline import NLUPipeline
from query_intelligence.retrieval.feature_builder import FeatureBuilder
from query_intelligence.retrieval.query_builder import QueryBuilder
from query_intelligence.retrieval.ranker import HybridRanker
from training.progress import ProgressBar


MANIFEST_PATH = ROOT / "data" / "training_assets" / "manifest.json"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TOTAL_QUERIES = 10_000
RETRIEVAL_QREL_QUERIES = 1_500
OOD_RETRIEVAL_CHECK_QUERIES = 500
TOP_K = 10

FINANCE_CLASSIFICATION_TARGET = 4_000
FINANCE_RETRIEVAL_TARGET = 1_500
OOD_PUBLIC_TARGET = 3_000
OOD_DIALOGUE_TARGET = 1_000
FINANCE_GENERATED_TARGET = 1_500
OOD_GENERATED_TARGET = 1_500
OOD_MUST_PASS_TARGET = 500

OOD_SOURCE_IDS = {"qrecc", "risawoz"}
FINANCE_RETRIEVAL_SOURCE_IDS = {"fincprg", "fir_bench_reports", "fir_bench_announcements", "fiqa"}
FINANCE_CLASSIFICATION_EXCLUDE = OOD_SOURCE_IDS | {"finfe", "chnsenticorp", "fin_news_sentiment"}
FINANCE_SOURCE_TYPES = {"news", "announcement", "research_note", "faq", "product_doc", "market_api", "fundamental_sql", "industry_sql", "macro_sql"}

BENCHMARK_REFERENCES = [
    {
        "name": "BEIR / FiQA-2018",
        "url": "https://huggingface.co/datasets/BeIR/fiqa",
        "proxy_metric": "NDCG@10",
        "proxy_target": 0.30,
        "note": "Public finance retrieval benchmark proxy; strong sparse/dense systems are usually above a 0.30 NDCG@10 floor.",
    },
    {
        "name": "T2Ranking",
        "url": "https://huggingface.co/datasets/THUIR/T2Ranking",
        "proxy_metric": "MRR@10",
        "proxy_target": 0.45,
        "note": "Chinese retrieval benchmark proxy; competitive reranking systems are expected to clear a mid-0.4 MRR@10 floor.",
    },
]

THRESHOLDS = {
    "finance_domain_recall": 0.97,
    "ood_rejection_accuracy": 0.95,
    "must_pass_ood_accuracy": 1.0,
    "product_type_accuracy": 0.90,
    "question_style_accuracy": 0.84,
    "intent_micro_f1": 0.78,
    "topic_micro_f1": 0.82,
    "source_plan_hit_at_5": 0.85,
    "source_plan_recall_at_5": 0.70,
    "clarification_recall": 0.90,
    "source_plan_support": 0.80,
    "recall_at_10": 0.75,
    "mrr_at_10": 0.45,
    "ndcg_at_10": 0.50,
    "ood_retrieval_abstention": 0.95,
}


@dataclass
class EvalItem:
    query: str
    bucket: str
    domain: str
    language: str
    expected_product_type: str | None = None
    expected_question_style: str | None = None
    expected_intents: tuple[str, ...] = ()
    expected_topics: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    expected_clarification: bool = False
    expected_out_of_scope: bool = False
    qrel_relevances: dict[str, int] | None = None
    positive_source_types: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    classification_rows = list(_read_jsonl(ROOT / "data" / "training_assets" / "classification.jsonl"))
    qrel_rows = list(_read_jsonl(ROOT / "data" / "training_assets" / "qrels.jsonl"))
    corpus_rows = list(_read_jsonl(ROOT / "data" / "training_assets" / "retrieval_corpus.jsonl"))

    master_items = build_master_eval_set(classification_rows, qrel_rows, corpus_rows)
    nlu_pipeline = NLUPipeline.build_default(Settings(models_dir=str(MODELS_DIR)))
    nlu_outputs = run_nlu_eval(master_items, nlu_pipeline)
    retrieval_report = run_retrieval_eval(master_items, nlu_outputs, corpus_rows)
    summary = build_summary(master_items, nlu_outputs, retrieval_report)

    json_path = REPORT_DIR / "2026-04-23-full-stack-eval.json"
    md_path = REPORT_DIR / "2026-04-23-full-stack-eval.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(summary), encoding="utf-8")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path)}, ensure_ascii=False), flush=True)


def build_master_eval_set(classification_rows: list[dict], qrel_rows: list[dict], corpus_rows: list[dict]) -> list[EvalItem]:
    finance_labeled = _sample_finance_classification_rows(classification_rows, FINANCE_CLASSIFICATION_TARGET)
    finance_retrieval = _sample_finance_retrieval_queries(qrel_rows, corpus_rows, FINANCE_RETRIEVAL_TARGET)
    ood_public = _sample_ood_public_queries(classification_rows, qrel_rows, OOD_PUBLIC_TARGET)
    ood_dialogue = _sample_ood_dialogue_queries(OOD_DIALOGUE_TARGET)
    finance_generated = _generate_finance_boundary_queries(FINANCE_GENERATED_TARGET)
    finance_must_pass = _generate_must_pass_finance_queries()
    ood_generated = _generate_ood_queries(OOD_GENERATED_TARGET)
    ood_must_pass = _generate_must_pass_ood_queries()

    master_items = _dedupe_eval_items(
        [
            *ood_must_pass,
            *finance_must_pass,
            *finance_labeled,
            *finance_retrieval,
            *ood_public,
            *ood_dialogue,
            *finance_generated,
            *ood_generated,
        ]
    )
    if len(master_items) < TOTAL_QUERIES:
        raise RuntimeError(f"evaluation set too small: {len(master_items)} < {TOTAL_QUERIES}")
    return master_items[:TOTAL_QUERIES]


def _sample_finance_classification_rows(rows: list[dict], target: int) -> list[EvalItem]:
    pool: list[EvalItem] = []
    for row in rows:
        if row.get("split") not in {"valid", "test"}:
            continue
        if str(row.get("source_id") or "") in FINANCE_CLASSIFICATION_EXCLUDE:
            continue
        product_type = str(row.get("product_type") or "").strip()
        question_style = str(row.get("question_style") or "").strip()
        if product_type in {"", "unknown"} or question_style == "":
            continue
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        expected_sources = tuple(_ordered_unique(_as_tuple(row.get("expected_document_sources")) + _as_tuple(row.get("expected_structured_sources"))))
        pool.append(
            EvalItem(
                query=query,
                bucket="finance_public",
                domain="finance",
                language=_detect_language(query),
                expected_product_type=product_type,
                expected_question_style=question_style,
                expected_intents=tuple(_as_tuple(row.get("intent_labels"))),
                expected_topics=tuple(_as_tuple(row.get("topic_labels"))),
                expected_sources=expected_sources,
                metadata={"source_id": row.get("source_id"), "split": row.get("split")},
            )
        )
    return _stable_group_sample(pool, target, lambda item: f"{item.metadata.get('source_id')}::{item.expected_question_style}")


def _sample_finance_retrieval_queries(qrel_rows: list[dict], corpus_rows: list[dict], target: int) -> list[EvalItem]:
    doc_source_types = {
        str(row.get("doc_id") or ""): _normalize_source_type(row)
        for row in corpus_rows
    }
    grouped: dict[str, dict[str, Any]] = {}
    for row in qrel_rows:
        if row.get("split") not in {"valid", "test"}:
            continue
        source_id = str(row.get("source_id") or "")
        if source_id not in FINANCE_RETRIEVAL_SOURCE_IDS:
            continue
        query = str(row.get("query") or "").strip()
        doc_id = str(row.get("doc_id") or "").strip()
        relevance = int(row.get("relevance") or 0)
        if not query or not doc_id or relevance <= 0:
            continue
        item = grouped.setdefault(
            query,
            {
                "source_id": source_id,
                "qrel_relevances": {},
                "positive_source_types": set(),
            },
        )
        item["qrel_relevances"][doc_id] = max(relevance, item["qrel_relevances"].get(doc_id, 0))
        source_type = doc_source_types.get(doc_id)
        if source_type:
            item["positive_source_types"].add(source_type)

    pool = [
        EvalItem(
            query=query,
            bucket="finance_retrieval",
            domain="finance",
            language=_detect_language(query),
            expected_sources=tuple(sorted(item["positive_source_types"])),
            qrel_relevances=dict(item["qrel_relevances"]),
            positive_source_types=tuple(sorted(item["positive_source_types"])),
            metadata={"source_id": item["source_id"]},
        )
        for query, item in grouped.items()
        if item["qrel_relevances"]
    ]
    return _stable_group_sample(pool, target, lambda item: str(item.metadata.get("source_id")))


def _sample_ood_public_queries(classification_rows: list[dict], qrel_rows: list[dict], target: int) -> list[EvalItem]:
    pool: list[EvalItem] = []
    for row in classification_rows:
        if row.get("split") not in {"valid", "test"}:
            continue
        source_id = str(row.get("source_id") or "")
        if source_id not in OOD_SOURCE_IDS:
            continue
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        pool.append(
            EvalItem(
                query=query,
                bucket="ood_public",
                domain="ood",
                language=_detect_language(query),
                expected_out_of_scope=True,
                metadata={"source_id": source_id},
            )
        )

    seen = {item.query for item in pool}
    for row in qrel_rows:
        if row.get("split") not in {"valid", "test"}:
            continue
        source_id = str(row.get("source_id") or "")
        if source_id != "t2ranking":
            continue
        query = str(row.get("query") or "").strip()
        if not query or query in seen:
            continue
        pool.append(
            EvalItem(
                query=query,
                bucket="ood_public",
                domain="ood",
                language=_detect_language(query),
                expected_out_of_scope=True,
                metadata={"source_id": source_id},
            )
        )
        seen.add(query)
    return _stable_group_sample(pool, target, lambda item: str(item.metadata.get("source_id")))


def _sample_ood_dialogue_queries(target: int) -> list[EvalItem]:
    pool: list[EvalItem] = []
    raw_root = ROOT / "data" / "external" / "raw" / "dailydialog" / "default"
    candidate_paths = sorted(raw_root.rglob("dialogues_*test*.txt")) + sorted(raw_root.rglob("dialogues_*validation*.txt"))
    for path in candidate_paths:
        if any(marker in path.name for marker in ("dialogues_act_", "dialogues_emotion_")):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_index, line in enumerate(lines):
            utterances = [item.strip() for item in line.split("__eou__") if item.strip()]
            for turn_index, utterance in enumerate(utterances):
                if len(utterance) < 2:
                    continue
                pool.append(
                    EvalItem(
                        query=utterance,
                        bucket="ood_dialogue_public",
                        domain="ood",
                        language=_detect_language(utterance),
                        expected_out_of_scope=True,
                        metadata={"source": "dailydialog", "path": path.name, "line_index": line_index, "turn_index": turn_index},
                    )
                )
    return _stable_sample(_dedupe_eval_items(pool), target, lambda item: item.query)


def _generate_finance_boundary_queries(target: int) -> list[EvalItem]:
    entities = _finance_entities()
    etf_terms = ["沪深300ETF", "证券ETF", "创业板ETF", "红利低波ETF", "中证A500ETF"]
    fund_terms = ["沪深300指数基金", "偏股混合基金", "债券基金", "LOF基金", "公募基金"]
    macro_terms = ["降息", "降准", "CPI", "PMI", "财政赤字", "美联储议息"]
    sector_terms = ["白酒板块", "券商板块", "半导体板块", "银行板块", "新能源板块"]
    advice_modifiers = ["最近", "现在", "短期", "长期", "今年下半年", "回调之后", "大跌以后", "高位回撤后"]
    comparison_phrases = [
        ("有什么区别", "compare"),
        ("哪个更适合长期定投", "advice"),
        ("哪个波动更小", "compare"),
        ("哪个更适合新手", "advice"),
        ("长期拿哪个更稳", "advice"),
    ]
    fact_templates = [
        "{entity}在2024年第一季度的营收和净利润分别是多少？",
        "{entity}最新财报里营收、净利润和毛利率分别是多少？",
        "How much revenue and net profit did {entity} report in Q1 2024?",
        "What do the latest fundamentals of {entity} look like?",
    ]
    why_templates = [
        "{entity}{modifier}为什么跌？",
        "{entity}{modifier}为什么涨？",
        "Why did {entity} {modifier_en} fall?",
        "Why did {entity} {modifier_en} rally?",
    ]
    macro_templates = [
        ("{macro}对{sector}有什么影响？", "fact", ("macro_policy_impact",), ("macro", "policy", "industry")),
        ("{macro}之后{sector}会怎么走？", "forecast", ("macro_policy_impact", "price_query"), ("macro", "policy", "industry", "price")),
        ("How does {macro} affect the {sector}?", "fact", ("macro_policy_impact",), ("macro", "policy", "industry")),
        ("What happens to the {sector} after {macro}?", "forecast", ("macro_policy_impact", "price_query"), ("macro", "policy", "industry", "price")),
    ]

    rows: list[EvalItem] = []
    for anchor in ["这个", "它", "这只", "那只", "这一只"]:
        for phrase in [
            f"{anchor}能买吗？",
            f"{anchor}现在还能拿吗？",
            f"{anchor}该不该止损？",
            f"{anchor}适合继续定投吗？",
        ]:
            rows.append(
                EvalItem(
                    query=phrase,
                    bucket="finance_generated_clarification",
                    domain="finance",
                    language=_detect_language(phrase),
                    expected_question_style="advice",
                    expected_clarification=True,
                )
            )
    for phrase in [
        "Should I buy it now?",
        "Would you still hold it?",
        "Is this still worth buying?",
        "Should I cut the loss here?",
    ]:
        rows.append(
            EvalItem(
                query=phrase,
                bucket="finance_generated_clarification",
                domain="finance",
                language="en",
                expected_question_style="advice",
                expected_clarification=True,
            )
        )

    for left in etf_terms:
        for right in fund_terms:
            for phrase, style in comparison_phrases:
                rows.append(
                    EvalItem(
                        query=f"{left}和{right}{phrase}？",
                        bucket="finance_generated_compare",
                        domain="finance",
                        language="zh",
                        expected_product_type="etf",
                        expected_question_style=style,
                        expected_sources=("research_note", "faq", "product_doc"),
                    )
                )
            rows.append(
                EvalItem(
                    query=f"Compare {left} and {right} for long-term investing.",
                    bucket="finance_generated_compare",
                    domain="finance",
                    language="en",
                    expected_product_type="etf",
                    expected_question_style="compare",
                    expected_sources=("research_note", "faq", "product_doc"),
                )
            )

    for entity in entities:
        for modifier in advice_modifiers:
            modifier_en = modifier.replace("最近", "recently").replace("现在", "now").replace("短期", "in the short term").replace("长期", "in the long term").replace("今年下半年", "in the second half of this year").replace("回调之后", "after the pullback").replace("大跌以后", "after the selloff").replace("高位回撤后", "after the drawdown")
            for template in why_templates:
                rows.append(
                    EvalItem(
                        query=template.format(entity=entity, modifier=modifier, modifier_en=modifier_en),
                        bucket="finance_generated_why",
                        domain="finance",
                        language=_detect_language(template),
                        expected_product_type="stock",
                        expected_question_style="why" if "Why" not in template else "why",
                        expected_intents=("market_explanation",),
                        expected_topics=("price",),
                        expected_sources=("news", "research_note", "announcement"),
                    )
                )
        for template in fact_templates:
            rows.append(
                EvalItem(
                    query=template.format(entity=entity),
                    bucket="finance_generated_fact",
                    domain="finance",
                    language=_detect_language(template),
                    expected_product_type="stock",
                    expected_question_style="fact",
                    expected_intents=("fundamental_analysis",),
                    expected_topics=("fundamentals",),
                    expected_sources=("research_note", "announcement", "fundamental_sql"),
                )
            )

    for macro in macro_terms:
        for sector in sector_terms:
            for template, style, intents, topics in macro_templates:
                rows.append(
                    EvalItem(
                        query=template.format(macro=macro, sector=sector),
                        bucket="finance_generated_macro",
                        domain="finance",
                        language=_detect_language(template),
                        expected_product_type="macro",
                        expected_question_style=style,
                        expected_intents=intents,
                        expected_topics=topics,
                        expected_sources=("news", "research_note", "macro_sql"),
                    )
                )
    return _stable_sample(_dedupe_eval_items(rows), target, lambda item: item.query)


def _generate_must_pass_finance_queries() -> list[EvalItem]:
    return [
        EvalItem(
            query="你觉得中国平安怎么样",
            bucket="finance_must_pass_open_evaluation",
            domain="finance",
            language="zh",
            expected_product_type="stock",
            expected_question_style="advice",
            expected_intents=("hold_judgment", "fundamental_analysis", "market_explanation"),
            expected_topics=("fundamentals", "valuation", "risk", "price", "news"),
            expected_sources=("research_note", "news", "announcement", "market_api", "fundamental_sql"),
        ),
        EvalItem(
            query="你怎么看贵州茅台",
            bucket="finance_must_pass_open_evaluation",
            domain="finance",
            language="zh",
            expected_product_type="stock",
            expected_question_style="advice",
            expected_intents=("hold_judgment", "fundamental_analysis", "market_explanation"),
            expected_topics=("fundamentals", "valuation", "risk", "price", "news"),
            expected_sources=("research_note", "news", "announcement", "market_api", "fundamental_sql"),
        ),
        EvalItem(
            query="你觉得中华汽车怎么样",
            bucket="finance_must_pass_unknown_entity",
            domain="finance",
            language="zh",
            expected_product_type="stock",
            expected_question_style="advice",
            expected_intents=("hold_judgment", "fundamental_analysis", "market_explanation"),
            expected_topics=("fundamentals", "valuation", "risk", "price", "news"),
            expected_clarification=True,
        ),
        EvalItem(
            query="星辰科技值得关注吗",
            bucket="finance_must_pass_unknown_entity",
            domain="finance",
            language="zh",
            expected_product_type="stock",
            expected_question_style="advice",
            expected_intents=("hold_judgment", "fundamental_analysis", "market_explanation"),
            expected_topics=("fundamentals", "valuation", "risk", "price", "news"),
            expected_clarification=True,
        ),
    ]


def _generate_ood_queries(target: int) -> list[EvalItem]:
    zh_topics = {
        "编程": ["Python快速排序", "Java并发", "Rust生命周期", "SQL索引优化", "前端动画"],
        "旅行": ["巴黎三日游", "东京地铁", "苏州园林", "冰岛环岛", "签证材料"],
        "医疗": ["胸痛急救", "发烧处理", "糖尿病饮食", "抑郁症状", "儿童咳嗽"],
        "体育": ["欧冠决赛", "NBA季后赛", "马拉松训练", "足球越位", "羽毛球发力"],
        "数学": ["二次方程", "线性代数", "概率分布", "微积分极限", "图论最短路"],
        "烹饪": ["意面做法", "红烧肉", "寿司米饭", "烘焙发酵", "咖啡萃取"],
        "法律": ["合同违约", "劳动仲裁", "刑事责任", "著作权侵权", "取证流程"],
        "历史": ["汉武帝", "法国大革命", "罗马帝国", "丝绸之路", "冷战格局"],
    }
    en_topics = {
        "coding": ["Python quicksort", "Java concurrency", "Rust lifetimes", "SQL indexing", "frontend animation"],
        "travel": ["Paris itinerary", "Tokyo subway", "Iceland ring road", "visa checklist", "best museums"],
        "medical": ["chest pain first aid", "fever treatment", "diabetes diet", "depression symptoms", "child cough"],
        "sports": ["Champions League final", "NBA playoffs", "marathon training", "offside rule", "badminton footwork"],
        "math": ["quadratic equation", "linear algebra", "probability distribution", "calculus limit", "shortest path"],
        "cooking": ["pasta recipe", "braised pork", "sushi rice", "bread proofing", "espresso extraction"],
        "law": ["contract breach", "labor arbitration", "criminal liability", "copyright infringement", "evidence rules"],
        "history": ["Han dynasty", "French Revolution", "Roman Empire", "Silk Road", "Cold War"],
    }
    templates_zh = [
        "请详细解释一下{item}。",
        "{item}怎么做？",
        "{item}和初学者有什么关系？",
        "{item}常见错误有哪些？",
        "{item}最近有什么变化？",
        "帮我列一个关于{item}的学习计划。",
        "{item}有哪些权威资料推荐？",
        "{item}和相近概念有什么区别？",
    ]
    templates_en = [
        "Explain {item} in detail.",
        "How do I learn {item}?",
        "What are common mistakes in {item}?",
        "Can you compare approaches to {item}?",
        "What changed recently about {item}?",
        "Give me a study plan for {item}.",
        "What are authoritative references for {item}?",
        "How is {item} different from related concepts?",
    ]
    rows: list[EvalItem] = []
    for _, items in zh_topics.items():
        for item in items:
            for template in templates_zh:
                rows.append(EvalItem(query=template.format(item=item), bucket="ood_generated", domain="ood", language="zh", expected_out_of_scope=True))
    for _, items in en_topics.items():
        for item in items:
            for template in templates_en:
                rows.append(EvalItem(query=template.format(item=item), bucket="ood_generated", domain="ood", language="en", expected_out_of_scope=True))
    return _stable_sample(_dedupe_eval_items(rows), target, lambda item: item.query)


def _generate_must_pass_ood_queries() -> list[EvalItem]:
    queries = [
        "你好",
        "你好？你是谁？",
        "啊数据库的建立撒娇的你离开",
        "你是谁",
        "你能做什么",
        "hello",
        "hello, who are you?",
        "who are you",
        "what can you do",
        "今天天气怎么样",
        "上海今天下雨吗",
        "write a python bubble sort",
        "帮我写个 Python 冒泡排序",
        "translate this sentence into chinese",
        "讲个笑话",
        "who won the nba game tonight",
        "法国大革命是什么",
        "胸痛急救怎么做",
        "巴黎三日游怎么安排",
        "意面怎么做",
        "什么是线性代数",
    ]
    rows = [
        EvalItem(
            query=query,
            bucket="ood_must_pass",
            domain="ood",
            language=_detect_language(query),
            expected_out_of_scope=True,
        )
        for query in queries
    ]
    return _stable_sample(rows, min(OOD_MUST_PASS_TARGET, len(rows)), lambda item: item.query)


def run_nlu_eval(items: list[EvalItem], nlu_pipeline: NLUPipeline) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    progress = ProgressBar("eval:nlu", len(items), every=250)
    progress.update(0, "starting", force=True)
    for index, item in enumerate(items, start=1):
        outputs[item.query] = nlu_pipeline.run(query=item.query, user_profile={}, dialog_context=[], debug=False)
        progress.update(index, "running NLU")
    progress.finish("NLU complete")
    return outputs


def run_retrieval_eval(items: list[EvalItem], nlu_outputs: dict[str, dict], corpus_rows: list[dict]) -> dict[str, Any]:
    ranker = HybridRanker.load_model(MODELS_DIR / "ranker.joblib")
    feature_builder = FeatureBuilder()
    query_builder = QueryBuilder()
    doc_index = {str(row.get("doc_id") or ""): _normalize_doc(row) for row in corpus_rows if str(row.get("doc_id") or "").strip()}
    doc_ids = sorted(doc_index)

    retrieval_items = [item for item in items if item.qrel_relevances][:RETRIEVAL_QREL_QUERIES]
    ood_items = [item for item in items if item.domain == "ood"][:OOD_RETRIEVAL_CHECK_QUERIES]

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    source_plan_support_hits = 0
    total = 0
    progress = ProgressBar("eval:retrieval", len(retrieval_items), every=100)
    progress.update(0, "starting", force=True)
    for index, item in enumerate(retrieval_items, start=1):
        nlu_result = nlu_outputs[item.query]
        query_bundle = query_builder.build(nlu_result)
        qrel_relevances = item.qrel_relevances or {}
        candidate_docs = _candidate_docs_for_query(item.query, qrel_relevances, doc_index, doc_ids, 40)
        ranked = _rank_candidates(query_bundle, candidate_docs, feature_builder, ranker)
        ranked_ids = [row["doc_id"] for row in ranked[:TOP_K]]
        if item.positive_source_types and set(item.positive_source_types).intersection(nlu_result.get("source_plan", [])[:5]):
            source_plan_support_hits += 1
        recall_hits += int(any(qrel_relevances.get(doc_id, 0) > 0 for doc_id in ranked_ids))
        reciprocal_ranks.append(_mrr_at_k(ranked_ids, qrel_relevances, TOP_K))
        ndcgs.append(_ndcg_at_k(ranked_ids, qrel_relevances, TOP_K))
        total += 1
        progress.update(index, "running retrieval")
    progress.finish("retrieval complete")

    abstentions = 0
    for item in ood_items:
        nlu_result = nlu_outputs[item.query]
        abstentions += int("out_of_scope_query" in nlu_result.get("risk_flags", []) or not nlu_result.get("source_plan"))

    return {
        "retrieval_eval_queries": len(retrieval_items),
        "ood_retrieval_queries": len(ood_items),
        "recall_at_10": round(recall_hits / max(total, 1), 4),
        "mrr_at_10": round(sum(reciprocal_ranks) / max(total, 1), 4),
        "ndcg_at_10": round(sum(ndcgs) / max(total, 1), 4),
        "source_plan_support": round(source_plan_support_hits / max(total, 1), 4),
        "ood_retrieval_abstention": round(abstentions / max(len(ood_items), 1), 4),
    }


def build_summary(items: list[EvalItem], nlu_outputs: dict[str, dict], retrieval_report: dict[str, Any]) -> dict[str, Any]:
    finance_items = [item for item in items if item.domain == "finance" and _is_self_contained_finance(item.query)]
    ood_items = [item for item in items if item.domain == "ood"]
    must_pass_items = [item for item in items if item.bucket == "ood_must_pass"]

    reliable_buckets = {
        "finance_generated_macro",
        "finance_generated_compare",
        "finance_generated_why",
        "finance_generated_fact",
        "finance_generated_clarification",
    }
    reliable_items = [item for item in items if item.bucket in reliable_buckets]

    product_eval = [item for item in reliable_items if item.expected_product_type]
    question_style_eval = [item for item in reliable_items if item.expected_question_style]
    source_plan_eval = [item for item in reliable_items if item.expected_sources]
    clarification_eval = [item for item in items if item.expected_clarification]
    intent_eval = [item for item in reliable_items if item.expected_intents]
    topic_eval = [item for item in reliable_items if item.expected_topics]

    product_hits = sum(
        1 for item in product_eval
        if _predicted_product_type(nlu_outputs[item.query]) == item.expected_product_type
    )
    question_style_hits = sum(
        1 for item in question_style_eval
        if nlu_outputs[item.query]["question_style"] == item.expected_question_style
    )
    finance_domain_hits = sum(
        1 for item in finance_items
        if "out_of_scope_query" not in nlu_outputs[item.query].get("risk_flags", [])
    )
    ood_hits = sum(
        1 for item in ood_items
        if "out_of_scope_query" in nlu_outputs[item.query].get("risk_flags", [])
    )
    must_pass_hits = sum(
        1 for item in must_pass_items
        if "out_of_scope_query" in nlu_outputs[item.query].get("risk_flags", [])
    )
    source_plan_hits = 0
    source_plan_recalls: list[float] = []
    for item in source_plan_eval:
        predicted = list(nlu_outputs[item.query].get("source_plan", [])[:5])
        expected = set(item.expected_sources)
        overlap = expected.intersection(predicted)
        source_plan_hits += int(bool(overlap))
        source_plan_recalls.append(len(overlap) / max(len(expected), 1))

    clarification_hits = sum(
        1
        for item in clarification_eval
        if "clarification_required" in nlu_outputs[item.query].get("risk_flags", [])
        and "missing_entity" in nlu_outputs[item.query].get("missing_slots", [])
    )

    intent_precision, intent_recall, intent_f1 = _micro_f1(
        [
            (
                set(item.expected_intents),
                {label["label"] for label in nlu_outputs[item.query].get("intent_labels", [])},
            )
            for item in intent_eval
        ]
    )
    topic_precision, topic_recall, topic_f1 = _micro_f1(
        [
            (
                set(item.expected_topics),
                {label["label"] for label in nlu_outputs[item.query].get("topic_labels", [])},
            )
            for item in topic_eval
        ]
    )

    metrics = {
        "finance_domain_recall": round(finance_domain_hits / max(len(finance_items), 1), 4),
        "ood_rejection_accuracy": round(ood_hits / max(len(ood_items), 1), 4),
        "must_pass_ood_accuracy": round(must_pass_hits / max(len(must_pass_items), 1), 4),
        "product_type_accuracy": round(product_hits / max(len(product_eval), 1), 4),
        "question_style_accuracy": round(question_style_hits / max(len(question_style_eval), 1), 4),
        "intent_micro_precision": round(intent_precision, 4),
        "intent_micro_recall": round(intent_recall, 4),
        "intent_micro_f1": round(intent_f1, 4),
        "topic_micro_precision": round(topic_precision, 4),
        "topic_micro_recall": round(topic_recall, 4),
        "topic_micro_f1": round(topic_f1, 4),
        "source_plan_hit_at_5": round(source_plan_hits / max(len(source_plan_eval), 1), 4),
        "source_plan_recall_at_5": round(sum(source_plan_recalls) / max(len(source_plan_recalls), 1), 4),
        "clarification_recall": round(clarification_hits / max(len(clarification_eval), 1), 4),
        **retrieval_report,
    }
    verdicts = {
        metric: {
            "value": value,
            "threshold": THRESHOLDS.get(metric),
            "pass": value >= THRESHOLDS.get(metric, 0.0),
        }
        for metric, value in metrics.items()
        if metric in THRESHOLDS
    }
    composition = {
        "total_queries": len(items),
        "bucket_counts": Counter(item.bucket for item in items),
        "domain_counts": Counter(item.domain for item in items),
        "language_counts": Counter(item.language for item in items),
        "finance_domain_eval_queries": len(finance_items),
        "reliable_nlu_eval_queries": len(reliable_items),
        "must_pass_queries": len(must_pass_items),
        "product_eval_queries": len(product_eval),
        "question_style_eval_queries": len(question_style_eval),
        "intent_eval_queries": len(intent_eval),
        "topic_eval_queries": len(topic_eval),
        "source_plan_eval_queries": len(source_plan_eval),
        "clarification_eval_queries": len(clarification_eval),
    }
    failure_examples = _build_failure_examples(
        finance_items=finance_items,
        must_pass_items=must_pass_items,
        product_eval=product_eval,
        question_style_eval=question_style_eval,
        intent_eval=intent_eval,
        topic_eval=topic_eval,
        source_plan_eval=source_plan_eval,
        clarification_eval=clarification_eval,
        nlu_outputs=nlu_outputs,
    )
    return {
        "eval_set_composition": composition,
        "metrics": metrics,
        "verdicts": verdicts,
        "benchmark_references": BENCHMARK_REFERENCES,
        "failure_examples": failure_examples,
        "failed_metrics": [metric for metric, payload in verdicts.items() if not payload["pass"]],
    }


def _build_failure_examples(
    *,
    finance_items: list[EvalItem],
    must_pass_items: list[EvalItem],
    product_eval: list[EvalItem],
    question_style_eval: list[EvalItem],
    intent_eval: list[EvalItem],
    topic_eval: list[EvalItem],
    source_plan_eval: list[EvalItem],
    clarification_eval: list[EvalItem],
    nlu_outputs: dict[str, dict],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "finance_domain": [
            _failure_base(item, nlu_outputs[item.query])
            for item in finance_items
            if "out_of_scope_query" in nlu_outputs[item.query].get("risk_flags", [])
        ][:50],
        "must_pass_ood": [
            _failure_base(item, nlu_outputs[item.query])
            for item in must_pass_items
            if "out_of_scope_query" not in nlu_outputs[item.query].get("risk_flags", [])
        ][:50],
        "product_type": [
            {
                **_failure_base(item, nlu_outputs[item.query]),
                "expected": item.expected_product_type,
                "predicted": _predicted_product_type(nlu_outputs[item.query]),
            }
            for item in product_eval
            if _predicted_product_type(nlu_outputs[item.query]) != item.expected_product_type
        ][:50],
        "question_style": [
            {
                **_failure_base(item, nlu_outputs[item.query]),
                "expected": item.expected_question_style,
                "predicted": nlu_outputs[item.query].get("question_style"),
            }
            for item in question_style_eval
            if nlu_outputs[item.query].get("question_style") != item.expected_question_style
        ][:50],
        "intent": [
            {
                **_failure_base(item, nlu_outputs[item.query]),
                "expected": list(item.expected_intents),
                "predicted": [label["label"] for label in nlu_outputs[item.query].get("intent_labels", [])],
            }
            for item in intent_eval
            if set(item.expected_intents) != {label["label"] for label in nlu_outputs[item.query].get("intent_labels", [])}
        ][:50],
        "topic": [
            {
                **_failure_base(item, nlu_outputs[item.query]),
                "expected": list(item.expected_topics),
                "predicted": [label["label"] for label in nlu_outputs[item.query].get("topic_labels", [])],
            }
            for item in topic_eval
            if set(item.expected_topics) != {label["label"] for label in nlu_outputs[item.query].get("topic_labels", [])}
        ][:50],
        "source_plan": [
            {
                **_failure_base(item, nlu_outputs[item.query]),
                "expected": list(item.expected_sources),
                "predicted": nlu_outputs[item.query].get("source_plan", [])[:5],
            }
            for item in source_plan_eval
            if not set(item.expected_sources).intersection(nlu_outputs[item.query].get("source_plan", [])[:5])
        ][:50],
        "clarification": [
            {
                **_failure_base(item, nlu_outputs[item.query]),
                "expected": {"risk_flag": "clarification_required", "missing_slot": "missing_entity"},
                "predicted": {
                    "risk_flags": nlu_outputs[item.query].get("risk_flags", []),
                    "missing_slots": nlu_outputs[item.query].get("missing_slots", []),
                },
            }
            for item in clarification_eval
            if "clarification_required" not in nlu_outputs[item.query].get("risk_flags", [])
            or "missing_entity" not in nlu_outputs[item.query].get("missing_slots", [])
        ][:50],
    }


def _failure_base(item: EvalItem, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": item.query,
        "bucket": item.bucket,
        "domain": item.domain,
        "language": item.language,
        "source_id": (item.metadata or {}).get("source_id"),
        "risk_flags": output.get("risk_flags", []),
        "missing_slots": output.get("missing_slots", []),
        "product_type": output.get("product_type", {}),
        "source_plan": output.get("source_plan", []),
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Query Intelligence Large-Scale Evaluation",
        "",
        "## Eval Set",
        f"- Total queries: {summary['eval_set_composition']['total_queries']}",
        f"- Buckets: {json.dumps(summary['eval_set_composition']['bucket_counts'], ensure_ascii=False)}",
        f"- Domains: {json.dumps(summary['eval_set_composition']['domain_counts'], ensure_ascii=False)}",
        f"- Languages: {json.dumps(summary['eval_set_composition']['language_counts'], ensure_ascii=False)}",
        "",
        "## Metrics",
    ]
    for metric, value in summary["metrics"].items():
        lines.append(f"- {metric}: {value}")
    lines.extend(["", "## Threshold Check"])
    for metric, payload in summary["verdicts"].items():
        lines.append(f"- {metric}: value={payload['value']}, threshold={payload['threshold']}, pass={payload['pass']}")
    lines.extend(["", "## Benchmark Proxies"])
    for item in summary["benchmark_references"]:
        lines.append(f"- {item['name']}: {item['proxy_metric']} >= {item['proxy_target']} ({item['url']})")
    lines.extend(["", "## Failed Metrics"])
    if summary["failed_metrics"]:
        for metric in summary["failed_metrics"]:
            lines.append(f"- {metric}")
    else:
        lines.append("- None")
    lines.extend(["", "## Failure Examples"])
    must_pass_failures = summary.get("failure_examples", {}).get("must_pass_ood", [])
    if must_pass_failures:
        for item in must_pass_failures:
            lines.append(f"- {item['query']}: risk_flags={item['risk_flags']}, product_type={item['product_type']}, source_plan={item['source_plan']}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _candidate_docs_for_query(query: str, qrel_relevances: dict[str, int], doc_index: dict[str, dict], doc_ids: list[str], negative_limit: int) -> list[dict]:
    candidates = [doc_index[doc_id] for doc_id in qrel_relevances if doc_id in doc_index]
    start = int(hashlib.sha1(query.encode("utf-8")).hexdigest(), 16) % max(len(doc_ids), 1)
    cursor = 0
    negatives: list[dict] = []
    excluded = set(qrel_relevances)
    while len(negatives) < negative_limit and cursor < len(doc_ids):
        doc_id = doc_ids[(start + cursor) % len(doc_ids)]
        cursor += 1
        if doc_id in excluded:
            continue
        negatives.append(doc_index[doc_id])
    return candidates + negatives


def _rank_candidates(query_bundle: dict, candidates: list[dict], feature_builder: FeatureBuilder, ranker: HybridRanker) -> list[dict]:
    for candidate in candidates:
        candidate["retrieval_score"] = _lexical_similarity(query_bundle["normalized_query"], candidate.get("title", ""), candidate.get("body", ""))
        feature_json = feature_builder.build(query_bundle, candidate)
        candidate["rank_score"] = ranker.score(feature_json)
    return sorted(candidates, key=lambda item: item["rank_score"], reverse=True)


def _lexical_similarity(query: str, title: str, body: str) -> float:
    haystack = f"{title} {body}".strip()
    if not query or not haystack:
        return 0.0
    q_ngrams = {query[i : i + 3] for i in range(max(len(query) - 2, 1))}
    d_ngrams = {haystack[i : i + 3] for i in range(max(len(haystack) - 2, 1))}
    if not q_ngrams or not d_ngrams:
        return 0.0
    return round(len(q_ngrams & d_ngrams) / max(len(q_ngrams | d_ngrams), 1), 4)


def _mrr_at_k(ranked_ids: list[str], relevances: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevances.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked_ids: list[str], relevances: dict[str, int], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = relevances.get(doc_id, 0)
        if rel <= 0:
            continue
        dcg += (2**rel - 1) / math.log2(rank + 1)
    ideal_rels = sorted((value for value in relevances.values() if value > 0), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _micro_f1(pairs: list[tuple[set[str], set[str]]]) -> tuple[float, float, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for expected, predicted in pairs:
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _normalize_doc(row: dict) -> dict:
    source_id = str(row.get("source_id") or "")
    source_type = _normalize_source_type(row)
    return {
        "doc_id": str(row.get("doc_id") or ""),
        "evidence_id": str(row.get("doc_id") or ""),
        "title": str(row.get("title") or ""),
        "summary": str(row.get("summary") or ""),
        "body": str(row.get("body") or row.get("text") or ""),
        "text": str(row.get("text") or row.get("body") or ""),
        "source_type": source_type,
        "entity_symbols": list(row.get("entity_symbols") or []),
        "credibility_score": 0.9 if source_type in {"announcement", "research_note"} else 0.7,
        "product_type": "stock" if source_id in {"fir_bench_reports", "fir_bench_announcements", "fincprg"} else "unknown",
        "publish_time": row.get("publish_time"),
    }


def _normalize_source_type(row: dict) -> str:
    source_type = str(row.get("source_type") or "").strip()
    if source_type:
        return source_type
    source_id = str(row.get("source_id") or "")
    if source_id == "fir_bench_announcements":
        return "announcement"
    return "research_note"


def _finance_entities() -> list[str]:
    entities = load_entities()
    names = [
        str(row.get("canonical_name") or "").strip()
        for row in entities
        if str(row.get("entity_type") or "").strip() == "stock"
    ]
    names = [name for name in names if name]
    return names[:200] or ["贵州茅台", "中国平安", "道通科技", "宁德时代", "招商银行"]


def _is_self_contained_finance(query: str) -> bool:
    lowered = query.lower()
    finance_terms = [
        "股票",
        "股价",
        "基金",
        "etf",
        "lof",
        "指数",
        "大盘",
        "财报",
        "营收",
        "净利润",
        "估值",
        "市盈率",
        "研报",
        "公告",
        "行业",
        "板块",
        "降息",
        "降准",
        "cpi",
        "pmi",
        "gdp",
        "buy",
        "sell",
        "hold",
        "stock",
        "fund",
        "market",
        "earnings",
        "valuation",
        "macro",
        "policy",
    ]
    return any(term in lowered for term in finance_terms)


def _predicted_product_type(nlu_result: dict) -> str:
    product = nlu_result.get("product_type") or {}
    return str(product.get("label") or "")


def _dedupe_eval_items(items: list[EvalItem]) -> list[EvalItem]:
    deduped: dict[str, EvalItem] = {}
    for item in items:
        deduped.setdefault(item.query, item)
    return list(deduped.values())


def _detect_language(text: str) -> str:
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return "en" if ascii_letters > cjk_chars else "zh"


def _stable_group_sample(items: list[EvalItem], target: int, group_key) -> list[EvalItem]:
    grouped: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        grouped[str(group_key(item))].append(item)
    if not grouped:
        return []
    per_group = max(target // len(grouped), 1)
    sampled: list[EvalItem] = []
    for key in sorted(grouped):
        sampled.extend(_stable_sample(grouped[key], min(per_group, len(grouped[key])), lambda item: item.query))
    if len(sampled) < target:
        seen = {item.query for item in sampled}
        remainder = [item for item in items if item.query not in seen]
        sampled.extend(_stable_sample(remainder, target - len(sampled), lambda item: item.query))
    return sampled[:target]


def _stable_sample(items: list[EvalItem], target: int, key_fn) -> list[EvalItem]:
    ordered = sorted(items, key=lambda item: _stable_hash(key_fn(item)))
    return ordered[:target]


def _stable_hash(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split("|") if item.strip())
    return ()


def _ordered_unique(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen or not value:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


if __name__ == "__main__":
    main()
