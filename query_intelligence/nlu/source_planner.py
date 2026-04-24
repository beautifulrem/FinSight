from __future__ import annotations

from dataclasses import dataclass
import re

from ..query_terms import contains_bucket_market_term
from .classifiers import MultiLabelClassifier
from .source_plan_reranker import SourcePlanReranker


SOURCE_PRIORITY = [
    "market_api",
    "news",
    "industry_sql",
    "fundamental_sql",
    "announcement",
    "research_note",
    "faq",
    "product_doc",
    "macro_sql",
]


GENERAL_FINANCE_TERMS = [
    "invest",
    "investment",
    "investing",
    "return on capital",
    "portfolio",
    "rebalanc",
    "rebalancing",
    "hedge",
    "inflation",
    "trading",
    "trader",
    "market order",
    "short selling",
    "stock",
    "shareholder",
    "options",
    "option",
    "futures",
    "forex",
    "broker",
    "brokerage",
    "brokerages",
    "credit card",
    "debit",
    "prepaid card",
    "mortgage",
    "loan",
    "tax",
    "ira",
    "401k",
    "bank account",
    "insurance",
    "budget",
    "debt",
    "gold",
    "finance",
    "financial",
    "kickstarter",
    "ofx",
    "ach",
    "echeck",
    "echecks",
    "dollar",
    "usd",
    "eur",
    "gbp",
    "yen",
    "currency",
    "exchange rate",
    "capital gains",
    "agi",
    "vat",
    "writeoff",
    "write-off",
    "deduct",
    "deductible",
    "1099",
    "1099s",
    "irs",
    "w-2",
    "rrsp",
    "rsu",
    "hsa",
    "hst",
    "payroll",
    "reimbursement",
    "income",
    "expense",
    "expenses",
    "cash flow",
    "apr",
    "roi",
    "present value",
    "perpetuity",
    "scholarship",
    "llc",
    "s-corp",
    "sole proprietor",
    "self-employed",
    "partnership",
    "startup",
    "business",
    "lender",
    "housing lender",
    "landlord",
    "rental",
    "renting",
    "real estate",
    "property",
    "homebuyers",
    "foreclosed",
    "lien",
    "car loan",
    "car finance",
    "lease",
    "rewards card",
    "visa card",
    "mastercard",
    "discover card",
    "pre-market",
    "after hours",
    "market cap",
    "market caps",
    "market capitalization",
    "price target",
    "price action",
    "technical analysis",
    "financial statements",
    "financial reports",
    "ftse",
    "p/e",
    "pe ratio",
    "price earnings",
    "ohlc",
    "dividend yield",
    "dividend yeild",
    "securities",
    "security",
    "derivative",
    "derivatives",
    "debenture",
    "fixed-income",
    "duration",
    "liquidity",
    "slippage",
    "spread",
    "volume",
    "volatility",
    "implied volatility",
    "average true range",
    "limit order",
    "stop limit",
    "margin",
    "bankruptcy",
    "fiduciary",
    "settlement date",
    "t+3",
    "swift",
    "hud-1",
    "money",
    "investments",
    "investor",
    "investors",
    "cds",
    "retirement",
    "pension",
    "pensions",
    "annuity",
    "annuities",
    "covered call",
    "naked put",
    "bear market",
    "treasuries",
    "currencies",
    "banks",
    "electronic transfers",
    "accredited investor",
    "return on equity",
    "standard deviation",
    "geometric mean",
    "arithmetic mean",
    "safe havens",
    "illiquid charges",
    "etrade",
    "quicken",
    "capital losses",
    "gains",
    "deductions",
    "taxpayers",
    "pay in full",
    "buying a car",
    "home loans",
    "unsecured loans",
    "large purchase",
    "savings",
    "checking",
    "checkbook",
    "payment",
    "payments",
    "bills",
    "balance",
    "maintenance fee",
    "salary",
    "compensated",
    "medical bills",
    "liability",
    "cash and cards",
]


COMPANY_FUNDAMENTAL_TERMS = [
    "营业收入",
    "收入",
    "营收",
    "净利润",
    "归母净利润",
    "扣非净利润",
    "利润率",
    "毛利率",
    "业绩",
    "业绩预告",
    "同比增长",
    "增长率",
    "每股收益",
    "归属于上市公司股东",
    "归属于母公司",
    "一季度",
    "第一季度",
    "1q",
    "1q24",
]


DISCLOSURE_TERMS = [
    "年度报告",
    "年报",
    "非经常性损益",
    "利润分配预案",
    "利润分配",
    "审计机构",
    "审计",
    "政府补助",
    "关联交易",
    "资源税",
    "行政处罚",
    "处罚决定",
    "保荐机构",
    "持续督导",
    "签字的保荐代表人",
    "有限售条件股份",
    "处置子公司",
    "资产减值",
    "所得税税率",
    "银行存款",
    "自有资金",
    "客户资金",
    "总资产",
    "母公司所有者权益",
    "盈余公积",
    "高新技术企业证书",
    "年度报告中",
]

STOCK_SCOPE_INTENTS = {
    "market_explanation",
    "hold_judgment",
    "buy_sell_timing",
    "fundamental_analysis",
    "valuation_analysis",
    "risk_analysis",
}

STOCK_SCOPE_TOPICS = {"price", "fundamentals", "valuation", "risk", "news"}

INDEX_SCOPE_INTENTS = {
    "market_explanation",
    "hold_judgment",
    "buy_sell_timing",
    "risk_analysis",
    "macro_policy_impact",
}

INDEX_SCOPE_TOPICS = {"price", "industry", "macro", "policy", "news", "risk"}

FUND_MECHANISM_TERMS = [
    "费率",
    "申赎",
    "定投",
    "机制",
    "区别",
    "不同",
    "LOF",
    "lof",
    "ETF",
    "etf",
    "指数基金",
    "场内",
    "场外",
]

MACRO_POLICY_TERMS = [
    "CPI",
    "cpi",
    "PMI",
    "pmi",
    "LPR",
    "lpr",
    "M2",
    "m2",
    "降息",
    "降准",
    "宏观",
    "政策",
    "央行",
    "汇率",
    "利率",
    "通胀",
]


def _sort_sources(sources: list[str]) -> list[str]:
    unique = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    return sorted(unique, key=lambda source: SOURCE_PRIORITY.index(source) if source in SOURCE_PRIORITY else 99)


def looks_like_general_finance_query(query: str) -> bool:
    lowered = query.lower()
    for term in GENERAL_FINANCE_TERMS:
        escaped = re.escape(term.lower())
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered):
            return True
    return False


def looks_like_company_fundamental_query(query: str) -> bool:
    lowered = query.lower()
    return any(term in query for term in COMPANY_FUNDAMENTAL_TERMS) or any(term in lowered for term in ["earnings", "revenue", "net profit"])


def looks_like_disclosure_query(query: str) -> bool:
    if any(term in query for term in DISCLOSURE_TERMS):
        return True
    return bool(("2023年" in query or "2024年" in query) and any(term in query for term in ["股份有限公司", "公司", "ST", "营业收入", "净利润", "所得税", "处罚", "总资产"]))


def _extend_missing(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


class RuleSourcePlanner:
    def plan(self, product_type: str, intents: list[dict], topics: list[dict], time_scope: str, query: str = "") -> dict:
        labels = {item["label"] for item in intents}
        topic_labels = {item["label"] for item in topics}
        bucket_market_query = contains_bucket_market_term(query)

        sources: list[str] = []
        required: list[str] = []

        if "market_explanation" in labels and product_type == "stock":
            if not bucket_market_query:
                sources.append("market_api")
            sources.extend(["news", "industry_sql", "fundamental_sql", "research_note"])
            if not bucket_market_query:
                sources.append("announcement")
            required.extend(["price", "news", "industry", "fundamentals", "risk"])
        if "event_news_query" in labels and product_type == "stock":
            sources.extend(["news", "announcement"])
            required.extend(["news"])
        if "hold_judgment" in labels and product_type == "stock":
            sources.extend(["market_api", "news", "fundamental_sql", "research_note"])
            if not bucket_market_query:
                sources.append("announcement")
            required.extend(["price", "fundamentals", "risk"])
        if "product_info" in labels and product_type in {"fund", "etf"}:
            sources.extend(["faq", "product_doc", "research_note"])
            required.extend(["product_mechanism"])
        if "buy_sell_timing" in labels and product_type in {"fund", "etf"}:
            sources.extend(["market_api", "product_doc", "research_note"])
            required.extend(["price", "risk"])
        if "risk_analysis" in labels and product_type in {"fund", "etf"}:
            sources.extend(["market_api", "faq", "product_doc"])
            required.extend(["risk"])
        if product_type == "stock":
            self._apply_stock_scope_fallback(labels, topic_labels, query, bucket_market_query, sources, required)
        if product_type in {"fund", "etf"}:
            self._apply_fund_scope_fallback(labels, topic_labels, query, sources, required)
        if product_type in {"index", "generic_market"} or bucket_market_query:
            self._apply_index_scope_fallback(labels, topic_labels, query, sources, required)
        if product_type in {"fund", "etf"} and not labels:
            sources.extend(["faq", "product_doc"])
            required.extend(["product_mechanism"])
        if "macro_policy_impact" in labels or "macro" in topic_labels:
            sources.extend(["macro_sql", "news", "research_note"])
            required.extend(["macro"])
            if "industry" in topic_labels or any(term in query for term in ["板块", "行业", "A股", "a股"]):
                sources.append("industry_sql")
                required.append("industry")
        if "fundamental_analysis" in labels or "valuation_analysis" in labels:
            sources.extend(["fundamental_sql", "research_note", "news"])
            required.extend(["fundamentals"])
        if "peer_compare" in labels:
            sources.extend(["market_api", "fundamental_sql", "research_note"])
            required.extend(["comparison"])

        if not sources:
            sources = ["news", "market_api"]
            required = ["news", "price"]

        if looks_like_general_finance_query(query):
            sources.insert(0, "research_note")
            if "risk" not in required and any(term in query.lower() for term in ["risk", "risky", "insurance", "debt", "loan", "mortgage"]):
                required.append("risk")
        if looks_like_company_fundamental_query(query):
            sources.extend(["research_note", "fundamental_sql"])
            if "fundamentals" not in required:
                required.append("fundamentals")
        if looks_like_disclosure_query(query):
            sources.insert(0, "announcement")
            if "fundamentals" not in required:
                required.append("fundamentals")

        unique_required = []
        for item in required:
            if item not in unique_required:
                unique_required.append(item)

        priority = _sort_sources(sources)
        return {
            "required_evidence_types": unique_required,
            "source_plan": priority,
            "source_priority": priority,
        }

    def _apply_stock_scope_fallback(
        self,
        labels: set[str],
        topic_labels: set[str],
        query: str,
        bucket_market_query: bool,
        sources: list[str],
        required: list[str],
    ) -> None:
        scope_like = bool(labels.intersection(STOCK_SCOPE_INTENTS))
        if "peer_compare" not in labels:
            scope_like = scope_like or bool(topic_labels.intersection(STOCK_SCOPE_TOPICS))
        broad_style = any(term in query for term in ["怎么样", "如何", "怎么看", "还能拿", "还能不能拿", "值得拿", "值得买", "值得入手", "入手", "适合买", "持有", "风险"])
        if not scope_like and not broad_style:
            return
        if bucket_market_query:
            _extend_missing(sources, ["news", "industry_sql", "research_note", "macro_sql"])
            _extend_missing(required, ["news", "industry", "risk"])
            return
        _extend_missing(sources, ["market_api", "industry_sql", "fundamental_sql", "news", "announcement", "research_note"])
        _extend_missing(required, ["price", "industry", "fundamentals", "risk", "news"])

    def _apply_fund_scope_fallback(
        self,
        labels: set[str],
        topic_labels: set[str],
        query: str,
        sources: list[str],
        required: list[str],
    ) -> None:
        mechanism_like = (
            "trading_rule_fee" in labels
            or "product_info" in labels
            or "product_mechanism" in topic_labels
            or any(term in query for term in FUND_MECHANISM_TERMS)
        )
        if not mechanism_like:
            return
        _extend_missing(sources, ["faq", "product_doc", "research_note"])
        if labels.intersection({"buy_sell_timing", "hold_judgment", "risk_analysis"}) or any(term in query for term in ["能买吗", "拿吗", "波动", "走势"]):
            _extend_missing(sources, ["market_api"])
            _extend_missing(required, ["price", "risk"])
        _extend_missing(required, ["product_mechanism"])

    def _apply_index_scope_fallback(
        self,
        labels: set[str],
        topic_labels: set[str],
        query: str,
        sources: list[str],
        required: list[str],
    ) -> None:
        scope_like = bool(labels.intersection(INDEX_SCOPE_INTENTS) or topic_labels.intersection(INDEX_SCOPE_TOPICS))
        if not scope_like:
            return
        _extend_missing(sources, ["market_api", "news", "research_note"])
        _extend_missing(required, ["price", "news"])
        if topic_labels.intersection({"industry", "macro", "policy", "risk"}) or any(term in query for term in ["板块", "行业", "大盘", "A股", "a股", "为什么", "为啥", "还能拿"]):
            _extend_missing(sources, ["industry_sql", "macro_sql"])
            _extend_missing(required, ["industry", "macro"])


@dataclass
class MLSourcePlanner:
    classifier: MultiLabelClassifier

    @classmethod
    def build_from_records(cls, records: list[dict]) -> "MLSourcePlanner":
        planner_records = []
        for record in records:
            source_labels = list(record.get("expected_document_sources", [])) + list(record.get("expected_structured_sources", []))
            planner_records.append({"query": record["query"], "source_labels": source_labels})
        return cls(classifier=MultiLabelClassifier.build_from_records(planner_records, "source_labels"))

    def predict_sources(self, query: str) -> list[str]:
        return [item["label"] for item in self.classifier.predict(query, threshold=0.18)]


class SourcePlanner:
    def __init__(self, ml_planner: MLSourcePlanner | None = None, source_plan_reranker: SourcePlanReranker | None = None) -> None:
        self.rule_planner = RuleSourcePlanner()
        self.ml_planner = ml_planner
        self.source_plan_reranker = source_plan_reranker

    def plan(
        self,
        query: str,
        product_type: str,
        intents: list[dict],
        topics: list[dict],
        time_scope: str,
        missing_slots: list[str] | None = None,
    ) -> dict:
        missing_slots = missing_slots or []
        bucket_market_query = contains_bucket_market_term(query)
        if "missing_entity" in missing_slots:
            return {
                "required_evidence_types": [],
                "source_plan": [],
                "source_priority": [],
            }

        rule_plan = self.rule_planner.plan(product_type, intents, topics, time_scope, query=query)
        if self._should_use_mechanism_whitelist(product_type, intents, topics):
            plan = self._mechanism_whitelist_plan(rule_plan)
        elif self.ml_planner is None:
            plan = rule_plan
        else:
            predicted_sources = self.ml_planner.predict_sources(query)
            if self._should_use_mechanism_whitelist(product_type, intents, topics):
                predicted_sources = [source for source in predicted_sources if source in {"faq", "product_doc", "research_note"}]
            merged_sources = _sort_sources(rule_plan["source_plan"] + predicted_sources)
            if bucket_market_query:
                merged_sources = [source for source in merged_sources if source != "announcement"]
            merged_required = list(rule_plan["required_evidence_types"])

            if any(source in {"faq", "product_doc"} for source in predicted_sources) and "product_mechanism" not in merged_required:
                merged_required.append("product_mechanism")
            labels = {item["label"] for item in intents}
            topic_labels = {item["label"] for item in topics}
            needs_risk = bool(labels.intersection({"hold_judgment", "risk_analysis", "fundamental_analysis", "valuation_analysis"})) or "risk" in topic_labels
            if any(source in {"research_note", "announcement"} for source in predicted_sources) and "risk" not in merged_required and product_type == "stock" and needs_risk:
                merged_required.append("risk")

            plan = {
                "required_evidence_types": merged_required,
                "source_plan": merged_sources,
                "source_priority": merged_sources,
            }

        if bucket_market_query:
            plan = self._apply_bucket_market_rules(plan, intents, topics)
        if looks_like_general_finance_query(query):
            plan = self._apply_general_finance_rules(plan)
        if looks_like_company_fundamental_query(query) or looks_like_disclosure_query(query):
            plan = self._apply_company_report_rules(plan, query)
        plan = self._apply_time_scope_rules(plan, time_scope)
        plan = self._apply_source_plan_reranker(query, product_type, intents, topics, time_scope, plan)
        return self._apply_scope_whitelists(query, product_type, intents, topics, plan)

    def _apply_general_finance_rules(self, plan: dict) -> dict:
        sources = list(plan["source_plan"])
        if "research_note" not in sources:
            sources.insert(0, "research_note")
        return {
            "required_evidence_types": list(plan["required_evidence_types"]),
            "source_plan": _sort_sources(sources),
            "source_priority": _sort_sources(sources),
        }

    def _apply_company_report_rules(self, plan: dict, query: str) -> dict:
        sources = list(plan["source_plan"])
        if looks_like_disclosure_query(query) and "announcement" not in sources:
            sources.insert(0, "announcement")
        if looks_like_company_fundamental_query(query) and "research_note" not in sources:
            sources.append("research_note")
        required = list(plan["required_evidence_types"])
        if "fundamentals" not in required:
            required.append("fundamentals")
        return {
            "required_evidence_types": required,
            "source_plan": _sort_sources(sources),
            "source_priority": _sort_sources(sources),
        }

    def _should_use_mechanism_whitelist(self, product_type: str, intents: list[dict], topics: list[dict]) -> bool:
        labels = {item["label"] for item in intents}
        topic_labels = {item["label"] for item in topics}
        mechanism_only_topics = topic_labels.issubset({"product_mechanism", "comparison"})
        blocking_labels = {
            "market_explanation",
            "hold_judgment",
            "buy_sell_timing",
            "risk_analysis",
            "macro_policy_impact",
            "fundamental_analysis",
            "valuation_analysis",
        }
        if product_type in {"fund", "etf"} and "trading_rule_fee" in labels:
            return True
        return (
            product_type in {"fund", "etf"}
            and mechanism_only_topics
            and not labels.intersection(blocking_labels)
            and labels.intersection({"product_info", "peer_compare"})
        )

    def _mechanism_whitelist_plan(self, current_plan: dict) -> dict:
        whitelisted_sources = [source for source in current_plan["source_plan"] if source in {"faq", "product_doc", "research_note"}]
        for source in ["faq", "product_doc", "research_note"]:
            if source not in whitelisted_sources:
                whitelisted_sources.append(source)
        whitelisted = _sort_sources(whitelisted_sources)
        required = list(current_plan["required_evidence_types"])
        if "product_mechanism" not in required:
            required.append("product_mechanism")
        return {
            "required_evidence_types": required,
            "source_plan": whitelisted,
            "source_priority": whitelisted,
        }

    def _apply_time_scope_rules(self, plan: dict, time_scope: str) -> dict:
        if time_scope != "long_term":
            return plan

        filtered_sources = [source for source in plan["source_plan"] if source != "announcement"]
        return {
            "required_evidence_types": list(plan["required_evidence_types"]),
            "source_plan": filtered_sources,
            "source_priority": filtered_sources,
        }

    def _apply_bucket_market_rules(self, plan: dict, intents: list[dict], topics: list[dict]) -> dict:
        labels = {item["label"] for item in intents}
        topic_labels = {item["label"] for item in topics}
        allowed_sources = {"news", "industry_sql", "research_note", "macro_sql"}
        filtered_sources = [source for source in plan["source_plan"] if source in allowed_sources]
        if "news" not in filtered_sources:
            filtered_sources.append("news")
        if "industry_sql" not in filtered_sources:
            filtered_sources.append("industry_sql")
        if (
            topic_labels.intersection({"risk", "fundamentals", "price", "macro"})
            or labels.intersection({"hold_judgment", "buy_sell_timing", "market_explanation", "fundamental_analysis", "valuation_analysis"})
        ) and "macro_sql" not in filtered_sources:
            filtered_sources.append("macro_sql")
        if "research_note" not in filtered_sources:
            filtered_sources.append("research_note")

        required = [item for item in plan["required_evidence_types"] if item in {"news", "industry", "risk", "macro", "fundamentals"}]
        for item in ["news", "industry"]:
            if item not in required:
                required.append(item)
        if labels.intersection({"hold_judgment", "buy_sell_timing", "fundamental_analysis", "valuation_analysis"}) or topic_labels.intersection({"risk", "fundamentals"}):
            for item in ["risk", "macro"]:
                if item not in required:
                    required.append(item)

        ordered = _sort_sources(filtered_sources)
        return {
            "required_evidence_types": required,
            "source_plan": ordered,
            "source_priority": ordered,
        }

    def _apply_source_plan_reranker(
        self,
        query: str,
        product_type: str,
        intents: list[dict],
        topics: list[dict],
        time_scope: str,
        plan: dict,
    ) -> dict:
        if self.source_plan_reranker is None:
            return plan

        intent_labels = [item["label"] for item in intents]
        topic_labels = [item["label"] for item in topics]
        scored = self.source_plan_reranker.score_candidates(
            query=query,
            product_type=product_type,
            intent_labels=intent_labels,
            topic_labels=topic_labels,
            time_scope=time_scope,
            rule_plan=plan["source_plan"],
            candidate_sources=list(SOURCE_PRIORITY),
        )
        selected_sources = [
            item["source"]
            for item in scored
            if item["score"] >= 0.42 or item["source"] in plan["source_plan"][:2]
        ]
        if not selected_sources:
            selected_sources = list(plan["source_plan"])

        if self._should_use_mechanism_whitelist(product_type, intents, topics):
            mechanism_sources = [source for source in selected_sources if source in {"faq", "product_doc", "research_note"}]
            for source in ["faq", "product_doc", "research_note"]:
                if source not in mechanism_sources:
                    mechanism_sources.append(source)
            selected_sources = mechanism_sources

        if "announcement" in selected_sources and "news" in selected_sources:
            if any(label == "event_news_query" for label in intent_labels) and "公告" in query:
                selected_sources.remove("announcement")
                selected_sources.insert(0, "announcement")
            elif "news" in selected_sources:
                selected_sources.remove("news")
                selected_sources.insert(0, "news")
        elif "news" in selected_sources and "research_note" in selected_sources:
            selected_sources.remove("news")
            selected_sources.insert(0, "news")
        if any(label == "event_news_query" for label in intent_labels) and "公告" in query:
            selected_sources = [source for source in selected_sources if source not in {"market_api", "fundamental_sql", "industry_sql"}]
        if any(label == "event_news_query" for label in intent_labels) and "公告" not in query:
            selected_sources = [source for source in selected_sources if source not in {"market_api", "fundamental_sql", "industry_sql"}]
        if any(label == "market_explanation" for label in intent_labels) and product_type == "stock" and "market_api" in plan["source_plan"] and "market_api" not in selected_sources:
            selected_sources.insert(0, "market_api")
        if looks_like_general_finance_query(query) and "research_note" not in selected_sources:
            selected_sources.insert(0, "research_note")
        if looks_like_company_fundamental_query(query) and "research_note" not in selected_sources:
            selected_sources.insert(0, "research_note")
        if looks_like_disclosure_query(query) and "announcement" not in selected_sources:
            selected_sources.insert(0, "announcement")
        if "fundamental_sql" in selected_sources and any(label in {"fundamental_analysis", "valuation_analysis"} for label in intent_labels):
            selected_sources.remove("fundamental_sql")
            selected_sources.insert(0, "fundamental_sql")
        if any(label in {"hold_judgment", "buy_sell_timing", "fundamental_analysis", "valuation_analysis"} for label in intent_labels) and "fundamental_sql" in plan["source_plan"] and "fundamental_sql" not in selected_sources:
            selected_sources.append("fundamental_sql")
        if product_type == "stock" and (
            set(intent_labels).intersection(STOCK_SCOPE_INTENTS)
            or set(topic_labels).intersection(STOCK_SCOPE_TOPICS)
            or any(term in query for term in ["怎么样", "如何", "怎么看", "还能拿", "还能不能拿", "值得买", "值得入手", "入手", "适合买", "风险"])
        ):
            for source in ["market_api", "news", "industry_sql", "fundamental_sql", "announcement", "research_note"]:
                if source in plan["source_plan"] and source not in selected_sources:
                    selected_sources.append(source)
        if product_type in {"index", "generic_market"} and (
            set(intent_labels).intersection(INDEX_SCOPE_INTENTS) or set(topic_labels).intersection(INDEX_SCOPE_TOPICS)
        ):
            for source in ["market_api", "news", "research_note", "industry_sql", "macro_sql"]:
                if source in plan["source_plan"] and source not in selected_sources:
                    selected_sources.append(source)
        if product_type == "stock" and "risk_analysis" in intent_labels and "faq" in selected_sources:
            selected_sources = [source for source in selected_sources if source != "faq"]
        if product_type == "stock" and (
            any(label in {"hold_judgment", "buy_sell_timing", "fundamental_analysis", "valuation_analysis", "risk_analysis"} for label in intent_labels)
            or set(topic_labels).intersection(STOCK_SCOPE_TOPICS)
            or any(term in query for term in ["怎么样", "如何", "怎么看", "值得买", "值得入手", "入手", "适合买", "持有", "风险"])
        ):
            selected_sources = [source for source in selected_sources if source not in {"faq", "product_doc"}]
        if (
            "macro_policy_impact" not in intent_labels
            and not {"macro", "policy"}.intersection(topic_labels)
            and "macro_sql" not in plan["source_plan"]
        ):
            selected_sources = [source for source in selected_sources if source != "macro_sql"]
        if contains_bucket_market_term(query) and not looks_like_disclosure_query(query):
            allowed_sources = {"news", "industry_sql", "research_note", "macro_sql"}
            selected_sources = [source for source in selected_sources if source in allowed_sources]

        deduped = []
        for source in selected_sources:
            if source not in deduped:
                deduped.append(source)
        required = self._align_required_evidence_with_sources(plan["required_evidence_types"], deduped)

        return {
            "required_evidence_types": required,
            "source_plan": deduped,
            "source_priority": deduped,
        }

    def _apply_scope_whitelists(
        self,
        query: str,
        product_type: str,
        intents: list[dict],
        topics: list[dict],
        plan: dict,
    ) -> dict:
        labels = {item["label"] for item in intents}
        topic_labels = {item["label"] for item in topics}
        if self._is_standalone_macro_policy_scope(query, product_type, labels, topic_labels):
            allowed = {"macro_sql", "news", "research_note", "industry_sql"}
            sources = [source for source in plan["source_plan"] if source in allowed]
            for source in ["macro_sql", "news", "research_note"]:
                if source not in sources:
                    sources.append(source)
            if ("industry" in topic_labels or any(term in query for term in ["板块", "行业", "A股", "a股", "哪些"])) and "industry_sql" not in sources:
                sources.append("industry_sql")
            required = ["macro", "news"]
            if "industry_sql" in sources:
                required.append("industry")
            ordered = _sort_sources(sources)
            return {
                "required_evidence_types": required,
                "source_plan": ordered,
                "source_priority": ordered,
            }

        if product_type in {"index", "generic_market"}:
            allowed = {"market_api", "news", "research_note", "industry_sql", "macro_sql"}
            sources = [source for source in plan["source_plan"] if source in allowed]
            for source in ["market_api", "news", "research_note"]:
                if source not in sources:
                    sources.append(source)
            if labels.intersection(INDEX_SCOPE_INTENTS) or topic_labels.intersection({"industry", "macro", "policy", "risk"}) or any(term in query for term in ["为什么", "为啥", "跌", "涨", "板块", "行业", "大盘"]):
                for source in ["industry_sql", "macro_sql"]:
                    if source not in sources:
                        sources.append(source)
            required = ["price", "news"]
            if "industry_sql" in sources:
                required.append("industry")
            if "macro_sql" in sources:
                required.append("macro")
            ordered = _sort_sources(sources)
            return {
                "required_evidence_types": required,
                "source_plan": ordered,
                "source_priority": ordered,
            }

        return plan

    @staticmethod
    def _is_standalone_macro_policy_scope(query: str, product_type: str, labels: set[str], topic_labels: set[str]) -> bool:
        if product_type in {"macro", "macro_indicator", "policy"}:
            return True
        if product_type in {"stock", "fund", "etf", "index"}:
            return False
        return (
            "macro_policy_impact" in labels
            or bool(topic_labels.intersection({"macro", "policy"}))
            or any(term in query for term in MACRO_POLICY_TERMS)
        )

    @staticmethod
    def _align_required_evidence_with_sources(required: list[str], sources: list[str]) -> list[str]:
        source_set = set(sources)
        filtered: list[str] = []
        for item in required:
            if item == "industry" and "industry_sql" not in source_set:
                continue
            if item == "macro" and "macro_sql" not in source_set:
                continue
            if item not in filtered:
                filtered.append(item)
        return filtered
