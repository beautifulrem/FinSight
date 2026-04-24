from __future__ import annotations


MACRO_INDICATOR_TERMS = {
    "CPI": ["cpi", "通胀", "物价"],
    "PMI": ["pmi", "制造业", "景气"],
    "M2": ["m2", "货币", "流动性", "社融", "降准", "降息", "lpr"],
    "CN10Y": ["10y", "十年期", "国债", "利率", "收益率", "降息", "lpr", "汇率"],
}


class SQLRetriever:
    def __init__(self, structured_data: dict) -> None:
        self.structured_data = structured_data

    def fetch(self, query_bundle: dict) -> list[dict]:
        results = []
        symbols = query_bundle.get("symbols", [])
        source_plan = set(query_bundle.get("source_plan", []))

        for symbol in symbols:
            if "fundamental_sql" in source_plan and symbol in self.structured_data["fundamental_sql"]:
                results.append(
                    {
                        "evidence_id": f"fundamental_{symbol}",
                        "source_type": "fundamental_sql",
                        "payload": self.structured_data["fundamental_sql"][symbol],
                    }
                )

        entity_names = query_bundle.get("entity_names", [])
        if "industry_sql" in source_plan:
            seen_industries = set()
            for entity_name in entity_names:
                industry_name = self.structured_data["entity_to_industry"].get(entity_name)
                if industry_name and industry_name in self.structured_data["industry_sql"] and industry_name not in seen_industries:
                    seen_industries.add(industry_name)
                    results.append(
                        {
                            "evidence_id": f"industry_{industry_name}",
                            "source_type": "industry_sql",
                            "payload": self.structured_data["industry_sql"][industry_name],
                        }
                    )
            for industry_name in query_bundle.get("industry_terms", []):
                if industry_name in self.structured_data["industry_sql"] and industry_name not in seen_industries:
                    seen_industries.add(industry_name)
                    results.append(
                        {
                            "evidence_id": f"industry_{industry_name}",
                            "source_type": "industry_sql",
                            "payload": self.structured_data["industry_sql"][industry_name],
                        }
                    )
            if not seen_industries and self._asks_for_industry_scope(query_bundle):
                for industry_name, payload in self.structured_data["industry_sql"].items():
                    results.append(
                        {
                            "evidence_id": f"industry_{industry_name}",
                            "source_type": "industry_sql",
                            "payload": payload,
                        }
                    )

        if "macro_sql" in source_plan:
            for indicator in self._filter_macro_indicators(query_bundle):
                results.append(
                    {
                        "evidence_id": f"macro_{indicator['indicator_code']}",
                        "source_type": "macro_sql",
                        "payload": indicator,
                    }
                )
        return results

    def _filter_macro_indicators(self, query_bundle: dict) -> list[dict]:
        indicators = list(self.structured_data["macro_sql"].values())
        query_text = self._query_text(query_bundle)
        matched = []
        seen_codes = set()
        for indicator in indicators:
            code = self._macro_code_family(indicator)
            aliases = MACRO_INDICATOR_TERMS.get(code, [])
            if any(alias in query_text for alias in aliases):
                matched.append(indicator)
                seen_codes.add(code)
        if "降准" in query_text or "降息" in query_text or "lpr" in query_text:
            for indicator in indicators:
                code = self._macro_code_family(indicator)
                if code in {"M2", "CN10Y"} and code not in seen_codes:
                    matched.append(indicator)
                    seen_codes.add(code)
        return matched or indicators

    def _macro_code_family(self, indicator: dict) -> str:
        code = str(indicator.get("indicator_code", "")).upper()
        if code.endswith("_CN"):
            code = code.removesuffix("_CN")
        if "10Y" in code:
            return "CN10Y"
        return code

    def _asks_for_industry_scope(self, query_bundle: dict) -> bool:
        query_text = self._query_text(query_bundle)
        topic_labels = set(query_bundle.get("topic_labels", []))
        return "industry" in topic_labels or any(term in query_text for term in ["哪些板块", "哪个板块", "板块", "行业"])

    def _query_text(self, query_bundle: dict) -> str:
        parts = [
            query_bundle.get("normalized_query", ""),
            *query_bundle.get("keywords", []),
            *query_bundle.get("industry_terms", []),
        ]
        return " ".join(str(part) for part in parts).lower()
