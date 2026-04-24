from __future__ import annotations

from query_intelligence.retrieval.pipeline import RetrievalPipeline


class _FakeMacroProvider:
    def fetch_indicators(self, query_bundle: dict) -> list[dict]:
        query = query_bundle.get("normalized_query", "").lower()
        if "cpi" in query:
            return [
                {
                    "evidence_id": "macro_live_CPI_CN",
                    "source_type": "macro_indicator",
                    "source_name": "akshare",
                    "provider": "akshare",
                    "payload": {
                        "indicator_code": "CPI_CN",
                        "indicator_name": "中国CPI同比",
                        "metric_date": "2026-03-31",
                        "metric_value": 0.9,
                        "unit": "%",
                        "source_name": "akshare",
                        "provider": "akshare",
                        "release_period": "2026-03",
                        "frequency": "monthly",
                    },
                }
            ]
        if "降准" in query:
            return [
                {
                    "evidence_id": "macro_live_M2_CN",
                    "source_type": "macro_indicator",
                    "source_name": "akshare",
                    "provider": "akshare",
                    "payload": {
                        "indicator_code": "M2_CN",
                        "indicator_name": "中国M2同比",
                        "metric_date": "2026-03-31",
                        "metric_value": 8.3,
                        "unit": "%",
                        "source_name": "akshare",
                        "provider": "akshare",
                        "release_period": "2026-03",
                        "frequency": "monthly",
                    },
                }
            ]
        return []


def test_live_macro_provider_returns_only_matching_cpi_and_replaces_seed_macro_sql() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.macro_provider = _FakeMacroProvider()

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "CPI 对市场有什么影响",
            "keywords": ["CPI"],
            "entity_names": ["CPI"],
            "symbols": [],
            "source_plan": ["macro_sql"],
        }
    )

    macro_items = [item for item in structured_items if item["source_type"] in {"macro_sql", "macro_indicator"}]
    assert [item["source_type"] for item in macro_items] == ["macro_indicator"]
    assert macro_items[0]["payload"]["indicator_code"] == "CPI_CN"
    assert macro_items[0]["payload"]["metric_value"] == 0.9
    assert macro_items[0]["payload"]["source_name"] == "akshare"


def test_live_macro_provider_maps_reserve_requirement_cut_to_m2_or_policy_signal() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.macro_provider = _FakeMacroProvider()

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "央行降准对A股有什么影响",
            "keywords": ["降准"],
            "entity_names": [],
            "symbols": [],
            "source_plan": ["macro_sql"],
        }
    )

    macro_items = [item for item in structured_items if item["source_type"] in {"macro_indicator", "policy_event"}]
    assert {item["payload"]["indicator_code"] for item in macro_items} >= {"M2_CN"}
    assert all(item["source_type"] != "macro_sql" or item["payload"]["indicator_code"] != "M2_CN" for item in structured_items)


def test_seed_macro_sql_remains_available_without_live_macro_provider() -> None:
    pipeline = RetrievalPipeline.build_demo()

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "CPI 对市场有什么影响",
            "keywords": ["CPI"],
            "entity_names": ["CPI"],
            "symbols": [],
            "source_plan": ["macro_sql"],
        }
    )

    assert [item["source_type"] for item in structured_items] == ["macro_sql"]
    assert structured_items[0]["payload"]["indicator_code"] == "CPI_CN"
    assert structured_items[0]["payload"]["source_name"] == "seed"


def test_akshare_macro_provider_tolerates_missing_optional_methods() -> None:
    from query_intelligence.integrations.akshare_macro_provider import AKShareMacroProvider

    class EmptyAkshareModule:
        pass

    provider = AKShareMacroProvider(ak_module=EmptyAkshareModule())

    assert provider.fetch_indicators({"normalized_query": "CPI PMI 降息", "keywords": [], "entity_names": []}) == []
