from __future__ import annotations


class APIRetriever:
    def __init__(self, structured_data: dict) -> None:
        self.structured_data = structured_data

    def fetch(self, query_bundle: dict) -> list[dict]:
        results = []
        source_plan = set(query_bundle.get("source_plan", []))
        for symbol in query_bundle.get("symbols", []):
            if "market_api" in source_plan and symbol in self.structured_data["market_api"]:
                results.append(
                    {
                        "evidence_id": f"price_{symbol}",
                        "source_type": "market_api",
                        "payload": self.structured_data["market_api"][symbol],
                    }
                )
        return results
