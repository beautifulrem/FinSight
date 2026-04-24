from __future__ import annotations

from functools import lru_cache

from .config import Settings
from .contracts import NLUResult, RetrievalResult
from .data_loader import clear_data_caches
from .nlu.pipeline import NLUPipeline
from .retrieval.pipeline import RetrievalPipeline


class QueryIntelligenceService:
    def __init__(self, nlu_pipeline: NLUPipeline, retrieval_pipeline: RetrievalPipeline) -> None:
        self.nlu_pipeline = nlu_pipeline
        self.retrieval_pipeline = retrieval_pipeline

    def analyze_query(
        self,
        query: str,
        user_profile: dict | None = None,
        dialog_context: list | None = None,
        debug: bool = False,
    ) -> dict:
        result = self.nlu_pipeline.run(
            query=query,
            user_profile=user_profile or {},
            dialog_context=dialog_context or [],
            debug=debug,
        )
        return NLUResult.model_validate(result).model_dump(mode="json")

    def retrieve_evidence(self, nlu_result: dict, top_k: int = 20, debug: bool = False) -> dict:
        result = self.retrieval_pipeline.run(nlu_result=nlu_result, top_k=top_k, debug=debug)
        return RetrievalResult.model_validate(result).model_dump(mode="json")

    def run_pipeline(
        self,
        query: str,
        user_profile: dict | None = None,
        dialog_context: list | None = None,
        top_k: int = 20,
        debug: bool = False,
    ) -> dict:
        nlu_result = self.analyze_query(query, user_profile=user_profile, dialog_context=dialog_context, debug=debug)
        retrieval_result = self.retrieve_evidence(nlu_result, top_k=top_k, debug=debug)
        return {"nlu_result": nlu_result, "retrieval_result": retrieval_result}


@lru_cache(maxsize=1)
def build_demo_service() -> QueryIntelligenceService:
    nlu_pipeline = NLUPipeline.build_demo()
    retrieval_pipeline = RetrievalPipeline.build_demo()
    return QueryIntelligenceService(nlu_pipeline=nlu_pipeline, retrieval_pipeline=retrieval_pipeline)


@lru_cache(maxsize=1)
def build_default_service() -> QueryIntelligenceService:
    settings = Settings.from_env()
    nlu_pipeline = NLUPipeline.build_default(settings)
    retrieval_pipeline = RetrievalPipeline.build_default(settings)
    return QueryIntelligenceService(nlu_pipeline=nlu_pipeline, retrieval_pipeline=retrieval_pipeline)


def clear_service_caches() -> None:
    clear_data_caches()
    build_demo_service.cache_clear()
    build_default_service.cache_clear()
