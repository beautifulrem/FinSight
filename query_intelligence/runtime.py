from __future__ import annotations

from pathlib import Path

from .config import Settings
from .contracts import NLUResult, RetrievalResult
from .integrations.akshare_provider import AKShareNewsProvider
from .integrations.cninfo_provider import CninfoAnnouncementProvider
from .integrations.tushare_provider import TushareMarketProvider
from .nlu.pipeline import NLUPipeline
from .repositories.postgres_repository import PostgresDocumentRepository, PostgresStructuredRepository
from .retrieval.pipeline import RetrievalPipeline
from .service import QueryIntelligenceService


def build_service(settings: Settings | None = None) -> QueryIntelligenceService:
    runtime_settings = settings or Settings.from_env()
    nlu_pipeline = NLUPipeline.build_default(runtime_settings)
    retrieval_pipeline = RetrievalPipeline.build_default(runtime_settings)
    return QueryIntelligenceService(nlu_pipeline=nlu_pipeline, retrieval_pipeline=retrieval_pipeline)


def model_path(settings: Settings, filename: str) -> Path:
    return Path(settings.models_dir) / filename
