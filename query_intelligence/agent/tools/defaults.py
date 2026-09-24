from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ToolRegistry
from .context import ToolContext
from .documents import build_document_tools
from .entity import build_resolve_entity
from .fundamentals import build_fundamentals_tool
from .macro import build_macro_tool
from .market import build_market_tools
from .sentiment import build_sentiment_tool

if TYPE_CHECKING:
    from ...service import QueryIntelligenceService

DEFAULT_TOOL_NAMES = (
    "resolve_entity",
    "get_price_history",
    "compute_indicators",
    "get_fundamentals",
    "get_macro_indicators",
    "search_news",
    "search_announcements",
    "search_knowledge",
    "analyze_sentiment",
)


def build_default_registry(context: ToolContext) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_resolve_entity(context))
    for spec in build_market_tools(context):
        registry.register(spec)
    registry.register(build_fundamentals_tool(context))
    registry.register(build_macro_tool(context))
    for spec in build_document_tools(context):
        registry.register(spec)
    registry.register(build_sentiment_tool(context))
    return registry


def build_registry_for_service(service: QueryIntelligenceService) -> ToolRegistry:
    return build_default_registry(ToolContext.from_service(service))
