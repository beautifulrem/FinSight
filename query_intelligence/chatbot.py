"""Compatibility facade for the /chat chatbot.

The implementation lives in ``query_intelligence.chat`` (config, language, llm_client, answer, page);
this module re-exports every name that used to be defined here.
"""

from __future__ import annotations

from .chat.answer import (
    _asks_for_current_market_data,
    _first_market_item,
    _is_known_non_trading_day,
    _non_trading_day_market_answer,
    _prepend_unique,
    _source_display_item,
    _template_key_points,
    apply_market_freshness_guard,
    build_chatbot_response,
    build_evidence_sources,
    template_answer,
)
from .chat.config import (
    DEFAULT_CHATBOT_CONFIG,
    DEFAULT_CONFIG_PATH,
    ROOT,
    _apply_env_overrides,
    _coerce_config_types,
    _deep_merge,
    _load_dotenv,
    _parse_bool,
    apply_live_data_env,
    load_chatbot_config,
)
from .chat.language import (
    DEFAULT_RISK_DISCLAIMER,
    DEFAULT_RISK_DISCLAIMER_EN,
    DEFAULT_RISK_DISCLAIMER_ZH,
    _default_risk_disclaimer,
    _language_signal,
    _target_language_name,
    _text_matches_language,
    answer_matches_language,
    detect_query_language,
)
from .chat.llm_client import (
    DeepSeekClient,
    DeepSeekError,
    _compact_document,
    _compact_structured_item,
    _evidence_ids,
    _parse_json_object,
    compact_evidence_payload,
    make_answer_language_repair_messages,
    make_answer_messages,
    normalize_llm_answer,
)
from .chat.page import (
    STATIC_DIR,
    render_index_html,
)

__all__ = [
    "DEFAULT_CHATBOT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RISK_DISCLAIMER",
    "DEFAULT_RISK_DISCLAIMER_EN",
    "DEFAULT_RISK_DISCLAIMER_ZH",
    "ROOT",
    "STATIC_DIR",
    "DeepSeekClient",
    "DeepSeekError",
    "_apply_env_overrides",
    "_asks_for_current_market_data",
    "_coerce_config_types",
    "_compact_document",
    "_compact_structured_item",
    "_deep_merge",
    "_default_risk_disclaimer",
    "_evidence_ids",
    "_first_market_item",
    "_is_known_non_trading_day",
    "_language_signal",
    "_load_dotenv",
    "_non_trading_day_market_answer",
    "_parse_bool",
    "_parse_json_object",
    "_prepend_unique",
    "_source_display_item",
    "_target_language_name",
    "_template_key_points",
    "_text_matches_language",
    "answer_matches_language",
    "apply_live_data_env",
    "apply_market_freshness_guard",
    "build_chatbot_response",
    "build_evidence_sources",
    "compact_evidence_payload",
    "detect_query_language",
    "load_chatbot_config",
    "make_answer_language_repair_messages",
    "make_answer_messages",
    "normalize_llm_answer",
    "render_index_html",
    "template_answer",
]
