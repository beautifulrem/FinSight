"""Browser page rendering; CSS and JS are served from ``/static``."""

from __future__ import annotations

import html
from typing import Any

from .config import DEFAULT_CHATBOT_CONFIG, ROOT

STATIC_DIR = ROOT / "query_intelligence" / "web" / "static"


def render_index_html(config: dict[str, Any]) -> str:
    """Render the browser page from ``web/static/index.html`` (CSS/JS are served from ``/static``)."""
    ui = config.get("ui") or {}
    values = {
        "title": str(ui.get("title") or DEFAULT_CHATBOT_CONFIG["ui"]["title"]),
        "placeholder": str(ui.get("input_placeholder") or DEFAULT_CHATBOT_CONFIG["ui"]["input_placeholder"]),
        "submit_text": str(ui.get("submit_text") or DEFAULT_CHATBOT_CONFIG["ui"]["submit_text"]),
    }
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", html.escape(value))
    return page
