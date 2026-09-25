"""Browser page rendering.

The React app is built from ``frontend/`` into ``web/dist`` (committed, so no Node is needed at run
time) and its assets are served from ``/static/app``. When the build is missing, or
``QI_WEB_UI=legacy`` is set, the original single-file page in ``web/static`` is served instead.
"""

from __future__ import annotations

import html
import os
from typing import Any

from .config import DEFAULT_CHATBOT_CONFIG, ROOT

STATIC_DIR = ROOT / "query_intelligence" / "web" / "static"
DIST_DIR = ROOT / "query_intelligence" / "web" / "dist"


def frontend_dist_available() -> bool:
    """True when the built React app exists and the legacy page was not requested."""
    if os.getenv("QI_WEB_UI", "").strip().lower() == "legacy":
        return False
    return (DIST_DIR / "index.html").is_file()


def render_index_html(config: dict[str, Any]) -> str:
    """Render ``/``: the built React app when available, otherwise the legacy static page."""
    ui = config.get("ui") or {}
    defaults = DEFAULT_CHATBOT_CONFIG["ui"]
    values = {
        "title": str(ui.get("title") or defaults["title"]),
        "placeholder": str(ui.get("input_placeholder") or defaults["input_placeholder"]),
        "submit_text": str(ui.get("submit_text") or defaults["submit_text"]),
    }
    if frontend_dist_available():
        # The React app is bilingual; it only uses placeholder/button text that was customized.
        for key, default_key in (("placeholder", "input_placeholder"), ("submit_text", "submit_text")):
            if values[key] == defaults[default_key]:
                values[key] = ""
        template = DIST_DIR / "index.html"
    else:
        template = STATIC_DIR / "index.html"
    page = template.read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", html.escape(value))
    return page
