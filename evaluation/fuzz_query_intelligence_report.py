from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "evaluation" / "fuzz_query_intelligence_report.json"


def build_fuzz_report(report_path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return deepcopy(payload)
