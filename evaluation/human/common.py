"""Shared helpers for the human-input kit: CSV I/O (UTF-8 with BOM), Wilson intervals, Cohen's kappa,
evidence formatting and run provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent_eval.runner import ROOT, _command, _display_path, _git_commit

HUMAN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "evaluation" / "results"
KAPPA_RESAMPLES = 2000
KAPPA_SEED = 20260930


# ------------------------------------------------------------------------------------------------ CSV
def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Rows of a CSV saved by Excel, Numbers or WPS: BOM stripped, keys and values trimmed.

    Excel on a Chinese Windows may save as GBK; that is tried when the file is not valid UTF-8.
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    reader = csv.DictReader(text.splitlines())
    rows = []
    for row in reader:
        clean = {str(key).strip(): (value or "").strip() for key, value in row.items() if key is not None}
        if any(clean.values()):
            rows.append(clean)
    return rows


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    """UTF-8 with BOM so Excel and Numbers open Chinese text correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_binary(value: str) -> int | None:
    """``1``/``0`` (also 是/否, y/n, true/false, ✓/✗); blank is ``None``; anything else raises ``ValueError``."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "1.0", "y", "yes", "true", "是", "对", "✓", "√"}:
        return 1
    if text in {"0", "0.0", "n", "no", "false", "否", "错", "✗", "×"}:
        return 0
    raise ValueError(f"expected 1 or 0, got {value!r}")


# ------------------------------------------------------------------------------------------ statistics
def wilson_ci(successes: int, n: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score 95% interval for a proportion."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(successes: int, n: int) -> dict[str, Any]:
    return {
        "n": n,
        "successes": successes,
        "rate": round(successes / n, 4) if n else None,
        "ci95_wilson": wilson_ci(successes, n),
    }


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """Cohen's kappa for two raters over the same items (any hashable labels). ``None`` when undefined."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    labels = sorted(set(a) | set(b), key=str)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum((list(a).count(label) / n) * (list(b).count(label) / n) for label in labels)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else None
    return round((observed - expected) / (1 - expected), 4)


def kappa_ci(a: Sequence[Any], b: Sequence[Any]) -> list[float] | None:
    """Percentile bootstrap 95% CI of Cohen's kappa over items (fixed seed); undefined resamples skipped."""
    if len(a) < 2:
        return None
    rng = random.Random(KAPPA_SEED)
    pairs = list(zip(a, b, strict=True))
    values = []
    for _ in range(KAPPA_RESAMPLES):
        sample = rng.choices(pairs, k=len(pairs))
        value = cohen_kappa([x for x, _ in sample], [y for _, y in sample])
        if value is not None:
            values.append(value)
    if len(values) < KAPPA_RESAMPLES // 2:
        return None
    values.sort()
    low = values[max(0, math.floor(0.025 * len(values)))]
    high = values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)]
    return [round(low, 4), round(high, 4)]


def agreement(a: Sequence[Any], b: Sequence[Any], labels: Sequence[Any], names: tuple[str, str]) -> dict[str, Any]:
    """Observed agreement, Cohen's kappa (with bootstrap CI) and the confusion matrix ``names[0]`` x ``names[1]``."""
    n = len(a)
    matrix = {str(x): {str(y): 0 for y in labels} for x in labels}
    for x, y in zip(a, b, strict=True):
        matrix.setdefault(str(x), {str(label): 0 for label in labels}).setdefault(str(y), 0)
        matrix[str(x)][str(y)] += 1
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return {
        "n": n,
        "observed_agreement": round(agree / n, 4) if n else None,
        "observed_agreement_ci95_wilson": wilson_ci(agree, n),
        "cohen_kappa": cohen_kappa(a, b),
        "cohen_kappa_ci95_bootstrap": kappa_ci(a, b),
        "confusion": {"rows": names[0], "columns": names[1], "matrix": matrix},
    }


# ------------------------------------------------------------------------------------------- evidence
_SKIP_PAYLOAD = {"provenance", "units_source", "recent_closes", "metric_units", "symbol", "name", "product_type"}


def _payload_summary(payload: Any, limit: int = 8) -> str:
    if not isinstance(payload, dict):
        return ""
    parts = []
    for key, value in payload.items():
        if key in _SKIP_PAYLOAD or isinstance(value, dict | list) or value in (None, ""):
            continue
        if isinstance(value, int | float) or key in {"as_of", "report_date", "period", "unit"}:
            parts.append(f"{key}={value}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def describe_evidence(item: dict[str, Any]) -> str:
    """One line per evidence item: id, title, source, as-of date, URL and the key values it carries."""
    payload = item.get("payload") or {}
    provenance = (payload.get("provenance") if isinstance(payload, dict) else None) or {}
    names = [item.get("source_name") or provenance.get("original_source"), item.get("source_type")]
    names.append(provenance.get("source_label"))
    source = " / ".join(dict.fromkeys(str(name) for name in names if name))
    parts = [f"[{item.get('evidence_id')}] {item.get('title') or ''}".strip()]
    if source:
        parts.append(f"来源: {source}")
    if item.get("as_of"):
        parts.append(f"截至: {item['as_of']}")
    if item.get("source_url"):
        parts.append(str(item["source_url"]))
    summary = _payload_summary(payload)
    if summary:
        parts.append(f"数值: {summary}")
    else:
        body = payload if isinstance(payload, dict) else {}
        text = str(item.get("text_excerpt") or body.get("summary") or body.get("snippet") or body.get("text") or "")
        if text.strip():
            parts.append(f"摘要: {' '.join(text.split())[:200]}")
    return " | ".join(parts)


def cited_evidence(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence items the answer cites, in citation order."""
    sources = {item.get("evidence_id"): item for item in response.get("evidence_sources") or []}
    return [sources[eid] for eid in response.get("evidence_used") or [] if eid in sources]


def answer_text(response: dict[str, Any]) -> str:
    """The text a user sees: answer (or clarification question), limitations and the risk disclaimer."""
    if response.get("status") == "needs_clarification":
        clarification = response.get("clarification") or {}
        text = f"【澄清问题】{clarification.get('question') or ''}"
        options = clarification.get("options") or []
        if options:
            text += "\n选项: " + " / ".join(
                str(option.get("label", option) if isinstance(option, dict) else option) for option in options
            )
        return text
    parts = [str(response.get("answer") or "").strip()]
    limitations = [str(item) for item in response.get("limitations") or [] if str(item).strip()]
    if limitations:
        parts.append("局限: " + "；".join(limitations))
    if str(response.get("risk_disclaimer") or "").strip():
        parts.append(f"风险提示: {response['risk_disclaimer']}")
    return "\n".join(part for part in parts if part)


# ----------------------------------------------------------------------------------------- provenance
def run_config(module: str, argv: list[str] | None, **extra: Any) -> dict[str, Any]:
    return {
        "commit": _git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": _command(module, argv),
        **extra,
    }


def display(path: str | Path) -> str:
    return _display_path(path)


def write_result(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    return path
