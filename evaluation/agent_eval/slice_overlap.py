"""Overlap of a held-out slice with the project's own dev tasks, router labels, claim rows and test literals.

Round-9 review I12: three turns of the round-8 slice had exact or near copies in round-12 tests and dev tasks (written
before the slice existed, see ``evaluation/heldout_r8/README.md``), and the near-copy tests in
``tests/test_agent_eval.py`` only compared new dev tasks with the slices that were already committed. This module
measures the overlap from the slice side, for any slice file or folder, at the time of a run:

* ``exact``: normalised equality (case, whitespace and punctuation ignored) with an own text;
* ``near``: character 3-gram Jaccard >= 0.8 (the threshold of the dev-task tests) or ``difflib`` ratio >= 0.85
  between normalised texts (the reviewer's measure), for slice texts of at least ``MIN_NEAR_CHARS`` normalised
  characters, so "Moutai?" or "茅台呢" do not count as near copies of everything.

Own texts: every question / claim / document text of EVERY task file under ``evaluation/agent_eval/tasks/`` (the dev
tasks, the router labels, the ext and shipped gate sets, the independent and test sets), of every claim file under
``evaluation/claim_bench/``, and every string literal in ``tests/*.py``. Round-10 review J9: the scan used a fixed
list (dev tasks, router labels, dev claims) and missed ``agent_eval_ext_v1.jsonl``, which holds a one-word variant of
a round-9 slice task; a new task file is now scanned without a code change. The report names the slice task ids and
the own locations (``tests/test_x.py:255``, ``dev:<task id>``, ``tasks/agent_eval_ext_v1.jsonl:<task id>``), never
the slice text, so it can be run on a slice the engineers must not open.

Content level (round-11 review K16): a slice item can repeat an own text's *content* in other words, e.g. the same
company and the same threshold number, or the same pair of companies with the same operation, which neither text
measure sees. ``content_report`` extracts an explainable signature from every text with fixed lexicons (no model):
the companies of the offline snapshot it names (Chinese names, short names and the English aliases of
``data/synonym_dict.json``), metric words, operation words (sum, share, difference, multiple, extremes, holding, cost
basis, forecast, foreign line, 52-week) and numbers (with 万 / 亿 / % and simple Chinese numerals; years, dates and
"52-week" are not numbers). Two tiers, both reported:

* ``content``: the texts share a company **and** (a number or an operation) **and** their ``difflib`` ratio is
  >= ``CONTENT_RATIO`` (0.70, the reviewer's measure) — a paraphrase of the same question;
* ``content_signature``: the texts share a (company, metric, number) triple, or name the same two or more companies
  with a shared operation and metric, or the same one company with a shared holding / cost-basis / foreign-line /
  52-week operation — the same content whatever the wording (an upper bound: it also counts items written
  independently for the same bug class).

``--own-at <commit>`` reads the own texts (tests, task and claim files) from that commit with ``git archive``, so a
scan for a run at an earlier commit sees exactly that commit's wording; the report records ``own_texts_commit`` next
to the scanner's ``commit``.

``runner`` attaches the summary to the run's ``config.slice_overlap`` when the task file lives in a ``heldout*``
folder; it can also be run on its own:

    python -m evaluation.agent_eval.slice_overlap evaluation/heldout_r8
    python -m evaluation.agent_eval.slice_overlap /path/to/heldout_r9/chat_r9_heldout.jsonl --out outputs/x.json
    python -m evaluation.agent_eval.slice_overlap evaluation/heldout_r10 --content --own-at 3480a96 \
        --chat-runs chat_heldout_r10-auto-nollm-prefix chat_heldout_r10-auto-nollm-after-fix --out <file>
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from .dirty_guard import add_dirty_arguments, check_dirty_write

ROOT = Path(__file__).resolve().parents[2]
TEXT_KEYS = frozenset({"claim", "query", "text", "title", "body", "question"})
MIN_EXACT_CHARS = 4
MIN_NEAR_CHARS = 10
NEAR_JACCARD = 0.8
NEAR_RATIO = 0.85


def normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def _grams(norm: str) -> frozenset[str]:
    return frozenset(norm[i : i + 3] for i in range(max(1, len(norm) - 2)))


def _collect_texts(row: Any, task: str, found: list[tuple[str, str]], key: str = "") -> None:
    if isinstance(row, dict):
        for name, item in row.items():
            _collect_texts(item, task, found, str(name))
    elif isinstance(row, list):
        for item in row:
            _collect_texts(item, task, found, key)
    elif isinstance(row, str) and key in TEXT_KEYS:
        found.append((task, row))


def jsonl_texts(file: Path) -> list[tuple[str, str]]:
    """``(row id, text)`` for every question / claim / document text of one ``.jsonl`` file."""
    found: list[tuple[str, str]] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            fallback = f"{file.name}:{number}"
            task = str(row.get("id") or fallback) if isinstance(row, dict) else fallback
            _collect_texts(row, task, found)
    return found


def slice_texts(path: str | Path) -> list[tuple[str, str]]:
    """``(task id, text)`` for every question / claim / document text in a slice file or folder (``*.jsonl``)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [item for file in files for item in jsonl_texts(file)]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Short location prefixes kept from the fixed-list version (reports and tests cite them); every other file is labelled
# with its path under evaluation/.
_PREFIXES = {"agent_eval_v1.jsonl": "dev", "router_labels_v1.jsonl": "router", "claims_v1.jsonl": "claims"}


def own_files(root: Path = ROOT) -> list[Path]:
    """Every task file under ``evaluation/agent_eval/tasks/`` and claim file under ``evaluation/claim_bench/``."""
    evaluation = root / "evaluation"
    return sorted((evaluation / "agent_eval" / "tasks").glob("*.jsonl")) + sorted(
        (evaluation / "claim_bench").glob("*.jsonl")
    )


@lru_cache(maxsize=2)
def own_texts(root: Path = ROOT) -> tuple[tuple[str, str], ...]:
    """``(location, text)`` for the project's own wording: every task and claim file, and the test literals."""
    found: list[tuple[str, str]] = []
    for file in own_files(root):
        prefix = _PREFIXES.get(file.name) or f"{file.parent.name}/{file.name}"
        found.extend((f"{prefix}:{row_id}", text) for row_id, text in jsonl_texts(file))
    for file in sorted((root / "tests").glob("*.py")):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        relative = file.relative_to(root)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and 2 <= len(node.value) <= 300:
                found.append((f"{relative}:{node.lineno}", node.value))
    return tuple(found)


def overlap_report(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Exact and near copies of the slice's texts among the project's own texts (ids and locations only)."""
    texts = slice_texts(path)
    own = [(location, normalise(text)) for location, text in own_texts(root)]
    own = [(location, norm) for location, norm in own if norm]
    exact_index: dict[str, list[str]] = {}
    for location, norm in own:
        exact_index.setdefault(norm, []).append(location)
    own_grams = [(location, norm, _grams(norm)) for location, norm in own if len(norm) >= MIN_NEAR_CHARS // 2]
    hits: list[dict[str, Any]] = []
    for task, text in texts:
        norm = normalise(text)
        if len(norm) >= MIN_EXACT_CHARS and norm in exact_index:
            hits.append({"task": task, "kind": "exact", "chars": len(norm), "own": sorted(set(exact_index[norm]))[:5]})
            continue
        if len(norm) < MIN_NEAR_CHARS:
            continue
        mine = _grams(norm)
        best: tuple[float, str] | None = None
        for location, other, grams in own_grams:
            if abs(len(other) - len(norm)) > max(len(other), len(norm)) // 2:
                continue
            jaccard = len(mine & grams) / len(mine | grams)
            matcher = difflib.SequenceMatcher(None, norm, other, autojunk=False)
            ratio = matcher.ratio() if matcher.quick_ratio() >= NEAR_RATIO else 0.0
            score = max(jaccard / NEAR_JACCARD, ratio / NEAR_RATIO)
            if score >= 1.0 and (best is None or score > best[0]):
                best = (score, location)
        if best is not None:
            hits.append({"task": task, "kind": "near", "chars": len(norm), "own": [best[1]]})
    tasks = sorted({hit["task"] for hit in hits})
    return {
        "slice": _display(path, root),
        "slice_texts": len(texts),
        "slice_tasks": len({task for task, _text in texts}),
        "own_texts": len(own),
        "own_files": [_display(file, root) for file in own_files(root)],
        "exact": sum(1 for hit in hits if hit["kind"] == "exact"),
        "near": sum(1 for hit in hits if hit["kind"] == "near"),
        "tasks_with_overlap": tasks,
        "hits": hits,
        "method": (
            f"normalised equality (>= {MIN_EXACT_CHARS} chars); near = 3-gram Jaccard >= {NEAR_JACCARD} or difflib "
            f"ratio >= {NEAR_RATIO} (>= {MIN_NEAR_CHARS} chars); own texts = every evaluation/agent_eval/tasks/*.jsonl "
            "and evaluation/claim_bench/*.jsonl text, string literals in tests/*.py"
        ),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """The part stored in a run's config: counts, the slice task ids with an overlap, and the method."""
    keys = ("slice", "slice_texts", "exact", "near", "tasks_with_overlap", "own_files", "method")
    found = {key: report[key] for key in keys if key in report}
    if "content" in report:
        content = report["content"]
        found["content"] = {key: content[key] for key in ("tasks_content", "tasks_content_signature", "method")}
    return found


# --- content level (round-11 review K16) -------------------------------------------------------------------------

CONTENT_RATIO = 0.70
STRUCTURE_MIN_CHARS = 12

# The companies and funds of the offline snapshot (v1 seed and the 2026-09-30 extension), with the Chinese short names
# the tests and slices use. English aliases come from data/synonym_dict.json (the product's own alias list).
_COMPANIES: dict[str, tuple[str, ...]] = {
    "贵州茅台": ("贵州茅台", "茅台"),
    "五粮液": ("五粮液",),
    "中国平安": ("中国平安", "平安"),
    "平安银行": ("平安银行",),
    "美的集团": ("美的集团", "美的"),
    "比亚迪": ("比亚迪",),
    "宁德时代": ("宁德时代", "宁德", "宁王"),
    "中信证券": ("中信证券", "中信"),
    "招商银行": ("招商银行", "招行"),
    "长江电力": ("长江电力", "长电"),
    "工商银行": ("工商银行", "工行"),
    "紫金矿业": ("紫金矿业", "紫金"),
    "中芯国际": ("中芯国际", "中芯"),
    "黄金ETF华安": ("黄金ETF华安", "华安黄金ETF", "黄金ETF"),
    "上证指数": ("上证指数", "上证综指", "沪指"),
    "创业板指": ("创业板指",),
    "沪深300": ("沪深300ETF", "沪深300"),
    "创业板ETF": ("创业板ETF",),
    "证券ETF": ("证券ETF",),
    "中证500ETF南方": ("中证500ETF",),
    "科创50ETF华夏": ("科创50ETF",),
}
_EXTRA_EN = {"中信证券": ("citic",), "长江电力": ("cypc",)}

# Metric and operation classes: (name, pattern) pairs over lower-cased text.
_METRICS: tuple[tuple[str, str], ...] = (
    ("profit_dedt", r"扣非|扣除非经常|non-recurring|recurring net profit|recurring profit"),
    ("circ_mv", r"流通市值|free[- ]float"),
    ("market_cap", r"(?<!流通)市值|market (?:cap|value)|capitali[sz]ation"),
    ("net_profit", r"净利|利润|赚|盈利|earn|profit"),
    ("revenue", r"营收|营业(?:总)?收入|收入|revenue|sales"),
    ("pe", r"市盈率|\bp/?e\b|earnings multiple"),
    ("pb", r"市净率|\bp/?b\b"),
    ("roe", r"roe|净资产收益率"),
    ("eps", r"每股收益|\beps\b"),
    ("gross_margin", r"毛利率|gross margin"),
    ("turnover", r"成交额|成交|turnover|traded value"),
    ("yoy", r"同比|year[- ]on[- ]year|\byoy\b"),
    ("price", r"收盘|股价|价格|报价|现价|收了|close|price|trade[sd]? at|quote"),
)
_OPERATIONS: tuple[tuple[str, str], ...] = (
    (
        "sum",
        r"加起来|加总|合计|一共|总共|总计|共多少|合在一|加在一|总和|together|combined|in sum|in total|altogether|"
        r"between (?:the two|them)|put together|\btotal\b",
    ),
    ("share", r"占|比例|份额|几成|百分之几|percentage of|share of|what percent|what fraction|portion"),
    (
        "difference",
        r"多多少|少多少|差多少|差了|相差|差距|高出|低了|高多少|低多少|弱多少|超出|多赚|少赚|gap|difference|"
        r"by how much|how far|how much more|how much less|\btop\b.*\bby\b|trail|outstrip|\blead",
    ),
    ("multiple", r"倍|times|ratio|multiple"),
    ("extremes", r"最高|最低|最赚钱|最不赚钱|最贵|最便宜|highest|lowest|priciest|cheapest|\bmost\b|\bleast\b"),
    (
        "holding",
        r"(?:\d|[一二两三四五六七八九十百千万])\s*(?:万\s*)?(?:股(?!价|息|东|票|份)|手|份)|"
        r"\d[\d,.]*\s*(?:shares|lots|units)\b|持有|持仓|我有|\bi (?:own|hold)\b|worth",
    ),
    ("cost_basis", r"成本|买入|买的|浮盈|浮亏|bought|\bcost|paper (?:gain|loss)"),
    (
        "forecast",
        r"明天|下周|明年|会不会|能不能|预测|预计|将会|\bwill\b|forecast|predict|next (?:week|year|quarter)|"
        r"20(?:2[7-9]|3\d)|q[1-4]\s*20\d\d",
    ),
    ("foreign_line", r"香港|港股|h股|hong kong|\bhk\b|hk-|港交所|联交所|港元|港币|\bgdr\b|\badr\b|伦敦|london|美股"),
    ("week52", r"52\s*周|52-week|一年内最|近一年"),
)
# Operations specific enough to identify a question about one company (unlike "a forecast about X").
_SINGLE_TARGET_OPS = frozenset({"holding", "cost_basis", "foreign_line", "week52"})
# Metrics with a definition of their own (the round-14 J8 aliases), as specific as an operation.
_SPECIFIC_METRICS = frozenset({"profit_dedt", "circ_mv"})

_CN_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMBER = re.compile(
    r"(?<![0-9A-Za-z.])(\d+(?:[.,]\d+)*)\s*(万亿|亿|万|%|％|个百分点|trillion|billion|bn|million|倍|times)?",
    re.IGNORECASE,
)
_CN_NUMBER = re.compile(r"([零〇一二两三四五六七八九十百千]+)(半)?\s*(万亿|亿|万|倍|成|股|手|份|元)(半)?")
_NOT_A_NUMBER = re.compile(
    r"(?:19|20)\d\d(?:\s*年|\s*q|-\d|\b(?![.,]?\d))|\d+\s*(?:月|日|号)|52\s*(?:周|-?week)|\d+\.(?:hk|sh|sz)|\d{6}(?!\d)",
    re.IGNORECASE,
)
_CURRENCY = re.compile(r"rmb|cny|usd|hkd|us\$|hk\$|\$")
_SCALE = {"万亿": 1e12, "trillion": 1e12, "亿": 1e8, "billion": 1e9, "bn": 1e9, "万": 1e4, "million": 1e6}


def _cn_value(text: str) -> float | None:
    total, current = 0, 0
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[char]
            current = 0
        else:
            return None
    return float(total + current)


def numbers(text: str) -> frozenset[str]:
    """Numbers stated in a text, normalised ("2e+10", "62%", "3.5x"); years, dates and the 52 of "52-week" are not
    numbers, and 万 / 亿 / billion / trillion are applied."""
    found: set[str] = set()
    lowered = _CURRENCY.sub(" ", text.lower())
    for match in _NUMBER.finditer(lowered):
        if _NOT_A_NUMBER.match(lowered, match.start()):
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = (match.group(2) or "").lower()
        if unit in {"%", "％", "个百分点"}:
            found.add(f"{value:g}%")
        elif unit in {"倍", "times"}:
            found.add(f"{value:g}x")
        else:
            found.add(f"{value * _SCALE.get(unit, 1):g}")
    for match in _CN_NUMBER.finditer(text):
        value = _cn_value(match.group(1))
        if not value:
            continue
        value += 0.5 if match.group(2) or match.group(4) else 0.0
        unit = match.group(3)
        if unit == "倍":
            found.add(f"{value:g}x")
        elif unit == "成":
            found.add(f"{value * 10:g}%")
        else:
            found.add(f"{value * _SCALE.get(unit, 1):g}")
    return frozenset(found)


@lru_cache(maxsize=1)
def _company_aliases() -> tuple[tuple[str, str], ...]:
    """``(alias, company)`` pairs, longest alias first; English aliases lower-case."""
    pairs: set[tuple[str, str]] = {(alias.lower(), name) for name, aliases in _COMPANIES.items() for alias in aliases}
    try:
        synonyms = json.loads((ROOT / "data" / "synonym_dict.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        synonyms = {}
    for alias, canonical in (synonyms.get("alias") or {}).items():
        if canonical in _COMPANIES and alias.isascii():
            pairs.add((alias.lower(), canonical))
    for canonical, english in (synonyms.get("display_en") or {}).items():
        if canonical in _COMPANIES:
            pairs.add((str(english).lower(), canonical))
    for canonical, aliases in _EXTRA_EN.items():
        pairs.update((alias, canonical) for alias in aliases)
    return tuple(sorted(pairs, key=lambda pair: (-len(pair[0]), pair)))


def companies(text: str) -> frozenset[str]:
    """The snapshot companies a text names (longest alias first, so 平安银行 is not also 中国平安)."""
    lowered = text.lower()
    taken = [False] * len(lowered)
    found: set[str] = set()
    for alias, name in _company_aliases():
        start = lowered.find(alias)
        while start >= 0:
            end = start + len(alias)
            bounded = not alias.isascii() or (
                (start == 0 or not lowered[start - 1].isalnum()) and (end == len(lowered) or not lowered[end].isalnum())
            )
            if bounded and not any(taken[start:end]):
                found.add(name)
                taken[start:end] = [True] * (end - start)
            start = lowered.find(alias, start + 1)
    return frozenset(found)


def _classes(text: str, table: tuple[tuple[str, str], ...]) -> frozenset[str]:
    lowered = text.lower()
    return frozenset(name for name, pattern in table if re.search(pattern, lowered))


def signature(text: str) -> dict[str, frozenset[str]]:
    """Companies, metric classes, operation classes and numbers of one text (all from fixed lexicons)."""
    metrics = set(_classes(text, _METRICS))
    if "profit_dedt" in metrics:
        metrics.discard("net_profit")
    if "circ_mv" in metrics:
        metrics.discard("market_cap")
    return {
        "companies": companies(text),
        "metrics": frozenset(metrics),
        "operations": _classes(text, _OPERATIONS),
        "numbers": numbers(text),
    }


def masked(text: str) -> str:
    """The normalised text with every snapshot company replaced by ``Ж`` and every number by ``Ш``: its structure
    ("我有Ж Ш股、Ж Ш手…，按收盘价一共值多少")."""
    lowered = _CURRENCY.sub(" ", text.lower())
    spans: list[tuple[int, int, str]] = []
    taken = [False] * len(lowered)
    for alias, _name in _company_aliases():
        start = lowered.find(alias)
        while start >= 0:
            end = start + len(alias)
            bounded = not alias.isascii() or (
                (start == 0 or not lowered[start - 1].isalnum()) and (end == len(lowered) or not lowered[end].isalnum())
            )
            if bounded and not any(taken[start:end]):
                spans.append((start, end, "Ж"))
                taken[start:end] = [True] * (end - start)
            start = lowered.find(alias, start + 1)
    for pattern in (_NUMBER, _CN_NUMBER):
        for match in pattern.finditer(lowered):
            end = match.end(1) if pattern is _NUMBER else match.end(2) if match.group(2) else match.end(1)
            if not any(taken[match.start() : end]):
                spans.append((match.start(), end, "Ш"))
                taken[match.start() : end] = [True] * (end - match.start())
    for start, end, mark in sorted(spans, reverse=True):
        lowered = lowered[:start] + mark + lowered[end:]
    return normalise(lowered)


def content_reasons(mine: dict[str, frozenset[str]], other: dict[str, frozenset[str]]) -> dict[str, list[str]]:
    """Why two signatures share content: ``similar`` reasons (the text ratio must hold too) and ``signature`` ones."""
    shared = {key: mine[key] & other[key] for key in mine}
    reasons: dict[str, list[str]] = {"similar": [], "signature": []}
    if not shared["companies"]:
        return reasons
    if shared["numbers"]:
        reasons["similar"].append("company+number")
    if shared["operations"]:
        reasons["similar"].append("company+operation")
    if shared["metrics"] & _SPECIFIC_METRICS:
        reasons["similar"].append("company+specific metric")
    if shared["numbers"] and shared["metrics"]:
        reasons["signature"].append("company+metric+number")
    same = mine["companies"] == other["companies"]
    if same and len(mine["companies"]) >= 2 and shared["operations"] and shared["metrics"]:
        reasons["signature"].append("companies+operation+metric")
    if (
        same
        and len(mine["companies"]) == 1
        and (shared["operations"] & _SINGLE_TARGET_OPS or shared["metrics"] & _SPECIFIC_METRICS)
        and (shared["metrics"] or not mine["metrics"])
    ):
        reasons["signature"].append("company+specific operation")
    return reasons


def _near(norm: str, other: str, *, ratio_floor: float, jaccard_floor: float | None = None) -> float:
    """The difflib ratio (or 3-gram Jaccard scaled to it) when it clears its floor, else 0."""
    best = 0.0
    if jaccard_floor is not None:
        mine, theirs = _grams(norm), _grams(other)
        jaccard = len(mine & theirs) / len(mine | theirs)
        if jaccard >= jaccard_floor:
            best = jaccard / jaccard_floor * ratio_floor
    matcher = difflib.SequenceMatcher(None, norm, other, autojunk=False)
    if matcher.quick_ratio() >= ratio_floor:
        ratio = matcher.ratio()
        if ratio >= ratio_floor:
            best = max(best, ratio)
    return best


def content_report(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Content-level overlap of a slice with the own texts (ids, locations and reasons only, never the text)."""
    texts = slice_texts(path)
    own = [(location, text) for location, text in own_texts(root) if normalise(text)]
    by_company: dict[str, list[int]] = {}
    own_signatures = []
    own_masked = []
    for index, (_location, text) in enumerate(own):
        found = signature(text)
        own_signatures.append(found)
        own_masked.append(masked(text) if found["operations"] else "")
        for name in found["companies"]:
            by_company.setdefault(name, []).append(index)
    hits: list[dict[str, Any]] = []
    for task, text in texts:
        mine = signature(text)
        norm = normalise(text)
        if len(norm) < MIN_NEAR_CHARS:
            continue
        best: tuple[float, str, list[str]] | None = None
        by_reason: dict[str, set[str]] = {}
        for index in sorted({index for name in mine["companies"] for index in by_company.get(name, [])}):
            reasons = content_reasons(mine, own_signatures[index])
            location = own[index][0]
            for reason in reasons["signature"]:
                by_reason.setdefault(reason, set()).add(location)
            if reasons["similar"]:
                ratio = _near(norm, normalise(own[index][1]), ratio_floor=CONTENT_RATIO)
                if ratio and (best is None or (ratio, location) > (best[0], best[1])):
                    best = (ratio, location, reasons["similar"])
        shape = masked(text)
        if mine["operations"] and len(shape) >= STRUCTURE_MIN_CHARS:
            for index, other in enumerate(own_masked):
                if not other or abs(len(other) - len(shape)) > max(len(other), len(shape)) // 2:
                    continue
                if not mine["operations"] & own_signatures[index]["operations"]:
                    continue
                ratio = _near(shape, other, ratio_floor=NEAR_RATIO, jaccard_floor=NEAR_JACCARD)
                location = own[index][0]
                if ratio and (best is None or (ratio, location) > (best[0], best[1])):
                    best = (ratio, location, ["structure (companies and numbers masked)+operation"])
        if best is not None:
            hits.append(
                {"task": task, "tier": "content", "ratio": round(best[0], 3), "reasons": best[2], "own": [best[1]]}
            )
        if by_reason:
            locations = sorted(set().union(*by_reason.values()))
            hits.append(
                {
                    "task": task,
                    "tier": "content_signature",
                    "reasons": sorted(by_reason),
                    "own": locations[:5],
                    "own_count": len(locations),
                }
            )
    return {
        "tasks_content": sorted({hit["task"] for hit in hits if hit["tier"] == "content"}),
        "tasks_content_signature": sorted({hit["task"] for hit in hits if hit["tier"] == "content_signature"}),
        "hits": hits,
        "method": (
            "signatures from fixed lexicons: snapshot companies (Chinese names, short names, data/synonym_dict.json "
            "English aliases), metric classes, operation classes, numbers (万/亿/%, simple Chinese numerals; years, "
            "dates, stock codes and 52-week excluded). content = (a shared company and a shared number, operation "
            f"or 扣非 / 流通市值 metric, and difflib ratio >= {CONTENT_RATIO}) or (a shared operation and, with "
            f"companies and numbers masked, 3-gram Jaccard >= {NEAR_JACCARD} or difflib ratio >= {NEAR_RATIO}, >= "
            f"{STRUCTURE_MIN_CHARS} masked chars); content_signature = a shared (company, metric, number), or the "
            "same >= 2 companies with a shared operation and metric, or the same one company with a shared holding / "
            "cost-basis / foreign-line / 52-week operation or 扣非 / 流通市值 metric, whatever the wording"
        ),
    }


@contextmanager
def own_root_at(commit: str | None) -> Iterator[Path]:
    """A root holding the own texts (tests, task files, claim files) as they were at ``commit`` (``git archive``)."""
    if not commit:
        yield ROOT
        return
    paths = ["tests", "evaluation/agent_eval/tasks", "evaluation/claim_bench"]
    with tempfile.TemporaryDirectory(prefix="slice-overlap-") as tmp:
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar", commit, *paths], check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        yield Path(tmp)


def is_heldout_path(path: str | Path) -> bool:
    path = Path(path).resolve()
    return any(part.lower().startswith("heldout") for part in (path.parent.name, path.name))


def _display(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else resolved.name


def without_overlap(report: dict[str, Any], run_names: list[str]) -> dict[str, Any]:
    """Pass counts of committed chat runs of the slice (``evaluation/results/<name>.json``) with and without the
    slice tasks the report found, e.g. ``{"chat_heldout_r9-auto-nollm-prefix": {"all": "18/52", ...}}``."""
    from .results import decode_outcomes, load_result

    overlapping = set(report["tasks_with_overlap"])
    counts: dict[str, Any] = {}
    for name in run_names:
        result = load_result(name)
        if not result:
            raise SystemExit(f"evaluation/results/{name}.json not found")
        outcomes = decode_outcomes(result.get("task_outcomes"))
        passed = {task for task, runs in outcomes.items() if runs and all(runs)}
        rest = [task for task in outcomes if task not in overlapping]
        excluded = sorted(task for task in outcomes if task in overlapping)
        counts[name] = {
            "commit": (result.get("config") or {}).get("commit"),
            "all": f"{len(passed)}/{len(outcomes)}",
            "without_overlap": f"{sum(task in passed for task in rest)}/{len(rest)}",
            "excluded": {task: task in passed for task in excluded},
        }
    return counts


def exclusions(
    report: dict[str, Any],
    run_names: list[str],
    extra: dict[str, list[str]] | None = None,
    *,
    claim_runs: list[str] | None = None,
    claim_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Pass counts of committed chat runs (and claim runs) with each exclusion set removed: the text-level overlap,
    text + ``content``, text + ``content`` + ``content_signature``, and any named extra set (e.g. the task ids a
    review counted). A claim run passes every slice claim not listed in its ``errors``."""
    from .results import decode_outcomes, load_result

    text = set(report["tasks_with_overlap"])
    content = report.get("content") or {}
    sets = {"text": text}
    if content:
        sets["text+content"] = text | set(content["tasks_content"])
        sets["text+content+signature"] = sets["text+content"] | set(content["tasks_content_signature"])
    for name, tasks in (extra or {}).items():
        sets[name] = set(tasks)

    def row(commit: Any, items: list[str], passed: set[str]) -> dict[str, Any]:
        found: dict[str, Any] = {"commit": commit, "all": f"{len(passed)}/{len(items)}"}
        for name, excluded in sets.items():
            rest = [item for item in items if item not in excluded]
            found[f"without_{name}"] = f"{sum(item in passed for item in rest)}/{len(rest)}"
        return found

    counts: dict[str, Any] = {}
    for run in [*run_names, *(claim_runs or [])]:
        result = load_result(run)
        if not result:
            raise SystemExit(f"evaluation/results/{run}.json not found")
        commit = (result.get("config") or {}).get("commit")
        if run in run_names:
            outcomes = decode_outcomes(result.get("task_outcomes"))
            counts[run] = row(commit, list(outcomes), {task for task, runs in outcomes.items() if runs and all(runs)})
        else:
            wrong = {str(error.get("id")) for error in result.get("errors") or []}
            items = list(claim_ids or [])
            counts[run] = row(commit, items, {item for item in items if item not in wrong})
    return {"excluded_tasks": {name: sorted(tasks) for name, tasks in sets.items()}, "runs": counts}


def slice_claim_ids(path: str | Path) -> list[str]:
    """Ids of the slice rows that are claims (a ``claim`` text and no turns)."""
    path = Path(path)
    ids: list[str] = []
    for file in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        for row in _jsonl(file):
            if isinstance(row, dict) and "claim" in row and "turns" not in row:
                ids.append(str(row.get("id")))
    return ids


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Overlap of a held-out slice with own dev tasks and tests.")
    parser.add_argument("slice", help="A slice .jsonl file or a folder of them.")
    parser.add_argument("--out", default=None, help="Write the full report (ids and locations only) here.")
    parser.add_argument(
        "--chat-runs",
        nargs="*",
        default=[],
        help="Committed chat runs of the slice (evaluation/results stems): pass counts with and without the overlap.",
    )
    parser.add_argument("--content", action="store_true", help="Add the content-level check (round-11 review K16).")
    parser.add_argument("--own-at", default=None, help="Read the own texts from this commit (git archive).")
    parser.add_argument(
        "--also-exclude",
        action="append",
        default=[],
        metavar="NAME=ID,ID,...",
        help="A further named exclusion set for --chat-runs (e.g. the task ids a review counted).",
    )
    parser.add_argument(
        "--claim-runs",
        nargs="*",
        default=[],
        help="Committed claim runs of the slice (evaluation/results stems), counted per exclusion.",
    )
    add_dirty_arguments(parser)
    args = parser.parse_args(argv)
    dirty = None
    if args.out:
        from .runner import _git_commit

        dirty = check_dirty_write(args.out, _git_commit(), allow_dirty=args.allow_dirty, notes=args.note)
    with own_root_at(args.own_at) as root:
        own_texts.cache_clear()
        try:
            report = overlap_report(args.slice, root=root)
            if args.content:
                report["content"] = content_report(args.slice, root=root)
        finally:
            own_texts.cache_clear()
    if args.own_at:
        report["own_files"] = [f"{name}@{args.own_at}" for name in report["own_files"]]
    if args.chat_runs:
        report["chat_runs"] = without_overlap(report, args.chat_runs)
        if args.content or args.also_exclude:
            extra = {}
            for item in args.also_exclude:
                name, _, ids = item.partition("=")
                extra[name] = [task for task in ids.split(",") if task]
            report["exclusions"] = exclusions(
                report,
                args.chat_runs,
                extra,
                claim_runs=args.claim_runs,
                claim_ids=slice_claim_ids(args.slice),
            )
    if args.out:
        from datetime import UTC, datetime

        from .runner import _git_commit

        given = sys.argv[1:] if argv is None else argv
        command = " ".join(["python -m evaluation.agent_eval.slice_overlap", *given])
        config = {
            "slice": report["slice"],
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": command,
        }
        if args.own_at:
            config["own_texts_commit"] = args.own_at
        if dirty:
            config["dirty_tree"] = dirty
        report = {"kind": "slice_overlap_scan", "config": config, **report}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary(report), ensure_ascii=False, indent=1))
    if "exclusions" in report:
        print(json.dumps(report["exclusions"]["runs"], ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
