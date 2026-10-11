"""Render the landing-page numbers of README.md and README_CN.md from committed result files (round-13 review M14).

The READMEs keep hand-written prose; the parts that quote results sit between markers and are rewritten here:

    <!-- generated:<region> (python -m evaluation.agent_eval.readme) -->
    ...
    <!-- /generated:<region> -->

Regions: ``pitch`` (the "In 30 seconds" list), ``headline-table`` (the "What is measured" table), ``ci-status`` (the
pre-push and CI paragraph) and ``limits`` (the short limits list). Every number in them is read from a committed file:

- the reviewer slices: ``evaluation/heldout_r<N>/`` with ``chat_heldout_r<N>-auto-nollm-{prefix,after-fix}.json``,
  ``claims_heldout_r<N>-{prefix,after-fix}.json`` and ``slice_overlap-heldout_r<N>-*.json`` (the overlap tiers are
  recomputed from the slice-side overlap lists and the two runs' ``task_outcomes``). The *latest slice* is the highest
  N that has a committed folder and an after-fix chat run; the fix round comes from its docs/agent-eval.md heading;
- the verifier stress test: the highest ``verifier_stress-round<N>.json``;
- the gates: ``gate-{dev,holdout,ext,shipped}.json``; the router labels: ``router_eval-independent_v2-*.json``;
- the real-claims sets: every blind ``claims_web_v<K>-first-run.json`` and its latest ``-after-exposure-round<R>``;
- the red team: the highest ``redteam-offline-r<N>.json``, the last full LLM run and the latest targeted LLM run;
- the CI run: the newest ``docs/results/ci/ci-run-*.json`` (written by ``python scripts/record_ci_run.py`` from the
  GitHub API after a push); the test count: the newest ``docs/results/ci/prepush-*.md`` (``scripts/prepush.sh``).

    python -m evaluation.agent_eval.readme            # rewrite the regions
    python -m evaluation.agent_eval.readme --check    # exit 1 if a region is missing or out of date

``python -m evaluation.agent_eval.report`` renders the regions too, and ``report --check`` (CI job heldout-verify,
pre-push step report-check) fails when they are stale: a new result file that changes a headline number has to be
committed with the re-rendered READMEs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
EVAL_DOC = ROOT / "docs" / "agent-eval.md"
CI_DIR = ROOT / "docs" / "results" / "ci"
READMES = {"en": ROOT / "README.md", "zh": ROOT / "README_CN.md"}
REGIONS = ("pitch", "headline-table", "ci-status", "limits")
COMMAND = "python -m evaluation.agent_eval.readme"

# Fixed inputs: historical runs that a later file does not supersede.
TEST_V3 = "ablation-final4-deepseek-testv3-holdout"  # 9536abf, the first LLM run on test v3
TEST_V3_PURE_LLM = "ablation-test_v3-purellm-deepseek"
LATENCY = "perf-merged-prefetch-deepseek"  # the current latency defaults, held-out, DeepSeek
LOAD_DRILL = ROOT / "docs" / "results" / "perf" / "sources-load" / "chaos-sources-load-8users.json"
ROUTER_FIRST = "router_eval-independent_v2-first-run"
GATES = ("dev", "holdout", "ext", "shipped")


def begin_marker(region: str) -> str:
    return f"<!-- generated:{region} ({COMMAND}) -->"


def end_marker(region: str) -> str:
    return f"<!-- /generated:{region} -->"


# ---------------------------------------------------------------------------------------------------------- loading


@cache
def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result(stem: str) -> dict[str, Any]:
    return _json(RESULTS / f"{stem}.json")


def _numbered(pattern: str) -> list[tuple[int, str]]:
    """(N, stem) for result files whose stem matches ``pattern`` with one ``(\\d+)`` group, sorted by N."""
    regex = re.compile(pattern)
    found = []
    for path in RESULTS.glob("*.json"):
        match = regex.fullmatch(path.stem)
        if match:
            found.append((int(match.group(1)), path.stem))
    return sorted(found)


def short(commit: Any) -> str:
    return str(commit or "")[:7]


# ------------------------------------------------------------------------------------------------- anchors (GitHub)


def slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation other than ``-`` / ``_`` dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


@dataclass(frozen=True)
class Section:
    level: int
    heading: str
    anchor: str
    body: str  # text up to the next heading of any level


def sections(path: Path) -> list[Section]:
    lines = path.read_text(encoding="utf-8").splitlines()
    heads: list[tuple[int, int, str]] = []
    fence = False
    for index, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
        match = None if fence else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if match:
            heads.append((index, len(match.group(1)), match.group(2)))
    seen: dict[str, int] = {}
    out = []
    for position, (index, level, heading) in enumerate(heads):
        base = slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        end = heads[position + 1][0] if position + 1 < len(heads) else len(lines)
        out.append(
            Section(level, heading, base if count == 0 else f"{base}-{count}", "\n".join(lines[index + 1 : end]))
        )
    return out


def anchor_by_heading(prefix: str, path: Path = EVAL_DOC) -> str:
    for section in sections(path):
        if section.heading.startswith(prefix):
            return section.anchor
    raise LookupError(f"{path.relative_to(ROOT)} has no heading starting with {prefix!r}")


def anchor_for_result(stem: str, path: Path = EVAL_DOC) -> str:
    """The section of docs/agent-eval.md that renders ``<stem>.json``: its heading names the file, else its text does."""
    cited = re.compile(rf"(?<![\w-]){re.escape(stem)}\.json")
    found = sections(path)
    for section in found:
        if cited.search(section.heading):
            return section.anchor
    for section in found:
        if section.level >= 3 and cited.search(section.body):
            return section.anchor
    raise LookupError(f"{path.relative_to(ROOT)} has no section for {stem}.json")


# ---------------------------------------------------------------------------------------------------------- facts


@dataclass(frozen=True)
class Slice:
    round: int
    fix_round: int | None
    anchor: str  # docs/agent-eval.md
    tasks: int
    before: int
    after: int
    claims: int | None
    claims_before: float | None
    claims_after: float | None
    overlap: str | None  # stem of the slice_overlap file used for the tiers
    tiers: tuple[tuple[int, int, int], ...]  # (tasks kept, passes before, passes after): text, + content, + signature
    readme_anchor: str | None  # evaluation/heldout_r<N>/README.md section on the after-fix run

    @property
    def rate_before(self) -> float:
        return self.before / self.tasks

    @property
    def rate_after(self) -> float:
        return self.after / self.tasks

    @property
    def readme(self) -> str:
        return f"evaluation/heldout_r{self.round}/README.md" + (f"#{self.readme_anchor}" if self.readme_anchor else "")


def _passes(outcomes: dict[str, str], keep: set[str]) -> int:
    return sum(1 for task in keep if outcomes.get(task) == "1")


def _overlap_tiers(stem: str, before: dict[str, str], after: dict[str, str]) -> tuple[tuple[int, int, int], ...]:
    scan = result(stem)
    content = scan.get("content") or {}
    chat = set(after)
    text = set(scan.get("tasks_with_overlap") or [])
    excluded = [text, text | set(content.get("tasks_content") or [])]
    excluded.append(excluded[1] | set(content.get("tasks_content_signature") or []))
    tiers = []
    for exclude in excluded:
        keep = chat - exclude
        tiers.append((len(keep), _passes(before, keep), _passes(after, keep)))
    return tuple(tiers)


def _slice(number: int) -> Slice:
    before = result(f"chat_heldout_r{number}-auto-nollm-prefix")
    after = result(f"chat_heldout_r{number}-auto-nollm-after-fix")
    heading_prefix = f"Round-{number} held-out chat slice"
    heading = next((s for s in sections(EVAL_DOC) if s.heading.startswith(heading_prefix)), None)
    if heading is None:
        raise LookupError(
            f"docs/agent-eval.md has no section {heading_prefix!r}: render the slice in evaluation/agent_eval/report.py"
        )
    fix = re.search(r"round-(\d+) fixes", heading.heading)
    claims_before = RESULTS / f"claims_heldout_r{number}-prefix.json"
    claims_after = RESULTS / f"claims_heldout_r{number}-after-fix.json"
    has_claims = claims_before.exists() and claims_after.exists()
    # the overlap scan with content-level lists (round-11 review K16); the newest one if there are several
    scans = sorted(
        path.stem
        for path in RESULTS.glob(f"slice_overlap-heldout_r{number}-*.json")
        if "content" in _json(path) and "all-tasks" not in path.stem
    )
    readme = ROOT / "evaluation" / f"heldout_r{number}" / "README.md"
    readme_anchor = None
    if readme.exists():
        readme_anchor = next(
            (
                s.anchor
                for s in sections(readme)
                if re.search(r"committed to the repository|after-fix run", s.heading, re.I)
            ),
            None,
        )
    return Slice(
        round=number,
        fix_round=int(fix.group(1)) if fix else None,
        anchor=heading.anchor,
        tasks=after["summary"]["tasks"],
        before=_passes(before["task_outcomes"], set(after["task_outcomes"])),
        after=_passes(after["task_outcomes"], set(after["task_outcomes"])),
        claims=_json(claims_after)["summary"]["claims"] if has_claims else None,
        claims_before=_json(claims_before)["summary"]["verdict_accuracy"] if has_claims else None,
        claims_after=_json(claims_after)["summary"]["verdict_accuracy"] if has_claims else None,
        overlap=scans[-1] if scans else None,
        tiers=_overlap_tiers(scans[-1], before["task_outcomes"], after["task_outcomes"]) if scans else (),
        readme_anchor=readme_anchor,
    )


def slice_rounds() -> list[int]:
    """Reviewer slices with a committed folder and both chat runs (before and after one round of fixes)."""
    rounds = []
    for number, _stem in _numbered(r"chat_heldout_r(\d+)-auto-nollm-after-fix"):
        if (ROOT / "evaluation" / f"heldout_r{number}").is_dir() and (
            RESULTS / f"chat_heldout_r{number}-auto-nollm-prefix.json"
        ).exists():
            rounds.append(number)
    return rounds


@dataclass(frozen=True)
class ClaimsSet:
    version: int
    claims: int
    first: float
    first_ci: tuple[float, float]
    first_commit: str
    after: float | None
    after_round: int | None
    first_stem: str


def claims_sets() -> list[ClaimsSet]:
    out = []
    for version, stem in _numbered(r"claims_web_v(\d+)-first-run"):
        data = result(stem)
        if "not blind" in str(data["config"].get("note", "")):
            continue  # v1: the collector had seen the checker
        later = _numbered(rf"claims_web_v{version}-after-exposure-round(\d+)")
        after = result(later[-1][1])["summary"]["verdict_accuracy"] if later else None
        summary = data["summary"]
        out.append(
            ClaimsSet(
                version=version,
                claims=summary["claims"],
                first=summary["verdict_accuracy"],
                first_ci=tuple(summary["ci95"]),
                first_commit=short(data["config"]["commit"]),
                after=after,
                after_round=later[-1][0] if later else None,
                first_stem=stem,
            )
        )
    return out


@dataclass(frozen=True)
class RedTeam:
    stem: str
    anchor: str
    sets: int
    runs: int
    successes: int
    llm_stem: str
    llm_anchor: str
    llm_commit: str
    llm_low: float
    llm_high: float
    targeted_stem: str | None
    targeted_set: str | None
    targeted: tuple[tuple[str, int, int], ...]  # (mode, successes, runs)


def red_team() -> RedTeam:
    stem = _numbered(r"redteam-offline-r(\d+)")[-1][1]
    data = result(stem)
    paths = data["paths"]
    # the last full LLM run (every attack set, not a targeted replay)
    full = sorted(
        (result(p.stem)["config"].get("run_at", ""), p.stem)
        for p in RESULTS.glob("redteam-*llm*.json")
        if result(p.stem)["config"].get("llm")
        and not result(p.stem)["config"].get("targeted_cases")
        and not result(p.stem)["config"].get("llm_replay")
    )
    llm_stem = full[-1][1]
    llm = result(llm_stem)
    rates = [p["attack_success"] for p in llm["paths"]]
    targeted = _numbered(r"redteam-r(\d+)-holdout\d+-llm")
    targeted_rows: tuple[tuple[str, int, int], ...] = ()
    targeted_set = None
    if targeted:
        tdata = result(targeted[-1][1])
        targeted_set = tdata["paths"][0]["attack_set"]
        targeted_rows = tuple((p["mode"], round(p["attack_success"] * p["runs"]), p["runs"]) for p in tdata["paths"])
    return RedTeam(
        stem=stem,
        anchor=anchor_for_result(stem),
        sets=len(data["config"]["attacks"]),
        runs=sum(p["runs"] for p in paths),
        successes=round(sum(p["attack_success"] * p["runs"] for p in paths)),
        llm_stem=llm_stem,
        llm_anchor=anchor_for_result(llm_stem),
        llm_commit=short(llm["config"].get("commit")),
        llm_low=min(rates),
        llm_high=max(rates),
        targeted_stem=targeted[-1][1] if targeted else None,
        targeted_set=targeted_set,
        targeted=targeted_rows,
    )


@dataclass(frozen=True)
class CiRun:
    path: Path
    run_id: int
    url: str
    sha: str
    conclusion: str
    jobs: tuple[tuple[str, str, float, int], ...]  # (name, conclusion, minutes, timeout-minutes)
    recorded_at: str


def ci_run() -> CiRun | None:
    records = [(_json(p).get("created_at", ""), p) for p in CI_DIR.glob("ci-run-*.json")]
    if not records:
        return None
    _created, path = max(records)
    data = _json(path)
    return CiRun(
        path=path,
        run_id=int(data["run_id"]),
        url=data["url"],
        sha=short(data["head_sha"]),
        conclusion=data["conclusion"],
        jobs=tuple((j["name"], j["conclusion"], float(j["minutes"]), int(j["timeout_minutes"])) for j in data["jobs"]),
        recorded_at=data["recorded_at"][:10],
    )


@dataclass(frozen=True)
class Prepush:
    path: Path
    sha: str
    run_at: str
    all_passed: bool
    pytest: tuple[tuple[int, str], ...]  # the pytest step's counts, e.g. ((4397, "passed"), (70, "skipped"))
    pytest_shipped: tuple[tuple[int, str], ...] | None

    def tests(self, lang: str, shipped: bool = False) -> str:
        counts = self.pytest_shipped if shipped else self.pytest
        if not counts:
            return ""
        if lang == "zh":
            return "、".join(f"{n:,} 个{_ZH_OUTCOME.get(word, word)}" for n, word in counts)
        return ", ".join(f"{n:,} {word}" for n, word in counts)


_ZH_OUTCOME = {
    "passed": "通过",
    "failed": "失败",
    "error": "错误",
    "errors": "错误",
    "skipped": "跳过",
    "xfailed": "预期失败",
    "xpassed": "意外通过",
    "deselected": "未选中",
}


def _step_line(text: str, step: str) -> tuple[str, str] | None:
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0] == step:
            return cells[2], cells[-1]
    return None


def prepush() -> Prepush | None:
    found = []
    for path in CI_DIR.glob("prepush-*.md"):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"Run by `scripts/prepush\.sh` \((\S+)\)", text)
        if match:
            found.append((match.group(1), path, text))
    if not found:
        return None
    run_at, path, text = max(found)
    pytest_row = _step_line(text, "pytest")
    shipped_row = _step_line(text, "pytest-shipped")

    def tests_line(row: tuple[str, str] | None) -> tuple[tuple[int, str], ...] | None:
        """(count, outcome) pairs from pytest's last line (warnings and the time left out)."""
        if not row:
            return None
        counts = re.findall(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b", row[1])
        return tuple((int(n), word) for n, word in counts)

    return Prepush(
        path=path,
        sha=path.stem.removeprefix("prepush-"),
        run_at=run_at[:10],
        all_passed="**all steps passed**" in text,
        pytest=tests_line(pytest_row) or (),
        pytest_shipped=tests_line(shipped_row),
    )


@dataclass(frozen=True)
class Facts:
    slices: list[Slice]
    verifier_round: int
    verifier_stem: str
    verifier_anchor: str
    verifier: dict[str, Any]
    test_v3: dict[str, Any]
    test_v3_anchor: str
    pure_llm: tuple[int, int]
    router_first: dict[str, Any]
    router_latest: dict[str, Any]
    router_anchor: str
    claims: list[ClaimsSet]
    red_team: RedTeam
    gates: dict[str, tuple[int, float]]
    gate_anchor: str
    latency_p95_s: float
    ttft_p50_s: float
    load_users: int
    load_requests: int
    load_errors: int
    ci: CiRun | None
    prepush: Prepush | None


def collect() -> Facts:
    _json.cache_clear()
    rounds = slice_rounds()
    if len(rounds) < 3:
        raise LookupError("fewer than three reviewer slices with before and after-fix runs")
    verifier_round, verifier_stem = _numbered(r"verifier_stress-round(\d+)")[-1]
    ablation = result(TEST_V3)
    pure = result(TEST_V3_PURE_LLM)["results"]["test_v3"]["pure_llm"]["summary"]
    router_rounds = _numbered(r"router_eval-independent_v2-round(\d+)")
    latency = result(LATENCY)["results"]["holdout"]["agent"]["summary"]
    load = _json(LOAD_DRILL)
    requests = sum(phase["requests"] for phase in load["phases"])
    errors = sum(
        count for phase in load["phases"] for status, count in phase["statuses"].items() if not status.startswith("2")
    )
    return Facts(
        slices=[_slice(number) for number in rounds],
        verifier_round=verifier_round,
        verifier_stem=verifier_stem,
        verifier_anchor=anchor_for_result(verifier_stem),
        verifier=result(verifier_stem),
        test_v3=ablation,
        test_v3_anchor=anchor_by_heading("Test set v3 (independent author, untouched)"),
        pure_llm=(round(pure["task_success"] * pure["tasks"]), pure["tasks"]),
        router_first=result(ROUTER_FIRST),
        router_latest=result(router_rounds[-1][1]) if router_rounds else result(ROUTER_FIRST),
        router_anchor=anchor_for_result(ROUTER_FIRST),
        claims=claims_sets(),
        red_team=red_team(),
        gates={
            name: (result(f"gate-{name}")["summary"]["tasks"], result(f"gate-{name}")["summary"]["task_success"])
            for name in GATES
        },
        gate_anchor=anchor_by_heading("Offline gate baselines"),
        latency_p95_s=latency["latency_ms_p95"] / 1000,
        ttft_p50_s=latency["ttft_ms_p50"] / 1000,
        load_users=load["load"]["users"],
        load_requests=requests,
        load_errors=errors,
        ci=ci_run(),
        prepush=prepush(),
    )


# ------------------------------------------------------------------------------------------------------ formatting


def rate(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def pct(value: float) -> str:
    if value == 0:
        return "0.0%"
    if value == 1:
        return "100%"
    return f"{value * 100:.2f}%"


def ci2(bounds: Any) -> str:
    low, high = bounds
    return f"[{low:.2f}, {high:.2f}]"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def frac(passes: int, tasks: int) -> str:
    return f"{passes}/{tasks}"


def tier_text(tier: tuple[int, int, int]) -> str:
    kept, before, after = tier
    return f"{before}/{kept} → {after}/{kept}"


_ZH_DIGITS = "零一二三四五六七八九"


def zh_number(value: int) -> str:
    """第十二轮-style Chinese numerals for 1–99 (review rounds are written this way in README_CN.md)."""
    tens, ones = divmod(value, 10)
    if tens == 0:
        return _ZH_DIGITS[ones]
    head = "" if tens == 1 else _ZH_DIGITS[tens]
    return f"{head}十{_ZH_DIGITS[ones] if ones else ''}"


def link_json(stem: str, target: str) -> str:
    return f"[`{stem}.json`]({target})"


# ---------------------------------------------------------------------------------------------------------- regions


def _test_v3(facts: Facts) -> dict[str, Any]:
    results = facts.test_v3["results"]["test_v3"]
    comparisons = facts.test_v3["comparisons"]["test_v3"]
    vs_workflow = comparisons["agent_vs_workflow"]
    return {
        "tasks": results["agent"]["summary"]["tasks"],
        "agent": rate(results["agent"]["summary"]["task_success"]),
        "agent_ci": ci2(results["agent"]["summary"]["ci"]["task_success"]),
        "composition": rate(results["workflow_llm"]["summary"]["task_success"]),
        "workflow": rate(results["workflow"]["summary"]["task_success"]),
        "diff": signed(vs_workflow["task_success"]["diff"]),
        "diff_ci": "[" + ", ".join(signed(x) for x in vs_workflow["task_success"]["ci"]) + "]",
        "p": f"{vs_workflow['mcnemar']['p_value']:.3f}",
        "composition_passk_significant": comparisons["agent_vs_workflow_llm"]["pass^k"]["significant"],
        "pure": f"{facts.pure_llm[0]}/{facts.pure_llm[1]}",
    }


def _verifier(facts: Facts) -> dict[str, Any]:
    data = facts.verifier
    return {
        "round": facts.verifier_round,
        "commit": short(data["config"]["commit"]),
        "gold": data["gold_answers"],
        "variants": f"{data['variants']:,}",
        "strict": pct(data["false_accept"]["claim"]),
        "derived": pct(data["false_accept"]["claim_derived"]),
        "true_accept": data["true_accept"]["claim"],
        "swap": pct(data["by_kind"]["swap"]["false_accept_claim"]),
        "swap_derived": pct(data["by_kind"]["swap"]["false_accept_claim_derived"]),
        "link": f"docs/agent-eval.md#{facts.verifier_anchor}",
        "stem": facts.verifier_stem,
    }


def render_pitch(facts: Facts, lang: str) -> str:
    v = _verifier(facts)
    t = _test_v3(facts)
    latest, previous, earlier = facts.slices[-1], facts.slices[-2], facts.slices[-3]
    blind = facts.claims[0] if facts.claims else None
    first_router = rate(facts.router_first["accuracy"])
    tiers = [f"{t_[2]}/{t_[0]}" for t_ in latest.tiers]
    if lang == "en":
        accepts = (
            f"all {v['gold']} correct ones"
            if v["true_accept"] == 1
            else f"{pct(v['true_accept'])} of the {v['gold']} correct ones"
        )
        claims = f"; claims {rate(latest.claims_before)} → {rate(latest.claims_after)}" if latest.claims else ""
        held = (
            f" The latest still holds with the tasks that overlap our own wording removed: {tiers[0]} by text, "
            f"{tiers[1]} by content and {tiers[2]} by content signature ([provenance]({latest.readme}))."
            if tiers
            else ""
        )
        lines = [
            f"- **Numbers you can trace.** A claim-level verifier ties every number to the evidence cited in its sentence. "
            f"On {v['variants']} corrupted answers it wrongly accepts **{v['strict']}** ({v['derived']} with derived "
            f"numbers allowed, the default), and it accepts {accepts} ([verifier stress test]({v['link']}), round "
            f"{v['round']}, `{v['commit']}`).",
            f"- **Measured on a set nobody tuned against.** On test v3 ({t['tasks']} tasks by a separate author, first LLM "
            f"run), the DeepSeek agent passes **{t['agent']}** {t['agent_ci']} under strict, citation-gated scoring. The "
            f"deterministic workflow passes {t['workflow']} and an LLM without tools passes {t['pure']} "
            f"([test v3](docs/agent-eval.md#{facts.test_v3_anchor})).",
            f"- **Out of sample after the fixes, with the overlap counted.** Each reviewer writes a held-out slice from "
            f"the bugs they found, before any fix. The latest slice (round {latest.round}) went "
            f"**{rate(latest.rate_before)} → {rate(latest.rate_after)}** after one round of fixes "
            f"({frac(latest.before, latest.tasks)} → {frac(latest.after, latest.tasks)} chat tasks{claims}), the two "
            f"before **{rate(previous.rate_before)} → {rate(previous.rate_after)}** and "
            f"**{rate(earlier.rate_before)} → {rate(earlier.rate_after)}**.{held}",
            f"- **First runs are reported, not replaced.** Every independent set keeps its first-run number next to any "
            f'later "after exposure" one. For example, a fresh router label set scored {first_router} on its first run'
            + (f", and a blind set of real claims {rate(blind.first)}" if blind else "")
            + " ([history](docs/evaluation-history.md#independent-sets-first-runs-vs-after-exposure)).",
            "- **Runs like a service.** It works without an LLM, and every LLM call is bounded by a run deadline. It has "
            "been load- and chaos-tested and runs on k3s with Postgres-shared sessions. It speaks MCP and A2A and ships "
            "with Prometheus, Grafana and Jaeger ([performance](docs/performance.md), [deployment](docs/deployment.md)).",
        ]
    else:
        accepts = (
            f"{v['gold']} 个正确答案全部通过"
            if v["true_accept"] == 1
            else f"{v['gold']} 个正确答案通过 {pct(v['true_accept'])}"
        )
        claims = f"；声明 {rate(latest.claims_before)} → {rate(latest.claims_after)}" if latest.claims else ""
        held = (
            f"去掉与自写措辞重合的任务后最近一组仍然成立：按文本 {tiers[0]}，按内容 {tiers[1]}，按内容特征 {tiers[2]}"
            f"（[来源说明]({latest.readme})）。"
            if tiers
            else ""
        )
        lines = [
            f"- **数字可追溯**：逐句校验器把每个数字绑定到本句引用的证据。{v['variants']} 个被篡改的答案里，它只误放 "
            f"**{v['strict']}**（默认允许派生数字时 {v['derived']}）；{accepts}（[校验器压力测试]({v['link']})，"
            f"第 {v['round']} 轮，`{v['commit']}`）。",
            f"- **在没人调过的任务集上测**：测试集 v3 由独立作者编写，共 {t['tasks']} 个任务，这里报的是首次 LLM 运行。"
            f"在要求引用的一票否决口径下，DeepSeek Agent 通过 **{t['agent']}** {t['agent_ci']}，确定性固定流程 "
            f"{t['workflow']}，不调工具的 LLM 为 {t['pure']}（[测试集 v3](docs/agent-eval.md#{facts.test_v3_anchor})）。",
            f"- **修复之后的样本外结果，并把重合算进去**：每轮评审在任何修复之前，按自己发现的缺陷写一组保留切片。最近一组"
            f"（第{zh_number(latest.round)}轮）切片经过一轮修复后从 **{rate(latest.rate_before)} → "
            f"{rate(latest.rate_after)}**（对话 {frac(latest.before, latest.tasks)} → "
            f"{frac(latest.after, latest.tasks)}{claims}），之前两组为 **{rate(previous.rate_before)} → "
            f"{rate(previous.rate_after)}** 和 **{rate(earlier.rate_before)} → {rate(earlier.rate_after)}**。{held}",
            f"- **首次运行的数字照报，不被替换**：每套独立任务集都同时保留首次运行的数字和之后标为「暴露后」的数字。例如新写"
            f"的独立路由标注首次得 {first_router}"
            + (f"，盲收集的真实说法首次得 {rate(blind.first)}" if blind else "")
            + "（[评测历史](docs/zh/evaluation-history.md#独立任务集首次运行与暴露后)）。",
            "- **按服务标准交付**：没有 LLM 也能运行，每次 LLM 调用都受运行截止时间约束。做过压测和故障演练，k3s 多副本经 "
            "Postgres 共享会话。提供 MCP 和 A2A，配有 Prometheus、Grafana、Jaeger 监控（[性能](docs/zh/performance.md)、"
            "[部署](docs/zh/deployment.md)）。",
        ]
    return "\n".join(lines)


def _slice_row(s: Slice, label: str, lang: str) -> str:
    chat_link = link_json(f"chat_heldout_r{s.round}-auto-nollm-after-fix", f"docs/agent-eval.md#{s.anchor}")
    overlap_link = link_json(s.overlap, s.readme) if s.overlap else ""
    links = chat_link + ((", " if lang == "en" else "、") + overlap_link if overlap_link else "")
    if lang == "en":
        what = f"{label} reviewer slice (round {s.round}): {s.tasks} chat tasks" + (
            f" and {s.claims} claims" if s.claims else ""
        )
        what += ", run once before and once after one round of fixes"
        result_text = f"Chat **{rate(s.rate_before)} → {rate(s.rate_after)}**"
        if s.claims:
            result_text += f", claims {rate(s.claims_before)} → {rate(s.claims_after)}"
        if s.tiers:
            result_text += (
                f". Without overlapping tasks: {tier_text(s.tiers[0])} (text), {tier_text(s.tiers[1])} (+ content), "
                f"{tier_text(s.tiers[2])} (+ signature)"
            )
    else:
        what = f"{label}评审切片（第{zh_number(s.round)}轮）：{s.tasks} 个对话任务" + (
            f"、{s.claims} 条声明" if s.claims else ""
        )
        what += "，一轮修复前后各跑一次"
        result_text = f"对话 **{rate(s.rate_before)} → {rate(s.rate_after)}**"
        if s.claims:
            result_text += f"，声明 {rate(s.claims_before)} → {rate(s.claims_after)}"
        if s.tiers:
            result_text += (
                f"。去掉重合任务后：{tier_text(s.tiers[0])}（文本），{tier_text(s.tiers[1])}（加内容），"
                f"{tier_text(s.tiers[2])}（加内容特征）"
            )
    return f"| {what} | {result_text} | {links} |"


def render_table(facts: Facts, lang: str) -> str:
    t = _test_v3(facts)
    v = _verifier(facts)
    r = facts.red_team
    latest, previous, earlier = facts.slices[-1], facts.slices[-2], facts.slices[-3]
    older = facts.slices[:-3]
    first_router = facts.router_first
    router_latest = facts.router_latest
    gates = facts.gates
    llm_range = (
        f"{r.llm_high * 100:.1f}%" if r.llm_low == r.llm_high else f"{r.llm_low * 100:.0f}–{r.llm_high * 100:.1f}%"
    )
    mode_en = {"workflow_llm": "composition", "agent": "agent"}
    mode_zh = {"workflow_llm": "组织答案", "agent": "Agent"}
    v3_link = link_json(TEST_V3, f"docs/agent-eval.md#{facts.test_v3_anchor}")
    stress_link = link_json(v["stem"], v["link"])
    router_link = link_json(ROUTER_FIRST, f"docs/agent-eval.md#{facts.router_anchor}")
    red_link = link_json(r.stem, f"docs/agent-eval.md#{r.anchor}")
    llm_link = link_json(r.llm_stem, f"docs/agent-eval.md#{r.llm_anchor}")
    targeted_link = (
        link_json(r.targeted_stem, f"docs/agent-eval.md#{anchor_for_result(r.targeted_stem)}")
        if r.targeted_stem
        else ""
    )
    gate_link_target = f"docs/agent-eval.md#{facts.gate_anchor}"
    if lang == "en":
        older_text = "; ".join(f"round {s.round}: {rate(s.rate_before)} → {rate(s.rate_after)}" for s in older)
        older_text = older_text[:1].upper() + older_text[1:]
        swap = (
            f"Numbers swapped in from another company: {v['swap']} accepted ({v['swap_derived']} with derived numbers)."
        )
        rows = [
            "| What | Result | Evidence |",
            "|---|---|---|",
            f"| Test v3: {t['tasks']} tasks by a separate author, first LLM run, DeepSeek | Agent **{t['agent']}** "
            f"{t['agent_ci']}, LLM composition {t['composition']}, deterministic workflow {t['workflow']}, LLM without "
            f"tools {t['pure']}. Agent vs workflow {t['diff']} {t['diff_ci']}, McNemar p = {t['p']}. Agent vs composition "
            f"is {'' if t['composition_passk_significant'] else 'not '}significant on pass^3. | {v3_link} |",
            f"| Verifier stress (round {v['round']}, `{v['commit']}`): {v['gold']} correct answers, {v['variants']} "
            f"corrupted ones | Corrupted answers accepted: **{v['strict']}** ({v['derived']} with derived numbers "
            f"allowed, the default). Correct answers accepted: {pct(v['true_accept'])}. {swap} | {stress_link} |",
            _slice_row(latest, "Latest", lang),
            _slice_row(previous, "Previous", lang),
            _slice_row(earlier, "Earlier", lang),
        ]
        if older:
            rows.append(
                f"| Earlier reviewer slices (chat, first run → after one round of fixes) | {older_text}. | "
                "[evaluation history](docs/evaluation-history.md#independent-sets-first-runs-vs-after-exposure) |"
            )
        rows.append(
            f"| Independent router labels ({first_router['queries']} queries, fresh set) | "
            f"{rate(first_router['accuracy'])} on the first run; {rate(router_latest['accuracy'])} after exposure | "
            f"{router_link} |"
        )
        for c in facts.claims:
            after = (
                f"; {rate(c.after)} after exposure (round-{c.after_round} checker)"
                if c.after is not None
                else "; no later run"
            )
            rows.append(
                f"| Real published claims collected blind to the checker, set v{c.version} ({c.claims}) | "
                f"{rate(c.first)} {ci2(c.first_ci)} on the first run (`{c.first_commit}`){after} | "
                f"{link_json(c.first_stem, f'evaluation/real_claims_web_v{c.version}/README.md')} |"
            )
        targeted = ""
        if r.targeted:
            parts = ", ".join(f"{mode_en.get(mode, mode)} {hits}/{runs}" for mode, hits, runs in r.targeted)
            targeted = f"; on the latest targeted set ({r.targeted_set}) {parts}"
        template = (
            f"0 successes on all {r.sets} attack sets ({r.runs:,} runs)"
            if r.successes == 0
            else f"{r.successes} successes in {r.runs:,} runs over {r.sets} attack sets"
        )
        rows.append(
            f"| Prompt injection planted in documents | Template path: {template}. LLM paths: {llm_range} of payloads "
            f"stated as fact per set in the last full run (`{r.llm_commit}`){targeted} | {red_link}, {llm_link}"
            + (f", {targeted_link}" if targeted_link else "")
            + " |"
        )
        rows.append(
            f"| Latency and load | DeepSeek agent P95 {facts.latency_p95_s:.1f} s (held-out); first token after about "
            f"{facts.ttft_p50_s:.0f} s. {facts.load_errors} errors in {facts.load_requests} requests from "
            f"{facts.load_users} users during a source chaos drill (no LLM) | [performance.md](docs/performance.md) |"
        )
        rows.append(
            f"| CI gates against committed baselines | dev {gates['dev'][0]} tasks {rate(gates['dev'][1])}, held-out "
            f"{gates['holdout'][0]} {rate(gates['holdout'][1])}, ext {gates['ext'][0]} {rate(gates['ext'][1])}, shipped "
            f"default {gates['shipped'][0]} {rate(gates['shipped'][1])} | [gate baselines]({gate_link_target}) |"
        )
    else:
        older_text = "；".join(f"第{zh_number(s.round)}轮 {rate(s.rate_before)} → {rate(s.rate_after)}" for s in older)
        swap = f"换入另一家公司的数字 {v['swap']} 放过（允许派生数字时 {v['swap_derived']}）"
        rows = [
            "| 测什么 | 结果 | 证据 |",
            "|---|---|---|",
            f"| 测试集 v3：独立作者写的 {t['tasks']} 个任务，首次 LLM 运行，DeepSeek | Agent **{t['agent']}** "
            f"{t['agent_ci']}，LLM 组织答案 {t['composition']}，确定性固定流程 {t['workflow']}，不调工具的 LLM "
            f"{t['pure']}。Agent 对固定流程 {t['diff']} {t['diff_ci']}，McNemar p = {t['p']}。Agent 对组织答案在 pass^3 上"
            f"{'显著' if t['composition_passk_significant'] else '不显著'} | {v3_link} |",
            f"| 校验器压力测试（第 {v['round']} 轮，`{v['commit']}`）：{v['gold']} 个正确答案，{v['variants']} 个篡改答案 | "
            f"篡改答案误放 **{v['strict']}**（默认允许派生数字时 {v['derived']}）；正确答案通过 "
            f"{pct(v['true_accept'])}；{swap} | {stress_link} |",
            _slice_row(latest, "最近一组", lang),
            _slice_row(previous, "上一组", lang),
            _slice_row(earlier, "更早一组", lang),
        ]
        if older:
            rows.append(
                f"| 更早的评审切片（对话，首次 → 一轮修复后） | {older_text} | "
                "[评测历史](docs/zh/evaluation-history.md#独立任务集首次运行与暴露后) |"
            )
        rows.append(
            f"| 独立路由标注（新写的 {first_router['queries']} 条） | 首次 {rate(first_router['accuracy'])}，暴露后 "
            f"{rate(router_latest['accuracy'])} | {router_link} |"
        )
        for c in facts.claims:
            after = (
                f"；之后 {rate(c.after)}（第 {c.after_round} 轮核查器，暴露后）"
                if c.after is not None
                else "；尚无后续运行"
            )
            rows.append(
                f"| 对核查器盲收集的真实公开说法，第 {c.version} 套（v{c.version}，{c.claims} 条） | 首次 {rate(c.first)} "
                f"{ci2(c.first_ci)}（`{c.first_commit}`）{after} | "
                f"{link_json(c.first_stem, f'evaluation/real_claims_web_v{c.version}/README.md')} |"
            )
        targeted = ""
        if r.targeted:
            parts = "，".join(f"{mode_zh.get(mode, mode)} {hits}/{runs}" for mode, hits, runs in r.targeted)
            targeted = f"；最近一次定向集（{r.targeted_set}）{parts}"
        template = (
            f"在 {r.sets} 个攻击集上 0 次成功（{r.runs:,} 次运行）"
            if r.successes == 0
            else f"{r.sets} 个攻击集 {r.runs:,} 次运行中成功 {r.successes} 次"
        )
        rows.append(
            f"| 投毒到文档里的提示注入 | 模板路径{template}；最近一次完整 LLM 运行（`{r.llm_commit}`）每个集合当作事实陈述 "
            f"{llm_range}{targeted} | {red_link}、{llm_link}" + (f"、{targeted_link}" if targeted_link else "") + " |"
        )
        rows.append(
            f"| 延迟与负载 | DeepSeek Agent P95 {facts.latency_p95_s:.1f} 秒（保留集），首字约 {facts.ttft_p50_s:.0f} 秒；"
            f"数据源故障演练下 {facts.load_users} 个用户（无 LLM）{facts.load_requests} 个请求 {facts.load_errors} 错误 | "
            "[性能](docs/zh/performance.md) |"
        )
        rows.append(
            f"| CI 门禁（与已提交基线比较） | 开发集 {gates['dev'][0]} 个任务 {rate(gates['dev'][1])}，保留集 "
            f"{gates['holdout'][0]} 个 {rate(gates['holdout'][1])}，ext {gates['ext'][0]} 个 {rate(gates['ext'][1])}，"
            f"发布默认配置 {gates['shipped'][0]} 个 {rate(gates['shipped'][1])} | [门禁基线]({gate_link_target}) |"
        )
    return "\n".join(rows)


def render_ci(facts: Facts, lang: str) -> str:
    p = facts.prepush
    ci = facts.ci
    zh = lang == "zh"
    dev_doc = "docs/zh/development.md" if zh else "docs/development.md"
    parts = []
    if zh:
        parts.append(
            f"推送到 `master` 之前，[`scripts/prepush.sh`](scripts/prepush.sh) 在本地跑一遍 CI 的全部步骤（[{dev_doc}]({dev_doc})）。"
        )
        if p:
            rel = p.path.relative_to(ROOT).as_posix()
            shipped = (
                f"；标记为 `shipped_default` 的测试在发布默认数据（扩展层和更正层）上再跑一遍：{p.tests('zh', shipped=True)}"
                if p.pytest_shipped
                else ""
            )
            verdict = "全部步骤通过" if p.all_passed else "至少一个步骤失败，见摘要"
            parts.append(
                f"最近一次提交的摘要是 [`{p.path.name}`]({rel})（`{p.sha}`，{p.run_at}，{verdict}）：pytest "
                f"{p.tests('zh')}{shipped}。"
            )
        if ci:
            rel = ci.path.relative_to(ROOT).as_posix()
            slowest = max(ci.jobs, key=lambda job: job[2] / job[3])
            ok = sum(1 for job in ci.jobs if job[1] == "success")
            status = (
                f"{len(ci.jobs)} 个 job 全部通过"
                if ok == len(ci.jobs)
                else f"{ok}/{len(ci.jobs)} 个 job 通过（结论 {ci.conclusion}）"
            )
            parts.append(
                f"`master` 最近一次记录的 CI 运行是 [{ci.run_id}]({ci.url})：在 `{ci.sha}` 上{status}（{ci.recorded_at} 记录于 "
                f"[`{ci.path.name}`]({rel})，由 `python scripts/record_ci_run.py` 从 GitHub API 生成）；占用预算比例最高的 job "
                f"`{slowest[0]}` 用了 {slowest[3]} 分钟里的 {slowest[2]:.1f} 分钟。页首的徽章显示 `master` 当前的状态。"
            )
        parts.append(
            "推送前检查会让所属 job 估计超过超时 60% 的步骤失败（[CI 时间预算](docs/zh/development.md#ci-时间预算)）。"
            "CI 怎么跑、人工评测工具包怎么用，见 [docs/zh/evaluation-history.md](docs/zh/evaluation-history.md#ci推送前检查与人工评测工具包)。"
        )
        return "".join(parts)
    parts.append(
        f"Before a push to `master`, [`scripts/prepush.sh`](scripts/prepush.sh) runs every CI step locally ([{dev_doc}]({dev_doc}))."
    )
    if p:
        rel = p.path.relative_to(ROOT).as_posix()
        shipped = (
            f"; the tests marked `shipped_default` again on the shipped default data (extension and corrections on): "
            f"{p.tests('en', shipped=True)}"
            if p.pytest_shipped
            else ""
        )
        verdict = "every step passed" if p.all_passed else "at least one step failed, see the summary"
        parts.append(
            f" The latest committed summary is [`{p.path.name}`]({rel}) (`{p.sha}`, {p.run_at}, {verdict}): pytest "
            f"{p.tests('en')}{shipped}."
        )
    if ci:
        rel = ci.path.relative_to(ROOT).as_posix()
        slowest = max(ci.jobs, key=lambda job: job[2] / job[3])
        ok = sum(1 for job in ci.jobs if job[1] == "success")
        status = (
            f"green, all {len(ci.jobs)} jobs"
            if ok == len(ci.jobs)
            else f"{ok} of {len(ci.jobs)} jobs green ({ci.conclusion})"
        )
        parts.append(
            f" The latest recorded CI run on `master` is [{ci.run_id}]({ci.url}): {status}, at `{ci.sha}` (recorded "
            f"{ci.recorded_at} in [`{ci.path.name}`]({rel}) by `python scripts/record_ci_run.py` from the GitHub API); "
            f"the job closest to its timeout, `{slowest[0]}`, took {slowest[2]:.1f} of its {slowest[3]} min. The badge "
            "at the top shows the current state of `master`."
        )
    parts.append(
        " The pre-push check fails a step whose job would exceed 60% of its timeout "
        "([CI time budgets](docs/development.md#ci-time-budgets)). How CI runs and how the human-evaluation kit works "
        "is described in [docs/evaluation-history.md](docs/evaluation-history.md#ci-the-pre-push-check-and-the-human-evaluation-kit)."
    )
    return "".join(parts)


def render_limits(facts: Facts, lang: str) -> str:
    latest = facts.slices[-1]
    sizes = [s.tasks for s in facts.slices]
    failing = latest.tasks - latest.after
    first_router = facts.router_first["accuracy"]
    latest_router = facts.router_latest["accuracy"]
    per_class = f"docs/agent-eval.md#{latest.anchor}"
    if lang == "en":
        lines = [
            f"- **Few unexposed sets.** Each reviewer slice is out of sample for exactly one round of fixes. The slices "
            f"are small ({min(sizes)}–{max(sizes)} chat tasks), and part of each one repeats our own test wording, as the "
            "overlap counts above show. That the engineers did not open a slice before its after-fix run is their "
            "statement. The run read the slice from the reviewer's folder, and nothing records file access "
            "([provenance](evaluation/heldout_r10/README.md#provenance-of-the-after-fix-run-what-the-record-shows-round-11-review-k16)).",
            "- **Comparisons generalise by lexicon.** The comparison grammar decides the operation from word classes that "
            "are regular expressions. Wordings outside those classes are still missed: on the latest slice (round "
            f"{latest.round}), {failing} of {latest.tasks} chat tasks still fail after the fixes ([per class]({per_class})).",
            f"- **Routing misses about {1 - first_router:.0%} of fresh questions** ({rate(first_router)} on the first run "
            f"of a fresh independent label set, {rate(latest_router)} after exposure).",
            "- **The agent loop is only slightly better than LLM composition.** With DeepSeek it is ahead per run but "
            "not on pass^3. With GLM the two are tied, and the agent costs 1.1–4x as much.",
            "- **The verifier proves traceability, not truth.** A fake price planted inside a news excerpt passes the "
            "check, because it *is* in the evidence.",
            "- **The LLM paths still sometimes state planted document content.** The output layer attributes or removes "
            "what it recognises; it cannot stop the model from mentioning a payload.",
            "- **Some results need the owner.** There are no human answer labels, no head-to-head against 问财 / 豆包 / "
            "Kimi, no user study and no owner-collected claims yet. The [kit](evaluation/human/README.md) is ready.",
        ]
    else:
        lines = [
            f"- **未暴露的任务集很少**：每组评审切片只对一轮修复是样本外的。切片规模小（{min(sizes)}–{max(sizes)} 个对话任务），"
            "而且每组都有一部分与自写测试措辞重合，见上面的重合计数。「工程师在修复后运行之前没打开过切片」是工程师的陈述：那次"
            "运行是从评审的目录读取切片的，没有任何记录能证明是否打开过（[来源说明](evaluation/heldout_r10/README.md#provenance-of-the-after-fix-run-what-the-record-shows-round-11-review-k16)）。",
            "- **比较类问题靠词表泛化**：比较语法按词类判断运算，而词类是正则表达式，词类之外的问法仍会漏掉。最近一组切片"
            f"（第{zh_number(latest.round)}轮）修复后仍有 {latest.tasks} 个里的 {failing} 个对话任务失败（[按类别]({per_class})）。",
            f"- **路由在新问法上约有 {1 - first_router:.0%} 出错**：新写的独立标注首次 {rate(first_router)}，暴露后 "
            f"{rate(latest_router)}。",
            "- **Agent 工具循环只比「LLM 组织答案」略好**：DeepSeek 上单次成功率领先，pass^3 不显著；GLM 上两者持平，Agent "
            "成本是后者的 1.1–4 倍。",
            "- **校验证明的是可追溯，不是真实**：投毒到新闻正文里的假价格能通过校验，因为它确实「在证据里」。",
            "- **LLM 路径有时仍会转述投毒文档的内容**：输出层只能对它认出的内容注明出处或删除，不能阻止模型提到投毒内容。",
            "- **有些结果需要所有者提供**：目前还没有人工答案标注，没有与问财 / 豆包 / Kimi 的对比，没有用户研究，也没有所有者"
            "收集的真实说法。[工具包](evaluation/human/README.md)已经就绪。",
        ]
    return "\n".join(lines)


RENDERERS = {
    "pitch": render_pitch,
    "headline-table": render_table,
    "ci-status": render_ci,
    "limits": render_limits,
}


def render(facts: Facts | None = None) -> dict[str, dict[str, str]]:
    facts = facts or collect()
    return {lang: {region: RENDERERS[region](facts, lang) for region in REGIONS} for lang in READMES}


def splice(text: str, regions: dict[str, str]) -> tuple[str, list[str]]:
    """Replace each region's body; returns the new text and the regions whose markers are missing."""
    missing = []
    for region, body in regions.items():
        begin, end = begin_marker(region), end_marker(region)
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
            missing.append(region)
            continue
        head, rest = text.split(begin, 1)
        _old, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text, missing


def problems(rendered: dict[str, dict[str, str]] | None = None) -> list[str]:
    """Why the committed READMEs differ from what the result files give (empty when they are up to date)."""
    if rendered is None:
        try:
            rendered = render()
        except (LookupError, KeyError, ValueError) as error:
            return [f"the README regions cannot be rendered from the committed files: {error!r}"]
    out = []
    for lang, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        updated, missing = splice(text, rendered[lang])
        for region in missing:
            out.append(
                f"{path.name}: the generated region {region!r} is missing (markers {begin_marker(region)} … {end_marker(region)})"
            )
        if updated != text:
            out.append(f"{path.name}: a generated region is out of date; run {COMMAND}")
    return out


def write(rendered: dict[str, dict[str, str]] | None = None) -> list[str]:
    rendered = rendered or render()
    messages = []
    for lang, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        updated, missing = splice(text, rendered[lang])
        for region in missing:
            messages.append(f"{path.name}: region {region!r} has no markers; not written")
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            messages.append(f"wrote {path.name}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the README headline numbers from committed result files.")
    parser.add_argument("--check", action="store_true", help="Fail if a generated README region is missing or stale.")
    args = parser.parse_args(argv)
    if args.check:
        found = problems()
        for problem in found:
            print(f"readme: {problem}")
        if found:
            return 1
        print("README.md and README_CN.md: every generated region is up to date")
        return 0
    for message in write() or ["README.md and README_CN.md: already up to date"]:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
