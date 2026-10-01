import { formatKpi, formatMoney } from "./format";
import type { Lang, MessageKey, Translate } from "./i18n";
import type {
  ClaimCheckItem,
  ClaimComparator,
  ClaimHeadline,
  ClaimReason,
  ClaimReport,
  ClaimStatus,
  EvidenceSource,
} from "./types";

/**
 * Display helpers for `POST /agent/claim-check` reports: claimed and actual values in the metric's unit,
 * readable notes, and an evidence record so the as-of date gets the same freshness styling as the
 * evidence ledger.
 */

const trim = (value: number, digits = 2) =>
  Number(value.toFixed(digits)).toLocaleString("en-US", { maximumFractionDigits: digits });

// Chinese and English money units in a claim ("1700亿", "174 billion"): multiplier to CNY.
const MONEY_UNITS: [RegExp, number][] = [
  [/^\s*万亿/, 1e12],
  [/^\s*亿/, 1e8],
  [/^\s*万/, 1e4],
  [/^\s*(?:trillion|tn)\b/i, 1e12],
  [/^\s*(?:billion|bn)\b/i, 1e9],
  [/^\s*(?:million|mn|m)\b/i, 1e6],
];

/**
 * The claimed amount in CNY, using the unit written after the number in the claim ("营收1700亿" → 1.7e11).
 * Uses the report's ``claimed_unit`` when present, otherwise reads the unit back from the claim text.
 */
export function claimedAmount(claim: string, value: number, unit?: string | null): number {
  if (unit) {
    for (const [pattern, scale] of MONEY_UNITS) if (pattern.test(unit)) return value * scale;
    if (/^(?:元|yuan)$/i.test(unit)) return value;
  }
  const digits = String(Math.abs(value)).replace(/\.0+$/, "");
  const pattern = new RegExp(`${digits.replace(".", "\\.")}(?:\\.0+)?`, "g");
  for (const match of claim.replace(/,/g, "").matchAll(pattern)) {
    const tail = claim.replace(/,/g, "").slice((match.index ?? 0) + match[0].length);
    for (const [unit, scale] of MONEY_UNITS) if (unit.test(tail)) return value * scale;
  }
  return value;
}

/** A claimed or actual value in the metric's unit: "24.6 倍", "-0.18%", "1409.5 元", "1,741.2 亿". */
export function formatClaimValue(
  lang: Lang,
  t: Translate,
  metric: string | null | undefined,
  value: number,
  side: "claimed" | "actual",
  claim = "",
  unit?: string | null,
): string {
  switch (metric) {
    case "close":
      return `${trim(value)} ${t("claim.unit.yuan")}`;
    case "pct_change_1d":
      return formatKpi(lang, value, "percent");
    case "pe_ttm":
    case "pe":
    case "pb":
      return lang === "zh" ? `${trim(value)} ${t("claim.unit.times")}` : `${trim(value)}${t("claim.unit.times")}`;
    case "roe":
    case "gross_margin":
    case "net_margin":
    case "debt_ratio":
    case "dividend_yield":
      // Claims state these in percent ("ROE 50%"); tool payloads are normalised to percent (tools/units.py).
      return side === "claimed" ? `${trim(value, 1)}%` : formatKpi(lang, value, "percentLevel");
    case "revenue_yoy":
    case "netprofit_yoy":
    case "cpi_yoy":
    case "ppi_yoy":
    case "m2_yoy":
    case "gdp_yoy":
      // Growth is in percent on both sides ("同比增长16%", revenue_yoy 1.47, CPI YoY 0.8).
      return formatKpi(lang, value, "percent");
    case "cn10y":
    case "lpr_1y":
    case "lpr_5y":
      // A rate level: "2.31%", no sign.
      return `${trim(value)}%`;
    case "pmi":
      return trim(value, 1);
    case "eps":
      return `${trim(value)} ${t("claim.unit.yuan")}`;
    case "revenue":
    case "net_profit":
    case "amount":
    case "market_cap": {
      const amount = side === "claimed" ? claimedAmount(claim, value, unit) : value;
      // "1,741.2 亿元" / "174.12B CNY"
      return `${formatMoney(lang, amount)}${lang === "zh" ? "" : " "}${t("claim.unit.yuan")}`;
    }
    default:
      return trim(value, 4);
  }
}

// Metrics quoted in percent: a difference of them is in percentage points ("高出3.6个百分点").
const RATE_METRICS = new Set([
  "roe",
  "gross_margin",
  "net_margin",
  "debt_ratio",
  "dividend_yield",
  "pct_change_1d",
  "revenue_yoy",
  "netprofit_yoy",
]);

const isDifference = (check: ClaimCheckItem) => check.kind === "difference" || check.kind === "relative_difference";

/**
 * A difference in the metric's unit (round 10): "+3.6 个百分点" for ROE, "+3.7 倍" for P/E, "-9.89%" relative to the
 * other side, "+445.2 亿元" for net profit. `signed` writes "+" for a positive value (a stated direction).
 */
export function formatDifference(
  lang: Lang,
  t: Translate,
  check: ClaimCheckItem,
  value: number,
  side: "claimed" | "actual",
  claim = "",
  signed = true,
): string {
  const sign = value < 0 ? "−" : signed ? "+" : "";
  const size = Math.abs(value);
  if (check.kind === "relative_difference") return `${sign}${trim(size)}%`;
  if (check.metric && RATE_METRICS.has(check.metric)) {
    return `${sign}${trim(size)} ${t("claim.unit.pp")}`;
  }
  return `${sign}${formatClaimValue(lang, t, check.metric, size, side, claim, check.claimed_unit)}`;
}

const known = <T,>(value: T | null | undefined): value is T => value !== null && value !== undefined;

/** A multiple of another target ("市净率是五粮液的1.5倍"): `claimed` is the multiple, `ratio` the actual one. */
const isMultiple = (check: ClaimCheckItem) => !isDifference(check) && Boolean(check.reference) && known(check.claimed);

/**
 * The actual side of a check: the target's value; for a difference the actual difference of the two; for a multiple
 * the actual ratio (round 12: compared like for like with the claimed multiple).
 */
export function actualText(lang: Lang, t: Translate, check: ClaimCheckItem): string | null {
  if (isDifference(check) && known(check.difference)) {
    return formatDifference(lang, t, check, check.difference, "actual", "", known(check.direction));
  }
  if (isMultiple(check) && known(check.ratio)) return `${trim(check.ratio)}×`;
  if (!known(check.actual)) return null;
  return formatClaimValue(lang, t, check.metric, check.actual, "actual");
}

/**
 * The data behind the actual side of a relational check (round 12, H14): the other side's value for a relation
 * ("五粮液 20.9 倍"), both values for a multiple or a difference ("贵州茅台 33% · 五粮液 29.4%"). These are values
 * from the sources, so they sit under "actual", never under "claimed".
 */
export function actualDetail(
  lang: Lang,
  t: Translate,
  check: ClaimCheckItem,
  name: (value: string) => string = (value) => value,
): string | undefined {
  if (!check.reference || !known(check.reference_value)) return undefined;
  const value = (number: number) => formatClaimValue(lang, t, check.metric, number, "actual");
  const other = `${name(check.reference)} ${value(check.reference_value)}`;
  const relation = !isDifference(check) && !known(check.claimed);
  if (relation || !check.target || !known(check.actual)) return other;
  return `${name(check.target)} ${value(check.actual)} · ${other}`;
}

const SYMBOLS: Record<ClaimComparator, string> = { eq: "", ne: "≠", gt: ">", ge: "≥", lt: "<", le: "≤", approx: "≈", range: "" };

const BOUNDS = new Set<ClaimComparator>(["gt", "ge", "lt", "le", "range"]);

/**
 * The claimed side of a check with its comparator: "> 30%", "≠ 15 倍", "≈ 25 倍", "20 – 30 倍", "< 0%".
 * A bound on a move is about its size in the stated direction: "跌超1%" is "跌幅 > 1%" ("Fall > 1%"), not
 * "> -1%". A relation reads subject, relation, object (round 12, H14): "茅台PE比五粮液高" is "贵州茅台 > 五粮液",
 * "a multiple" is "贵州茅台 ≈ 1.5× 五粮液", and a difference names its two sides in `detail` ("贵州茅台 − 五粮液").
 * The other side's value is data, shown under the actual side (`actualDetail`).
 * `label` is the same in words for screen readers ("高于 30%", "跌幅高于 1%").
 */
export function claimedText(
  lang: Lang,
  t: Translate,
  check: ClaimCheckItem,
  claim = "",
  name: (value: string) => string = (value) => value,
): { text: string; label: string; detail?: string } {
  const value = (number: number) => formatClaimValue(lang, t, check.metric, number, "claimed", claim, check.claimed_unit);
  const comparator = check.comparator ?? "eq";
  const symbol = SYMBOLS[comparator] || "=";
  const subject = check.target ? name(check.target) : t("claim.unknownTarget");
  if (isDifference(check) && check.claimed !== null && check.claimed !== undefined) {
    // "茅台ROE比五粮液高出约3.6个百分点": "差值 ≈ +3.6 个百分点", the other side's value and the actual difference below
    const reference = check.reference ? name(check.reference) : t("claim.unknownTarget");
    const signed = check.direction !== null && check.direction !== undefined;
    const diff = (number: number) => formatDifference(lang, t, check, number, "claimed", claim, signed);
    const bounded = check.claimed_high !== null && check.claimed_high !== undefined && comparator === "gt";
    const words = bounded ? `${diff(check.claimed)} – ${diff(check.claimed_high!)}` : diff(check.claimed);
    const stated = bounded || !SYMBOLS[comparator] ? words : `${SYMBOLS[comparator]} ${words}`;
    const word = t(bounded ? "claim.cmp.range" : (`claim.cmp.${comparator}` as MessageKey));
    return {
      text: `${t("claim.difference")} ${stated}`,
      label: `${t("claim.difference")} (${subject} − ${reference}) ${word} ${words}`,
      detail: `${subject} − ${reference}`,
    };
  }
  if (check.reference && check.claimed !== null && check.claimed !== undefined) {
    // "市净率是五粮液的1.5倍": "贵州茅台 ≈ 1.5× 五粮液"; the actual ratio and both values are on the actual side
    const reference = name(check.reference);
    const multiple = `${trim(check.claimed)}×`;
    const word = t(`claim.cmp.${comparator}` as MessageKey);
    return {
      text: `${subject} ${SYMBOLS[comparator] || "="} ${multiple} ${reference}`,
      label: `${subject} ${word} ${multiple} ${reference}`,
    };
  }
  if (check.claimed === null || check.claimed === undefined) {
    // "茅台PE比五粮液高": "贵州茅台 > 五粮液" (subject, relation, object), never a bare "> 五粮液" under the subject
    const reference = check.reference ? name(check.reference) : t("claim.unknownTarget");
    const word = t(`claim.cmp.${comparator}` as MessageKey);
    return { text: `${subject} ${symbol} ${reference}`, label: `${subject} ${word} ${reference}` };
  }
  if (check.direction && BOUNDS.has(comparator)) {
    const move = t(`claim.move.${check.direction}` as MessageKey);
    const size = (number: number) => value(Math.abs(number)).replace(/^\+/, "");
    const gap = lang === "zh" ? "" : " ";
    if (comparator === "range" && check.claimed_high !== null && check.claimed_high !== undefined) {
      const [low, high] = [Math.abs(check.claimed), Math.abs(check.claimed_high)].sort((a, b) => a - b) as [number, number];
      const span = `${size(low)} – ${size(high)}`;
      const word = t(check.negated ? "claim.cmp.outside" : "claim.cmp.range");
      return { text: `${move} ${check.negated ? "∉ " : ""}${span}`, label: `${move}${gap}${word} ${span}` };
    }
    const bound = size(check.claimed);
    const word = t(`claim.cmp.${comparator}` as MessageKey);
    if (comparator === "gt" && check.claimed_high !== null && check.claimed_high !== undefined) {
      const high = size(check.claimed_high);
      const span = `${bound} – ${high}`;
      return { text: `${move} > ${bound}, < ${high}`, label: `${move}${gap}${t("claim.cmp.range")} ${span}` };
    }
    return { text: `${move} ${SYMBOLS[comparator]} ${bound}`, label: `${move}${gap}${word} ${bound}` };
  }
  if (comparator === "gt" && check.claimed_high !== null && check.claimed_high !== undefined) {
    // "八百多亿" (round 10): more than 800亿 and less than 900亿
    const span = `${value(check.claimed)} – ${value(check.claimed_high)}`;
    return { text: `> ${value(check.claimed)}, < ${value(check.claimed_high)}`, label: `${t("claim.cmp.range")} ${span}` };
  }
  if (comparator === "range" && check.claimed_high !== null && check.claimed_high !== undefined) {
    const span = `${value(check.claimed)} – ${value(check.claimed_high)}`;
    const word = t(check.negated ? "claim.cmp.outside" : "claim.cmp.range");
    return { text: check.negated ? `∉ ${span}` : span, label: `${word} ${span}` };
  }
  const text = value(check.claimed);
  if (!SYMBOLS[comparator]) return { text, label: text };
  return { text: `${symbol} ${text}`, label: `${t(`claim.cmp.${comparator}` as MessageKey)} ${text}` };
}

const NOTES: Record<string, MessageKey> = {
  "no listed company, fund or index": "claim.note.noTarget",
  "metric not recognised": "claim.note.noMetric",
  "no data for this metric": "claim.note.noData",
};

const REASONS: Record<ClaimReason, MessageKey> = {
  no_target: "claim.note.noTarget",
  no_metric: "claim.note.noMetric",
  no_data: "claim.note.noData",
  growth_unavailable: "claim.note.growthUnavailable",
  unit_mismatch: "claim.note.unitMismatch",
  no_unit: "claim.note.noUnit",
  forecast: "claim.note.forecast",
  period_mismatch: "claim.note.periodMismatch",
  multi_day: "claim.note.multiDay",
  no_reference: "claim.note.noReference",
};

/** The checker's reason (or English note), localised; unknown notes are shown as written. */
export function noteText(
  t: Translate,
  note: string | null | undefined,
  check?: ClaimCheckItem,
  lang: Lang = "zh",
  target = "",
): string {
  const date = check?.as_of ?? "";
  const reason = check?.reason ? REASONS[check.reason] : undefined;
  if (reason) return t(reason, { date });
  if (check && isDifference(check) && check.difference !== null && check.difference !== undefined) {
    const signed = check.direction !== null && check.direction !== undefined;
    const value = formatDifference(lang, t, check, check.difference, "actual", "", true);
    const text = t(check.kind === "relative_difference" ? "claim.note.relativeDifference" : "claim.note.difference", {
      value,
    });
    return signed ? text : `${text}${lang === "zh" ? "；" : ". "}${t("claim.note.differenceSize")}`;
  }
  if (check?.kind === "stated_reference" && !note) return t("claim.note.stated", { target: target || check.target || "" });
  if (!note) return "";
  if (note.startsWith("compared with the report for the period ending")) return t("claim.note.interim", { date });
  const ratio = /^ratio ([\d.]+) = /.exec(note);
  if (ratio) return t("claim.note.ratio", { ratio: ratio[1] ?? "" });
  if (note === "the two moves are not both in the stated direction") return t("claim.note.ratioDirection");
  const convention = /^convention: '(.+)' means a move of (at least|less than) ([\d.]+%)/.exec(note);
  if (convention) {
    const size = t(convention[2] === "at least" ? "claim.size.atLeast" : "claim.size.lessThan", {
      value: convention[3] ?? "",
    });
    return t("claim.note.convention", { word: convention[1] ?? "", size });
  }
  const key = NOTES[note.trim().toLowerCase()];
  return key ? t(key) : note;
}

// "听说茅台市盈率只有15倍，是真的吗": a hearsay cue or a "is it true" question around a number or a move.
const CLAIM_CUE =
  /听说|据说|传言|传闻|有人说|网上说|听人说|据传|号称|是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实|靠谱吗|\bis it true\b|\bi heard\b|\bsomeone said\b|\brumou?r\b|\bis (?:that|this) (?:true|right)\b/i;
// Mirrors query_intelligence/agent/hearsay.py, which checks the claim for the answer's inline fact check.
const CLAIM_CONTENT =
  /\d|[一二两三四五六七八九十]+(?:点[〇零一二三四五六七八九]+)?(?:倍|成|%|元|亿)|涨了|跌了|大涨|大跌|涨停|跌停|比.{1,12}(?:高|低|贵|便宜|多|少)|高于|低于|\b(?:rose|fell|higher than|lower than)\b/i;
const LEADING_CUE = /^\s*(?:我)?(?:听说|据说|传言|传闻|有人说|网上说|听人说|据传|I heard(?: that)?|someone said(?: that)?|is it true(?: that)?)[，,：:\s]*/i;
const TRAILING_QUESTION =
  /[，,。\s]*(?:这|这个|这话|这是)?(?:是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实吗?|靠谱吗|,?\s*is (?:that|this|it) (?:true|right))?\s*[？?！!。.]*\s*$/i;

/**
 * The claim inside a chat message that reads like hearsay to verify, or null. "听说茅台市盈率只有15倍，是真的吗？"
 * → "茅台市盈率只有15倍". The chat only suggests the fact-check view; it does not change the backend route.
 */
export function claimInMessage(message: string): string | null {
  const text = message.trim();
  if (text.length < 4 || !CLAIM_CUE.test(text) || !CLAIM_CONTENT.test(text)) return null;
  const claim = text.replace(LEADING_CUE, "").replace(TRAILING_QUESTION, "").trim();
  return claim.length >= 2 ? claim : null;
}

/** Chinese target name → the English name the server gave (`targets[].name_en`), for the English UI. */
export function targetName(lang: Lang, report: ClaimReport): (name: string) => string {
  if (lang !== "en") return (name) => name;
  const names = new Map<string, string>(Object.entries(report.labels_en ?? {}));
  for (const target of report.targets ?? []) {
    if (target.name && target.name_en) names.set(target.name, target.name_en);
  }
  return (name) => names.get(name) ?? name;
}

export function statusCounts(report: ClaimReport): Record<ClaimStatus | "unchecked", number> {
  const counts: Record<ClaimStatus | "unchecked", number> = { supported: 0, contradicted: 0, unverifiable: 0, unchecked: 0 };
  for (const check of report.checks) counts[check.status] = (counts[check.status] ?? 0) + 1;
  counts.unchecked = report.unchecked?.length ?? 0;
  return counts;
}

/**
 * The headline of a report. "Supported" means every *check* agreed; when parts of the claim were not checked (or a
 * server without `coverage` lists unchecked clauses) the headline is "partly checked" instead, so the card never says
 * "all the numbers agree" about a claim whose other parts nobody looked at (round 9, E2).
 */
export function claimHeadline(report: ClaimReport): ClaimHeadline {
  if (report.verdict !== "supported") return report.verdict;
  const counts = statusCounts(report);
  const partial = report.coverage ? report.coverage === "partial" : counts.unchecked + counts.unverifiable > 0;
  return partial ? "partly_checked" : "supported";
}

/** Every part of the claim the card lists: one row per check plus one per clause that was not checked. */
export function partCount(report: ClaimReport): number {
  return report.checks.length + (report.unchecked?.length ?? 0);
}

/** An evidence record for a check, so its date shows age and "may be stale" like the evidence ledger. */
export function checkEvidence(check: ClaimCheckItem, report: ClaimReport): EvidenceSource | null {
  if (!check.evidence_id) return null;
  const listed = report.evidence_sources?.find((item) => item.evidence_id === check.evidence_id);
  const id = check.evidence_id;
  return {
    evidence_id: id,
    kind: "structured",
    source_type: id.startsWith("price_")
      ? "market_api"
      : id.startsWith("fundamental_")
        ? "fundamental_sql"
        : id.startsWith("macro_")
          ? "macro_sql"
          : id.startsWith("industry_")
            ? "industry_sql"
            : null,
    source_name: check.source ?? listed?.source_name ?? null,
    title: listed?.title ?? null,
    as_of: check.as_of ?? listed?.as_of ?? null,
    provenance: listed?.provenance ?? undefined,
  };
}
