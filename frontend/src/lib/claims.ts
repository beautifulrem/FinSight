import { formatKpi, formatMoney } from "./format";
import type { Lang, MessageKey, Translate } from "./i18n";
import type { ClaimCheckItem, ClaimComparator, ClaimReason, ClaimReport, ClaimStatus, EvidenceSource } from "./types";

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
      // Claims state these in percent ("ROE 50%"); a payload may store a fraction (0.33).
      return side === "claimed" ? `${trim(value, 1)}%` : formatKpi(lang, value, "fraction");
    case "revenue_yoy":
    case "netprofit_yoy":
      // Growth is in percent on both sides ("同比增长16%", revenue_yoy 1.47).
      return formatKpi(lang, value, "percent");
    case "eps":
      return `${trim(value)} ${t("claim.unit.yuan")}`;
    case "revenue":
    case "net_profit":
    case "market_cap": {
      const amount = side === "claimed" ? claimedAmount(claim, value, unit) : value;
      // "1,741.2 亿元" / "174.12B CNY"
      return `${formatMoney(lang, amount)}${lang === "zh" ? "" : " "}${t("claim.unit.yuan")}`;
    }
    default:
      return trim(value, 4);
  }
}

const SYMBOLS: Record<ClaimComparator, string> = { eq: "", ne: "≠", gt: ">", ge: "≥", lt: "<", le: "≤", approx: "≈", range: "" };

/**
 * The claimed side of a check with its comparator: "> 30%", "≠ 15 倍", "≈ 25 倍", "20 – 30 倍", "< 0%".
 * `label` is the same in words for screen readers ("高于 30%").
 */
export function claimedText(
  lang: Lang,
  t: Translate,
  check: ClaimCheckItem,
  claim = "",
): { text: string; label: string } {
  const value = (number: number) => formatClaimValue(lang, t, check.metric, number, "claimed", claim, check.claimed_unit);
  const comparator = check.comparator ?? "eq";
  if (comparator === "range" && check.claimed_high !== null && check.claimed_high !== undefined) {
    const span = `${value(check.claimed)} – ${value(check.claimed_high)}`;
    const word = t(check.negated ? "claim.cmp.outside" : "claim.cmp.range");
    return { text: check.negated ? `∉ ${span}` : span, label: `${word} ${span}` };
  }
  const text = value(check.claimed);
  const symbol = SYMBOLS[comparator] ?? "";
  if (!symbol) return { text, label: text };
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
};

/** The checker's reason (or English note), localised; unknown notes are shown as written. */
export function noteText(t: Translate, note: string | null | undefined, check?: ClaimCheckItem): string {
  const date = check?.as_of ?? "";
  const reason = check?.reason ? REASONS[check.reason] : undefined;
  if (reason) return t(reason, { date });
  if (!note) return "";
  if (note.startsWith("compared with the report for the period ending")) return t("claim.note.interim", { date });
  const key = NOTES[note.trim().toLowerCase()];
  return key ? t(key) : note;
}

// "听说茅台市盈率只有15倍，是真的吗": a hearsay cue or a "is it true" question around a number or a move.
const CLAIM_CUE =
  /听说|据说|传言|传闻|有人说|网上说|听人说|据传|号称|是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实|靠谱吗|\bis it true\b|\bi heard\b|\bsomeone said\b|\brumou?r\b|\bis (?:that|this) (?:true|right)\b/i;
const CLAIM_CONTENT =
  /\d|[一二两三四五六七八九十]+(?:点[〇零一二三四五六七八九]+)?(?:倍|成|%|元|亿)|涨了|跌了|大涨|大跌|涨停|跌停|\b(?:rose|fell)\b/i;
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

export function statusCounts(report: ClaimReport): Record<ClaimStatus, number> {
  const counts: Record<ClaimStatus, number> = { supported: 0, contradicted: 0, unverifiable: 0 };
  for (const check of report.checks) counts[check.status] = (counts[check.status] ?? 0) + 1;
  return counts;
}

/** An evidence record for a check, so its date shows age and "may be stale" like the evidence ledger. */
export function checkEvidence(check: ClaimCheckItem, report: ClaimReport): EvidenceSource | null {
  if (!check.evidence_id) return null;
  const listed = report.evidence_sources?.find((item) => item.evidence_id === check.evidence_id);
  const id = check.evidence_id;
  return {
    evidence_id: id,
    kind: "structured",
    source_type: id.startsWith("price_") ? "market_api" : id.startsWith("fundamental_") ? "fundamental_sql" : null,
    source_name: check.source ?? listed?.source_name ?? null,
    title: listed?.title ?? null,
    as_of: check.as_of ?? listed?.as_of ?? null,
    provenance: listed?.provenance ?? undefined,
  };
}
