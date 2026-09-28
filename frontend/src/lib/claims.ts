import { formatKpi, formatMoney } from "./format";
import type { Lang, MessageKey, Translate } from "./i18n";
import type { ClaimCheckItem, ClaimReport, ClaimStatus, EvidenceSource } from "./types";

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
 * The report only carries the bare number, so the unit is read back from the claim text.
 */
export function claimedAmount(claim: string, value: number): number {
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
      // Claims state ROE in percent ("ROE 50%"); the fundamentals payload stores a fraction (0.33).
      return side === "claimed" ? `${trim(value, 1)}%` : formatKpi(lang, value, "fraction");
    case "revenue":
    case "net_profit": {
      const amount = side === "claimed" ? claimedAmount(claim, value) : value;
      // "1,741.2 亿元" / "174.12B CNY"
      return `${formatMoney(lang, amount)}${lang === "zh" ? "" : " "}${t("claim.unit.yuan")}`;
    }
    default:
      return trim(value, 4);
  }
}

const NOTES: Record<string, MessageKey> = {
  "no listed company, fund or index": "claim.note.noTarget",
  "metric not recognised": "claim.note.noMetric",
  "no data for this metric": "claim.note.noData",
};

/** The checker's English notes, localised; unknown notes are shown as written. */
export function noteText(t: Translate, note: string | null | undefined): string {
  if (!note) return "";
  const key = NOTES[note.trim().toLowerCase()];
  return key ? t(key) : note;
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
  };
}
