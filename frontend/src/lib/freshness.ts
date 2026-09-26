import { ageInDays } from "./format";
import type { EvidenceSource, Provenance } from "./types";

/**
 * Data freshness per evidence item and for a whole answer. Structured evidence carries
 * `payload.provenance` (live / live_fallback / last_known_good / snapshot, as-of date, freshness,
 * fallback reason); documents only have `as_of`, so their age decides.
 */
export type SourceMode = "live" | "fallback" | "cached" | "snapshot" | "unknown";

export interface EvidenceFreshness {
  asOf?: string;
  days?: number;
  stale: boolean;
  mode: SourceMode;
  provenance?: Provenance;
}

// Calendar days before a record is "possibly stale"; mirrors FRESHNESS_WINDOW_DAYS on the server.
const WINDOWS: Record<string, number> = {
  market_api: 10,
  industry_sql: 10,
  technical_indicators: 10,
  indicators: 10,
  valuation: 10,
  fundamental_sql: 200,
  macro_sql: 75,
  news: 30,
  announcement: 90,
  research_note: 180,
};

/** Evidence types whose values move daily: mixing their dates is what misleads readers. */
const DAILY = new Set(["market_api", "industry_sql", "technical_indicators", "indicators", "valuation"]);

function typeOf(source: EvidenceSource): string {
  if (source.evidence_id.startsWith("indicators_")) return "technical_indicators";
  return source.source_type ?? "";
}

export function staleAfterDays(source: EvidenceSource): number | undefined {
  const type = typeOf(source);
  if (type in WINDOWS) return WINDOWS[type];
  // Knowledge-base and product documents describe rules, not prices: never "stale".
  if (type === "faq" || type === "product_doc" || type === "entity") return undefined;
  return source.kind === "structured" ? 120 : 90;
}

export function provenanceOf(source: EvidenceSource): Provenance | undefined {
  const top = source.provenance;
  if (top && typeof top === "object") return top;
  const nested = source.payload?.provenance;
  return nested && typeof nested === "object" ? (nested as Provenance) : undefined;
}

function modeOf(provenance: Provenance | undefined): SourceMode {
  switch (provenance?.mode) {
    case "live":
      return provenance.fallback_reason ? "fallback" : "live";
    case "live_fallback":
      return "fallback";
    case "last_known_good":
      return "cached";
    case "snapshot":
      return "snapshot";
    default:
      if (provenance && provenance.is_live === false) return "snapshot";
      return provenance?.is_live ? "live" : "unknown";
  }
}

export function evidenceFreshness(source: EvidenceSource, now = new Date()): EvidenceFreshness {
  const provenance = provenanceOf(source);
  const asOf = source.as_of ?? provenance?.as_of ?? undefined;
  const days = ageInDays(asOf, now);
  const window = staleAfterDays(source);
  const stale = provenance?.freshness === "stale" || (days !== undefined && window !== undefined && days > window);
  return { asOf, days, stale, mode: modeOf(provenance), provenance };
}

export interface FreshnessSummary {
  /** Earliest and latest as-of date among the answer's structured evidence. */
  from?: string;
  to?: string;
  /** Days between the oldest and newest daily-moving value (price, industry, indicators). */
  spreadDays: number;
  mixed: boolean;
  stale: string[];
  snapshot: string[];
  fallback: string[];
  cached: string[];
  live: number;
  /** "warn" when data is stale, a snapshot, or dated inconsistently; "info" when only fallback sources were used. */
  level: "none" | "info" | "warn";
}

const dayOf = (value: string) => value.slice(0, 10);

/**
 * The banner for an answer. Only evidence the reader sees matters: structured data (shown as tiles
 * and numbers) and cited documents. `mixed` flags e.g. an April industry snapshot next to September
 * prices.
 */
export function summarizeFreshness(sources: EvidenceSource[], cited: Set<string>, now = new Date()): FreshnessSummary {
  const relevant = sources.filter((source) => source.kind === "structured" || cited.has(source.evidence_id));
  const summary: FreshnessSummary = { spreadDays: 0, mixed: false, stale: [], snapshot: [], fallback: [], cached: [], live: 0, level: "none" };
  const dates: string[] = [];
  const daily: number[] = [];
  for (const source of relevant) {
    const info = evidenceFreshness(source, now);
    const id = source.evidence_id;
    if (info.stale) summary.stale.push(id);
    if (info.mode === "snapshot") summary.snapshot.push(id);
    else if (info.mode === "fallback") summary.fallback.push(id);
    else if (info.mode === "cached") summary.cached.push(id);
    else if (info.mode === "live") summary.live += 1;
    if (source.kind === "structured" && info.asOf && info.days !== undefined) {
      dates.push(dayOf(info.asOf));
      if (DAILY.has(typeOf(source))) daily.push(new Date(`${dayOf(info.asOf)}T00:00:00`).getTime());
    }
  }
  dates.sort();
  summary.from = dates[0];
  summary.to = dates[dates.length - 1];
  if (daily.length >= 2) summary.spreadDays = Math.round((Math.max(...daily) - Math.min(...daily)) / 86_400_000);
  summary.mixed = summary.spreadDays > 10;
  if (summary.stale.length || summary.snapshot.length || summary.mixed) summary.level = "warn";
  else if (summary.fallback.length || summary.cached.length) summary.level = "info";
  return summary;
}
