import type { AgentResponse, ClassicResponse, StructuredItem } from "./types";

export interface PricePoint {
  time: string; // YYYY-MM-DD
  value: number;
}

export interface PriceSeries {
  evidenceId?: string;
  symbol?: string;
  name?: string;
  points: PricePoint[];
}

export type KpiTone = "up" | "down" | "neutral";
export type KpiFormat = "price" | "fundPrice" | "percent" | "percentLevel" | "ratio" | "fraction" | "money" | "number" | "volume";

export interface Kpi {
  key: string;
  /** i18n key for the label, e.g. "kpi.close". */
  label: string;
  value: number;
  format: KpiFormat;
  tone?: KpiTone;
  unit?: string;
  evidenceId?: string;
  subject?: string;
  asOf?: string;
}

export interface MarketData {
  series: PriceSeries[];
  kpis: Kpi[];
  trend?: string;
}

type Payload = Record<string, unknown>;

function num(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return Number(value);
  return undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : value == null ? undefined : String(value);
}

function isoDate(value: unknown): string | undefined {
  const raw = str(value);
  if (!raw) return undefined;
  const compact = /^(\d{4})(\d{2})(\d{2})$/.exec(raw);
  if (compact) return `${compact[1]}-${compact[2]}-${compact[3]}`;
  const iso = /^(\d{4})-(\d{1,2})-(\d{1,2})/.exec(raw);
  if (iso) return `${iso[1]}-${iso[2]!.padStart(2, "0")}-${iso[3]!.padStart(2, "0")}`;
  return undefined;
}

function tone(value: number | undefined): KpiTone {
  if (value === undefined || value === 0) return "neutral";
  return value > 0 ? "up" : "down";
}

/** Daily closes from either the agent tool payload (`recent_closes`) or provider rows (`history`). */
function seriesFrom(payload: Payload): PricePoint[] {
  const rows = (payload.recent_closes ?? payload.history) as unknown;
  if (!Array.isArray(rows)) return [];
  const byDate = new Map<string, number>();
  for (const row of rows as Payload[]) {
    const time = isoDate(row?.date ?? row?.trade_date);
    const value = num(row?.close);
    if (time && value !== undefined) byDate.set(time, value);
  }
  return [...byDate.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([time, value]) => ({ time, value }));
}

function subjectOf(payload: Payload): string | undefined {
  return str(payload.name ?? payload.canonical_name ?? payload.industry_name ?? payload.symbol);
}

function unitOf(payload: Payload, metric: string): string | undefined {
  const units = payload.metric_units as Record<string, unknown> | undefined;
  return units && typeof units === "object" ? str(units[metric]) : undefined;
}

/**
 * Turnover is CNY when the payload says so (`amount_unit`, set by the agent's tool normalisation). Raw provider
 * rows (classic `/chat`) carry no unit: Tushare's `daily.amount` is in thousands of CNY, others report CNY. The
 * provider name can sit on the payload, on the evidence item, or in the provenance (`original_source`).
 */
function amountFormat(payload: Payload, sourceName?: string): KpiFormat {
  if (str(payload.amount_unit) === "CNY") return "money";
  const provenance = (payload.provenance ?? {}) as Payload;
  const names = [payload.source_name, payload.source, payload.provider, provenance.source, provenance.original_source, sourceName];
  return names.some((name) => /tushare/i.test(String(name ?? ""))) ? "volume" : "money";
}

// Exchange-traded funds quote in 0.001 CNY (1.021), stocks in 0.01: SSE 5xxxxx and SZSE 15xxxx/16xxxx fund codes.
const FUND_SYMBOL = /^(?:5\d|1[56])\d{4}(?:\.(?:SH|SZ))?$/i;

/** Fund prices keep three decimals ("1.021", not "1.02"); other prices two (round 9, E13). */
function priceFormat(payload: Payload): KpiFormat {
  const type = str(payload.product_type)?.toLowerCase();
  if (type === "etf" || type === "fund" || type === "lof") return "fundPrice";
  if (type === "stock" || type === "index") return "price";
  return FUND_SYMBOL.test(str(payload.symbol ?? payload.ts_code) ?? "") ? "fundPrice" : "price";
}

function kpisFrom(sourceType: string, payload: Payload, evidenceId?: string, asOf?: string, sourceName?: string): Kpi[] {
  const subject = subjectOf(payload);
  const base = { evidenceId, subject, asOf };
  const out: Kpi[] = [];
  const add = (key: string, label: string, value: unknown, format: KpiFormat, extra: Partial<Kpi> = {}) => {
    const n = num(value);
    if (n !== undefined) out.push({ key: `${evidenceId ?? sourceType}:${key}`, label, value: n, format, ...base, ...extra });
  };
  const metrics = (payload.metrics as Payload | undefined) ?? payload;
  switch (sourceType) {
    case "market_api": {
      const change = num(payload.pct_change_1d);
      add("close", "kpi.close", payload.close ?? payload.latest_close, priceFormat(payload), { tone: tone(change), asOf: isoDate(payload.trade_date ?? payload.as_of) ?? asOf });
      add("pct", "kpi.change", change, "percent", { tone: tone(change) });
      add("high", "kpi.high", payload.high, priceFormat(payload));
      add("low", "kpi.low", payload.low, priceFormat(payload));
      add("amount", "kpi.amount", payload.amount, amountFormat(payload, sourceName));
      break;
    }
    case "fundamental_sql":
      add("pe", "kpi.pe", metrics.pe_ttm, "ratio", { asOf: isoDate(payload.report_date) ?? asOf });
      add("pb", "kpi.pb", metrics.pb, "ratio");
      // Agent payloads declare units (metric_units.roe = "%"); raw provider rows may hold a fraction.
      add("roe", "kpi.roe", metrics.roe, unitOf(payload, "roe") === "%" ? "percentLevel" : "fraction");
      add("revenue", "kpi.revenue", metrics.revenue, "money");
      add("net_profit", "kpi.netProfit", metrics.net_profit, "money");
      break;
    case "industry_sql":
      add("pe", "kpi.industryPe", payload.pe, "ratio", { asOf: isoDate(payload.trade_date) ?? asOf });
      add("pct", "kpi.industryChange", payload.pct_change, "percent", { tone: tone(num(payload.pct_change)) });
      break;
    case "technical_indicators":
    case "indicators":
      add("ma5", "kpi.ma5", payload.ma5, priceFormat(payload));
      add("ma20", "kpi.ma20", payload.ma20, priceFormat(payload));
      add("rsi", "kpi.rsi", payload.rsi_14, "number");
      add("vol", "kpi.volatility", payload.volatility_20d, "fraction");
      break;
    case "macro_sql":
    case "macro":
      add("macro", "kpi.macro", payload.metric_value ?? payload.value, "number", {
        subject: str(payload.indicator_name ?? payload.indicator_code) ?? subject,
        unit: str(payload.unit),
        asOf: isoDate(payload.metric_date ?? payload.period_end ?? payload.as_of) ?? asOf,
      });
      break;
    default:
      break;
  }
  return out;
}

const SYMBOL_IN_ID = /_(\d{6}\.(?:SH|SZ|BJ))$/i;
// "五粮液 (000858.SZ) fundamentals": the agent's structured evidence titles name the company and its symbol.
const NAME_IN_TITLE = /^(.+?) \((\d{6}\.(?:SH|SZ|BJ))\)/i;

/** The company named in an agent evidence title, or undefined. */
export function nameInTitle(title: string | null | undefined): string | undefined {
  return NAME_IN_TITLE.exec(title ?? "")?.[1];
}

function collect(items: StructuredItem[]): MarketData {
  const series: PriceSeries[] = [];
  const kpis: Kpi[] = [];
  const names = new Map<string, string>(); // symbol -> name, from payloads that carry both
  for (const item of items) {
    const payload = item.payload;
    if (!payload || typeof payload !== "object") continue;
    const type = item.evidence_id?.startsWith("indicators_") ? "technical_indicators" : (item.source_type ?? "");
    const symbol = str(payload.symbol);
    const name = str(payload.name ?? payload.canonical_name);
    if (symbol && name && name !== symbol) names.set(symbol.toUpperCase(), name);
    const titled = NAME_IN_TITLE.exec(item.title ?? "");
    if (titled?.[1] && titled[2] && !names.has(titled[2].toUpperCase())) names.set(titled[2].toUpperCase(), titled[1]);
    const points = seriesFrom(payload);
    if (points.length >= 2) {
      series.push({ evidenceId: item.evidence_id, symbol, name: subjectOf(payload), points });
    }
    kpis.push(...kpisFrom(type, payload, item.evidence_id, isoDate(item.as_of), item.source_name ?? undefined));
  }
  // Fundamentals payloads may hold only the metrics: name their tiles from the evidence id's symbol, so a
  // company's price and valuation tiles group together (see selectKpis).
  // A payload with a symbol but no name ("600519.SH") gets the name another payload gives for it.
  for (const kpi of kpis) {
    const symbol = kpi.subject ?? SYMBOL_IN_ID.exec(kpi.evidenceId ?? "")?.[1];
    if (symbol) kpi.subject = names.get(symbol.toUpperCase()) ?? symbol;
  }
  return { series, kpis };
}

// The tiles worth showing first, in order: a comparison shows the same ones for every company.
const PRIORITY = [
  "kpi.close",
  "kpi.change",
  "kpi.pe",
  "kpi.pb",
  "kpi.roe",
  "kpi.revenue",
  "kpi.netProfit",
  "kpi.amount",
  "kpi.high",
  "kpi.low",
];

const rank = (kpi: Kpi) => {
  const index = PRIORITY.indexOf(kpi.label);
  return index < 0 ? PRIORITY.length : index;
};

/**
 * At most `max` tiles. One subject: the first `max`. Several (a comparison, "茅台和五粮液对比"): an equal
 * share per subject, the metrics they have in common first and in the same order, grouped by subject, so
 * each company gets a row of comparable tiles instead of the first company taking every slot.
 */
export function selectKpis(kpis: Kpi[], max: number): Kpi[] {
  const groups = new Map<string, Kpi[]>();
  for (const kpi of kpis) {
    const key = kpi.subject ?? "";
    groups.set(key, [...(groups.get(key) ?? []), kpi]);
  }
  // Companies (price or fundamentals tiles) share the slots; industry and macro tiles fill what is left.
  const companies = [...groups.values()].filter((tiles) => tiles.some((kpi) => PRIORITY.includes(kpi.label)));
  if (companies.length < 2) return kpis.slice(0, max);
  const share = Math.max(1, Math.floor(max / companies.length));
  const seen = new Map<string, number>();
  for (const tiles of companies) {
    for (const label of new Set(tiles.map((kpi) => kpi.label))) seen.set(label, (seen.get(label) ?? 0) + 1);
  }
  const shared = (kpi: Kpi) => ((seen.get(kpi.label) ?? 0) > 1 ? 0 : 1);
  const picked: Kpi[] = [];
  for (const tiles of companies) {
    const ordered = tiles
      .map((kpi, index) => ({ kpi, index }))
      .sort((a, b) => shared(a.kpi) - shared(b.kpi) || rank(a.kpi) - rank(b.kpi) || a.index - b.index)
      .map(({ kpi }) => kpi);
    picked.push(...ordered.slice(0, share));
  }
  const rest = [...groups.values()].filter((tiles) => !companies.includes(tiles)).flat();
  return [...picked, ...rest].slice(0, max);
}

/** Structured data for charts/KPI tiles from an agent response (only when payloads are included). */
export function marketDataFromAgent(response: AgentResponse): MarketData {
  const items = (response.evidence_sources ?? [])
    .filter((source) => source.payload)
    .map((source) => ({
      evidence_id: source.evidence_id,
      source_type: source.source_type ?? undefined,
      source_name: source.source_name ?? undefined,
      as_of: source.as_of,
      title: source.title,
      payload: source.payload,
    }));
  return collect(items);
}

/** Structured data from the classic `/chat` response (`retrieval_result.structured_data`). */
export function marketDataFromClassic(response: ClassicResponse): MarketData {
  const retrieval = response.retrieval_result ?? {};
  const data = collect(retrieval.structured_data ?? []);
  const summary = retrieval.analysis_summary as Payload | null | undefined;
  const signal = summary?.market_signal;
  const first = (Array.isArray(signal) ? signal[0] : signal) as Payload | undefined;
  if (first) {
    data.trend = str(first.trend_signal);
    const known = new Set(data.kpis.map((kpi) => kpi.label));
    const extra = kpisFrom("technical_indicators", first, undefined, isoDate(first.trade_date)).filter(
      (kpi) => !known.has(kpi.label),
    );
    data.kpis.push(...extra);
  }
  return data;
}

export function hasMarketData(data: MarketData | undefined): data is MarketData {
  return Boolean(data && (data.series.length || data.kpis.length));
}
