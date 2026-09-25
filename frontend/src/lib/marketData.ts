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
export type KpiFormat = "price" | "percent" | "ratio" | "fraction" | "money" | "number" | "volume";

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

function kpisFrom(sourceType: string, payload: Payload, evidenceId?: string, asOf?: string): Kpi[] {
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
      add("close", "kpi.close", payload.close ?? payload.latest_close, "price", { tone: tone(change), asOf: isoDate(payload.trade_date ?? payload.as_of) ?? asOf });
      add("pct", "kpi.change", change, "percent", { tone: tone(change) });
      add("high", "kpi.high", payload.high, "price");
      add("low", "kpi.low", payload.low, "price");
      // Tushare reports `amount` in thousands of CNY; AKShare/efinance report CNY.
      const thousands = /tushare/i.test(String(payload.source_name ?? payload.provider ?? ""));
      add("amount", "kpi.amount", payload.amount, thousands ? "volume" : "money");
      break;
    }
    case "fundamental_sql":
      add("pe", "kpi.pe", metrics.pe_ttm, "ratio", { asOf: isoDate(payload.report_date) ?? asOf });
      add("pb", "kpi.pb", metrics.pb, "ratio");
      add("roe", "kpi.roe", metrics.roe, "fraction");
      add("revenue", "kpi.revenue", metrics.revenue, "money");
      add("net_profit", "kpi.netProfit", metrics.net_profit, "money");
      break;
    case "industry_sql":
      add("pe", "kpi.industryPe", payload.pe, "ratio", { asOf: isoDate(payload.trade_date) ?? asOf });
      add("pct", "kpi.industryChange", payload.pct_change, "percent", { tone: tone(num(payload.pct_change)) });
      break;
    case "technical_indicators":
    case "indicators":
      add("ma5", "kpi.ma5", payload.ma5, "price");
      add("ma20", "kpi.ma20", payload.ma20, "price");
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

function collect(items: StructuredItem[]): MarketData {
  const series: PriceSeries[] = [];
  const kpis: Kpi[] = [];
  for (const item of items) {
    const payload = item.payload;
    if (!payload || typeof payload !== "object") continue;
    const type = item.evidence_id?.startsWith("indicators_") ? "technical_indicators" : (item.source_type ?? "");
    const points = seriesFrom(payload);
    if (points.length >= 2) {
      series.push({ evidenceId: item.evidence_id, symbol: str(payload.symbol), name: subjectOf(payload), points });
    }
    kpis.push(...kpisFrom(type, payload, item.evidence_id, isoDate(item.as_of)));
  }
  return { series, kpis };
}

/** Structured data for charts/KPI tiles from an agent response (only when payloads are included). */
export function marketDataFromAgent(response: AgentResponse): MarketData {
  const items = (response.evidence_sources ?? [])
    .filter((source) => source.payload)
    .map((source) => ({
      evidence_id: source.evidence_id,
      source_type: source.source_type ?? undefined,
      as_of: source.as_of,
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
