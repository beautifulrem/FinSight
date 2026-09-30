import type { Lang } from "./i18n";
import type { KpiFormat } from "./marketData";

const locale = (lang: Lang) => (lang === "zh" ? "zh-CN" : "en-US");

export function formatMs(ms: number | undefined | null): string {
  if (ms == null || !Number.isFinite(ms)) return "–";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function formatInt(lang: Lang, value: number | undefined | null): string {
  if (value == null) return "–";
  return new Intl.NumberFormat(locale(lang)).format(value);
}

/** Large CNY amounts: 亿/万 in Chinese, B/M in English. */
export function formatMoney(lang: Lang, value: number): string {
  const abs = Math.abs(value);
  if (lang === "zh") {
    if (abs >= 1e8) return `${trim(value / 1e8)} 亿`;
    if (abs >= 1e4) return `${trim(value / 1e4)} 万`;
    return trim(value);
  }
  if (abs >= 1e9) return `${trim(value / 1e9)}B`;
  if (abs >= 1e6) return `${trim(value / 1e6)}M`;
  if (abs >= 1e3) return `${trim(value / 1e3)}K`;
  return trim(value);
}

function trim(value: number, digits = 2): string {
  return Number(value.toFixed(digits)).toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function formatKpi(lang: Lang, value: number, format: KpiFormat, unit?: string): string {
  switch (format) {
    case "percent":
      return `${value > 0 ? "+" : ""}${trim(value)}%`;
    case "percentLevel":
      // a level already in percent (ROE 33 -> "33%", ROE 0.8 -> "0.8%"), no sign
      return `${trim(value, 1)}%`;
    case "fraction":
      // ROE / volatility arrive as fractions (0.33 → 33%); values above 1 are already percentages.
      return `${trim(Math.abs(value) <= 1 ? value * 100 : value, 1)}%`;
    case "ratio":
      return trim(value, 2);
    case "money":
      return formatMoney(lang, value);
    case "volume":
      // raw Tushare `amount` (thousands of CNY, no unit field); agent payloads are already in CNY ("money")
      return formatMoney(lang, value * 1000);
    case "price":
      return trim(value, 2);
    default:
      return `${trim(value)}${unit ? ` ${unit}` : ""}`;
  }
}

export function formatCost(value: number | null | undefined, currency: string | null | undefined): string | null {
  if (value == null) return null;
  const symbol = currency === "CNY" ? "¥" : currency === "USD" ? "$" : "";
  const digits = value < 0.01 ? 5 : 4;
  return `${symbol}${value.toFixed(digits)}${symbol ? "" : ` ${currency ?? ""}`}`.trim();
}

/** Whole days between an ISO-ish date and now; undefined when the date cannot be parsed. */
export function ageInDays(asOf: string | null | undefined, now = new Date()): number | undefined {
  if (!asOf) return undefined;
  const parsed = new Date(asOf.length === 10 ? `${asOf}T00:00:00` : asOf);
  if (Number.isNaN(parsed.getTime())) return undefined;
  return Math.max(0, Math.floor((now.getTime() - parsed.getTime()) / 86_400_000));
}

export function formatDate(lang: Lang, asOf: string | null | undefined): string {
  if (!asOf) return "";
  const parsed = new Date(asOf.length === 10 ? `${asOf}T00:00:00` : asOf);
  if (Number.isNaN(parsed.getTime())) return asOf;
  const hasTime = asOf.length > 10 && !/T00:00:00/.test(asOf);
  return new Intl.DateTimeFormat(locale(lang), {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    ...(hasTime ? { hour: "2-digit", minute: "2-digit", hour12: false } : {}),
  }).format(parsed);
}

// Industry names of the industry snapshots: the same table as query_intelligence/agent/names.py INDUSTRY_EN
// (tests/test_web_ui.py checks they are equal). The server also sends them as `name_en` on agent evidence.
export const INDUSTRY_EN: Record<string, string> = {
  白酒: "baijiu (liquor)",
  保险: "insurance",
  券商: "brokerage",
  证券: "securities",
  银行: "banking",
  宽基指数: "broad-based index",
  成长指数: "growth index",
  新能源: "new energy",
  医药: "pharmaceuticals",
  半导体: "semiconductors",
};

/** An industry's English name, capitalised for a label ("白酒" → "Baijiu (liquor)"); undefined if unknown. */
export function industryEnglish(name: string): string | undefined {
  const english = INDUSTRY_EN[name];
  return english ? english.charAt(0).toUpperCase() + english.slice(1) : undefined;
}

const INDUSTRIES_EN: [string, string][] = Object.keys(INDUSTRY_EN).map((name) => [name, industryEnglish(name)!]);

// Structured evidence titles come from the tools in English ("贵州茅台 (600519.SH) daily market data");
// the Chinese UI shows the kind in Chinese. Document titles are left as written.
const TITLE_KINDS: [RegExp, string][] = [
  [/ intraday quote and daily market data$/, " 盘中报价与日行情"],
  [/ daily market data$/, " 日行情"],
  [/ fundamentals$/, " 基本面"],
  [/ technical indicators$/, " 技术指标"],
  [/ industry snapshot$/, " 行业快照"],
  [/ macro indicator$/, " 宏观指标"],
  [/^Document sentiment for (.+)$/, "$1 文档情绪"],
];

export function evidenceTitle(
  lang: Lang,
  title: string | null | undefined,
  englishNames?: Map<string, string>,
): string {
  if (!title) return "";
  if (lang !== "zh") {
    // "贵州茅台 (600519.SH) daily market data" → "Kweichow Moutai (600519.SH) daily market data"
    let out = title;
    for (const [name, english] of [...(englishNames ?? []), ...INDUSTRIES_EN])
      if (name && out.includes(name)) out = out.split(name).join(english);
    return out;
  }
  for (const [pattern, zh] of TITLE_KINDS) if (pattern.test(title)) return title.replace(pattern, zh);
  return title;
}
