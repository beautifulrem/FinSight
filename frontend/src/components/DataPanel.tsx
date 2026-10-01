import { ArrowDownRight, ArrowUpRight, CalendarClock, DatabaseBackup, LineChart } from "lucide-react";
import { lazy, Suspense } from "react";

import { cn } from "@/lib/cn";
import { formatDate, formatKpi } from "@/lib/format";
import type { EvidenceFreshness } from "@/lib/freshness";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { selectKpis, type Kpi, type MarketData } from "@/lib/marketData";

const PriceChart = lazy(() => import("./PriceChart"));

const MAX_TILES = 8;

function KpiTile({
  kpi,
  subject,
  freshness,
  onEvidence,
}: {
  kpi: Kpi;
  subject?: string;
  freshness?: EvidenceFreshness;
  onEvidence?: (id: string) => void;
}) {
  const { lang, t } = useI18n();
  const snapshot = freshness?.mode === "snapshot";
  const stale = freshness?.stale;
  const Icon = kpi.tone === "up" ? ArrowUpRight : kpi.tone === "down" ? ArrowDownRight : null;
  const body = (
    <>
      <span className={cn("block truncate text-[11.5px]", kpi.featured ? "font-medium text-cobalt" : "text-muted")}>
        {subject ? `${subject} · ` : ""}
        {t(kpi.label as MessageKey)}
        {kpi.featured && <span className="sr-only"> ({t("kpi.asked")})</span>}
      </span>
      <span
        className={cn(
          "mt-0.5 flex items-center gap-0.5 text-lg leading-6 font-semibold tabular-nums",
          kpi.tone === "up" && "text-up",
          kpi.tone === "down" && "text-down",
        )}
      >
        {Icon && <Icon className="size-4" aria-hidden />}
        {formatKpi(lang, kpi.value, kpi.format, kpi.unit)}
      </span>
      {kpi.asOf && (
        <span className={cn("flex flex-wrap items-center gap-x-1 text-[11px]", stale || snapshot ? "text-warn" : "text-muted")}>
          {formatDate(lang, kpi.asOf)}
          {(snapshot || stale) && (
            <span className="kpi-freshness inline-flex items-center gap-0.5 font-medium">
              {snapshot ? <DatabaseBackup className="size-3" aria-hidden /> : <CalendarClock className="size-3" aria-hidden />}
              {t(snapshot ? "fresh.snapshot" : "fresh.stale")}
            </span>
          )}
        </span>
      )}
    </>
  );
  const className = cn(
    "kpi-tile block min-w-0 rounded-lg border bg-surface px-3 py-2 text-left",
    stale || snapshot ? "border-warn/45 border-dashed" : "border-line",
    kpi.featured && "kpi-featured ring-1 ring-cobalt/45",
  );
  return kpi.evidenceId && onEvidence ? (
    <button type="button" className={cn(className, "hover:border-cobalt")} onClick={() => onEvidence(kpi.evidenceId!)}>
      {body}
    </button>
  ) : (
    <div className={className}>{body}</div>
  );
}

export function DataPanel({
  data,
  themeKey,
  freshness,
  onEvidence,
  turn,
  displayName = (name) => name,
  asked = [],
}: {
  data: MarketData;
  /** The turn number, so each turn's data region has its own name (axe landmark-unique). */
  turn?: number;
  /** The name to show for a subject (the English name in the English UI). */
  displayName?: (name: string) => string;
  /** Metric keys the question asks about: their tiles come first and are marked. */
  asked?: string[];
  themeKey: string;
  /** Per-evidence freshness, so a snapshot or stale tile is marked next to live ones. */
  freshness?: Map<string, EvidenceFreshness>;
  onEvidence?: (id: string) => void;
}) {
  const { lang, t } = useI18n();
  const series = data.series[0];
  const first = series?.points[0];
  const last = series?.points[series.points.length - 1];
  const trendKey = data.trend ? (`trend.${data.trend}` as MessageKey) : null;
  return (
    <section
      aria-label={turn === undefined ? t("answer.data") : t("a11y.inTurn", { label: t("answer.data"), n: turn })}
      className="data-panel space-y-2.5"
    >
      {series && first && last && (
        <figure className="price-chart rounded-xl border border-line bg-surface p-3">
          <figcaption className="mb-1 flex flex-wrap items-baseline gap-x-2 text-[13px]">
            <LineChart className="size-3.5 self-center text-cobalt" aria-hidden />
            <span className="font-medium">
              {displayName(series.name ?? series.symbol ?? "")} {t("chart.title")}
            </span>
            <span className="text-muted">{t("chart.points", { n: series.points.length })}</span>
            {trendKey && <span className="ml-auto text-muted">{t(trendKey)}</span>}
          </figcaption>
          <Suspense fallback={<div className="h-44 motion-safe:animate-pulse rounded-lg bg-surface-2 sm:h-52" />}>
            <PriceChart
              points={series.points}
              themeKey={themeKey}
              label={t("chart.label", {
                name: displayName(series.name ?? series.symbol ?? ""),
                from: formatDate(lang, first.time),
                to: formatDate(lang, last.time),
                a: first.value,
                b: last.value,
              })}
            />
          </Suspense>
        </figure>
      )}
      {data.kpis.length > 0 && (
        <div className="kpi-grid grid grid-cols-2 gap-2 sm:grid-cols-4">
          {selectKpis(data.kpis, MAX_TILES, asked).map((kpi) => (
            <KpiTile
              key={kpi.key}
              kpi={kpi}
              subject={kpi.subject ? displayName(kpi.subject) : undefined}
              freshness={kpi.evidenceId ? freshness?.get(kpi.evidenceId) : undefined}
              onEvidence={onEvidence}
            />
          ))}
        </div>
      )}
    </section>
  );
}
