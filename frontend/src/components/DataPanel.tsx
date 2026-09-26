import { ArrowDownRight, ArrowUpRight, CalendarClock, DatabaseBackup, LineChart } from "lucide-react";
import { lazy, Suspense } from "react";

import { cn } from "@/lib/cn";
import { formatDate, formatKpi } from "@/lib/format";
import type { EvidenceFreshness } from "@/lib/freshness";
import { useI18n, type MessageKey } from "@/lib/i18n";
import type { Kpi, MarketData } from "@/lib/marketData";

const PriceChart = lazy(() => import("./PriceChart"));

const MAX_TILES = 8;

function KpiTile({ kpi, freshness, onEvidence }: { kpi: Kpi; freshness?: EvidenceFreshness; onEvidence?: (id: string) => void }) {
  const { lang, t } = useI18n();
  const snapshot = freshness?.mode === "snapshot";
  const stale = freshness?.stale;
  const Icon = kpi.tone === "up" ? ArrowUpRight : kpi.tone === "down" ? ArrowDownRight : null;
  const body = (
    <>
      <span className="block truncate text-[11.5px] text-muted">
        {kpi.subject ? `${kpi.subject} · ` : ""}
        {t(kpi.label as MessageKey)}
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
}: {
  data: MarketData;
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
    <section aria-label={t("answer.data")} className="space-y-2.5">
      {series && first && last && (
        <figure className="price-chart rounded-xl border border-line bg-surface p-3">
          <figcaption className="mb-1 flex flex-wrap items-baseline gap-x-2 text-[13px]">
            <LineChart className="size-3.5 self-center text-cobalt" aria-hidden />
            <span className="font-medium">
              {series.name ?? series.symbol} {t("chart.title")}
            </span>
            <span className="text-muted">{t("chart.points", { n: series.points.length })}</span>
            {trendKey && <span className="ml-auto text-muted">{t(trendKey)}</span>}
          </figcaption>
          <Suspense fallback={<div className="h-44 animate-pulse rounded-lg bg-surface-2 sm:h-52" />}>
            <PriceChart
              points={series.points}
              themeKey={themeKey}
              label={t("chart.label", {
                name: series.name ?? series.symbol ?? "",
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
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {data.kpis.slice(0, MAX_TILES).map((kpi) => (
            <KpiTile
              key={kpi.key}
              kpi={kpi}
              freshness={kpi.evidenceId ? freshness?.get(kpi.evidenceId) : undefined}
              onEvidence={onEvidence}
            />
          ))}
        </div>
      )}
    </section>
  );
}
