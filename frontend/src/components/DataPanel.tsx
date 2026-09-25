import { ArrowDownRight, ArrowUpRight, LineChart } from "lucide-react";
import { lazy, Suspense } from "react";

import { cn } from "@/lib/cn";
import { formatDate, formatKpi } from "@/lib/format";
import { useI18n, type MessageKey } from "@/lib/i18n";
import type { Kpi, MarketData } from "@/lib/marketData";

const PriceChart = lazy(() => import("./PriceChart"));

const MAX_TILES = 8;

function KpiTile({ kpi, onEvidence }: { kpi: Kpi; onEvidence?: (id: string) => void }) {
  const { lang, t } = useI18n();
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
      {kpi.asOf && <span className="block text-[11px] text-faint">{formatDate(lang, kpi.asOf)}</span>}
    </>
  );
  const className = "kpi-tile block min-w-0 rounded-lg border border-line bg-surface px-3 py-2 text-left";
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
  onEvidence,
}: {
  data: MarketData;
  themeKey: string;
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
            <span className="text-faint">{t("chart.points", { n: series.points.length })}</span>
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
            <KpiTile key={kpi.key} kpi={kpi} onEvidence={onEvidence} />
          ))}
        </div>
      )}
    </section>
  );
}
