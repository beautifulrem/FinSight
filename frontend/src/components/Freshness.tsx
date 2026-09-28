import { CalendarClock, Clock3, DatabaseBackup, Radio, Shuffle } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";
import { fallbackReasonText } from "@/lib/codes";
import { formatDate } from "@/lib/format";
import { evidenceFreshness, type EvidenceFreshness, type FreshnessSummary } from "@/lib/freshness";
import { sourceNameLabel, useI18n, type MessageKey } from "@/lib/i18n";
import type { EvidenceSource } from "@/lib/types";

import { Badge } from "./ui/badge";
import { Tooltip } from "./ui/tooltip";

const MODE_KEYS = {
  live: { label: "fresh.live", hint: "fresh.liveHint", tone: "neutral", icon: Radio },
  fallback: { label: "fresh.fallback", hint: "fresh.fallbackHint", tone: "cobalt", icon: Shuffle },
  cached: { label: "fresh.cached", hint: "fresh.cachedHint", tone: "warn", icon: Clock3 },
  snapshot: { label: "fresh.snapshot", hint: "fresh.snapshotHint", tone: "warn", icon: DatabaseBackup },
} as const;

function ProvenanceDetails({ info, mode }: { info: EvidenceFreshness; mode: keyof typeof MODE_KEYS }) {
  const { lang, t } = useI18n();
  const provenance = info.provenance;
  const lines: ReactNode[] = [t(MODE_KEYS[mode].hint)];
  const label = sourceNameLabel(lang, provenance);
  if (label) lines.push(t("fresh.source", { s: label }));
  if (provenance?.fetched_at) lines.push(t("fresh.fetched", { t: formatDate(lang, provenance.fetched_at) }));
  if (provenance?.fallback_reason) lines.push(t("fresh.reason", { r: fallbackReasonText(lang, provenance.fallback_reason) }));
  return (
    <span className="block space-y-0.5">
      {lines.map((line, i) => (
        <span key={i} className="block">
          {line}
        </span>
      ))}
    </span>
  );
}

/** "2026/09/24 · 2 days ago": the as-of date and its age, in the warning colour once possibly stale. */
export function AsOf({ info }: { info: EvidenceFreshness }) {
  const { lang, t } = useI18n();
  const { asOf, days, stale } = info;
  if (days === undefined || !asOf) return <span className="text-muted">{t("evidence.unknownTime")}</span>;
  return (
    <span className="inline-flex flex-wrap items-center gap-x-1.5">
      <time dateTime={asOf} className="text-muted">
        {formatDate(lang, asOf)}
      </time>
      <span className={cn(stale ? "text-warn" : "text-muted")}>
        {days === 0 ? t("evidence.today") : t(stale ? "evidence.stale" : "evidence.fresh", { d: days })}
      </span>
    </span>
  );
}

/** Live / fallback / cached / snapshot and "may be stale" badges for one evidence item. */
export function FreshnessBadges({ source, info = evidenceFreshness(source) }: { source: EvidenceSource; info?: EvidenceFreshness }) {
  const { t } = useI18n();
  const mode = info.mode === "unknown" ? null : info.mode;
  const spec = mode ? MODE_KEYS[mode] : null;
  return (
    <>
      {spec && mode && (
        <Tooltip content={<ProvenanceDetails info={info} mode={mode} />}>
          <Badge tone={spec.tone} tabIndex={0} className="freshness-badge" data-mode={mode}>
            <spec.icon aria-hidden />
            {t(spec.label)}
          </Badge>
        </Tooltip>
      )}
      {info.stale && (
        <Badge tone="warn" className="stale-badge">
          <CalendarClock aria-hidden />
          {t("fresh.stale")}
        </Badge>
      )}
    </>
  );
}

/** Answer-level "data as of / stale / fallback" banner. Hidden when every value is live and current. */
export function FreshnessBanner({ summary, onReview }: { summary: FreshnessSummary; onReview?: () => void }) {
  const { lang, t } = useI18n();
  if (summary.level === "none") return null;
  const warn = summary.level === "warn";
  const counts: [MessageKey, number][] = [
    ["fresh.nSnapshot", summary.snapshot.length],
    ["fresh.nStale", summary.stale.length],
    ["fresh.nCached", summary.cached.length],
    ["fresh.nFallback", summary.fallback.length],
  ];
  const dates =
    summary.from && summary.to
      ? summary.from === summary.to
        ? t("fresh.asOf", { d: formatDate(lang, summary.from) })
        : t("fresh.range", { from: formatDate(lang, summary.from), to: formatDate(lang, summary.to) })
      : null;
  return (
    <section
      className={
        warn
          ? "freshness-banner flex items-start gap-2 rounded-lg border border-warn/35 bg-warn-soft px-3 py-2 text-[12.5px] text-ink"
          : "freshness-banner flex items-start gap-2 rounded-lg border border-cobalt/25 bg-cobalt-soft/60 px-3 py-2 text-[12.5px] text-ink"
      }
      data-level={summary.level}
      aria-label={t(warn ? "fresh.bannerWarn" : "fresh.bannerInfo")}
    >
      {warn ? (
        <CalendarClock className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
      ) : (
        <Shuffle className="mt-0.5 size-4 shrink-0 text-cobalt" aria-hidden />
      )}
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="font-medium">{t(warn ? "fresh.bannerWarn" : "fresh.bannerInfo")}</p>
        <p className="text-muted">
          {[
            dates,
            summary.mixed ? t("fresh.spread", { n: summary.spreadDays }) : null,
            ...counts.filter(([, n]) => n > 0).map(([key, n]) => t(key, { n })),
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
      {onReview && (
        <button
          type="button"
          onClick={onReview}
          className="shrink-0 rounded-md px-1.5 py-0.5 font-medium text-cobalt hover:bg-surface/60 hover:underline"
        >
          {t("fresh.review")}
        </button>
      )}
    </section>
  );
}
