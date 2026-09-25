import { cn } from "@/lib/cn";
import { useI18n, type MessageKey } from "@/lib/i18n";
import type { SentimentSummary } from "@/lib/types";

const ORDER = ["positive", "neutral", "negative"] as const;
// A-share reading: good news in the "rise" colour, bad news in the "fall" colour.
const COLORS: Record<string, string> = { positive: "bg-up", neutral: "bg-faint/60", negative: "bg-down" };

export function SentimentBar({ sentiment }: { sentiment: SentimentSummary }) {
  const { t } = useI18n();
  const counts = sentiment.label_counts ?? {};
  const total = ORDER.reduce((sum, label) => sum + (counts[label] ?? 0), 0);
  if (!total) return null;
  const overall = sentiment.overall_label && ORDER.includes(sentiment.overall_label as (typeof ORDER)[number])
    ? t(`sentiment.${sentiment.overall_label}` as MessageKey)
    : sentiment.overall_label;
  return (
    <section className="sentiment-summary space-y-1.5" aria-label={t("answer.sentiment")}>
      <div className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
        <span className="font-medium">{t("answer.sentiment")}</span>
        {overall && <span className="text-ink">{overall}</span>}
        <span className="text-faint">
          {t("sentiment.docs", { n: total })}
          {sentiment.mean_score !== undefined && ` · ${t("sentiment.mean", { v: sentiment.mean_score.toFixed(2) })}`}
          {sentiment.backend && ` · ${t("sentiment.model", { m: sentiment.backend })}`}
        </span>
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={ORDER.map((label) => `${t(`sentiment.${label}` as MessageKey)} ${counts[label] ?? 0}`).join(", ")}>
        {ORDER.map((label) =>
          counts[label] ? (
            <span key={label} className={cn("h-full", COLORS[label])} style={{ width: `${((counts[label] ?? 0) / total) * 100}%` }} />
          ) : null,
        )}
      </div>
      <div className="flex gap-3 text-[11.5px] text-muted">
        {ORDER.map((label) => (
          <span key={label} className="inline-flex items-center gap-1">
            <span className={cn("size-2 rounded-full", COLORS[label])} aria-hidden />
            {t(`sentiment.${label}` as MessageKey)} {counts[label] ?? 0}
          </span>
        ))}
      </div>
    </section>
  );
}
