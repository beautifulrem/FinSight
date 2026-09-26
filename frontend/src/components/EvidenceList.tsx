import { ExternalLink } from "lucide-react";
import { useEffect, useRef } from "react";

import { cn } from "@/lib/cn";
import { formatDate } from "@/lib/format";
import { evidenceFreshness, type EvidenceFreshness } from "@/lib/freshness";
import { sourceTypeLabel, toolLabel, useI18n } from "@/lib/i18n";
import type { EvidenceSource } from "@/lib/types";

import { FreshnessBadges } from "./Freshness";
import { Badge } from "./ui/badge";

/** "2026/09/24 · 2 days ago": the as-of date and its age, in the warning colour once possibly stale. */
function AsOf({ info }: { info: EvidenceFreshness }) {
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

interface Props {
  sources: EvidenceSource[];
  cited: Set<string>;
  highlight?: string | null;
  /** Changes on every citation click so the same id can be re-flashed. */
  highlightNonce?: number;
}

/** The evidence ledger: numbered sources in citation order (E1, E2, … match the answer chips). */
export function EvidenceList({ sources, cited, highlight, highlightNonce }: Props) {
  const { lang, t } = useI18n();
  const refs = useRef(new Map<string, HTMLLIElement>());

  useEffect(() => {
    if (!highlight) return;
    const element = refs.current.get(highlight);
    if (!element) return;
    element.scrollIntoView({ behavior: "smooth", block: "nearest" });
    element.classList.remove("evidence-flash");
    void element.offsetWidth; // restart the animation
    element.classList.add("evidence-flash");
    element.focus({ preventScroll: true });
  }, [highlight, highlightNonce]);

  if (!sources.length) return <p className="px-1 py-6 text-center text-[13px] text-faint">{t("evidence.none")}</p>;

  return (
    <ol className="evidence-list space-y-2">
      {sources.map((source, i) => {
        const isCited = cited.has(source.evidence_id);
        const url = source.source_url && /^https?:\/\//i.test(source.source_url) ? source.source_url : null;
        const info = evidenceFreshness(source);
        return (
          <li
            key={source.evidence_id}
            ref={(element) => {
              if (element) refs.current.set(source.evidence_id, element);
              else refs.current.delete(source.evidence_id);
            }}
            tabIndex={-1}
            data-evidence-id={source.evidence_id}
            aria-current={highlight === source.evidence_id ? "true" : undefined}
            className={cn(
              "evidence-item grid grid-cols-[2.6rem_1fr] gap-x-2 rounded-xl border bg-surface p-3 transition-colors outline-none",
              highlight === source.evidence_id ? "border-gilt" : "border-line",
            )}
          >
            <span
              className={cn(
                "font-mono text-[22px] leading-7 font-medium tabular-nums",
                isCited ? "text-gilt" : "text-faint",
              )}
              aria-label={`E${i + 1}`}
            >
              E{i + 1}
            </span>
            <div className="min-w-0 space-y-1">
              <div className="flex flex-wrap items-center gap-1">
                <Badge tone="cobalt">{sourceTypeLabel(lang, source.source_type)}</Badge>
                <Badge tone={isCited ? "gilt" : "neutral"}>{isCited ? t("evidence.cited") : t("evidence.notCited")}</Badge>
                <FreshnessBadges source={source} info={info} />
              </div>
              <p className="text-[13.5px] leading-snug font-medium text-pretty text-ink">
                {source.title || source.evidence_id}
              </p>
              <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[12px]">
                {(info.provenance?.source_label || source.source_name) && (
                  <span className="text-muted">{info.provenance?.source_label || source.source_name}</span>
                )}
                <AsOf info={info} />
                {source.produced_by && (
                  <span className="text-muted" title={toolLabel(lang, source.produced_by)}>
                    {t("evidence.by", { tool: "" })}
                    <code className="font-mono text-[11px]">{source.produced_by}</code>
                  </span>
                )}
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 pt-0.5">
                <code className="min-w-0 truncate font-mono text-[11px] text-muted">{source.evidence_id}</code>
                {url ? (
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-[12px] font-medium text-cobalt hover:underline"
                  >
                    {t("evidence.open")}
                    <ExternalLink className="size-3" aria-hidden />
                  </a>
                ) : source.kind === "structured" ? (
                  <span className="text-[11.5px] text-muted">{t("evidence.noLink")}</span>
                ) : null}
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
