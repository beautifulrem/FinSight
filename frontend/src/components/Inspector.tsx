import { Tabs } from "radix-ui";
import { Activity, Gauge, Layers, ScanSearch } from "lucide-react";

import type { Turn } from "@/hooks/useChat";
import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";
import type { AnswerView } from "@/lib/view";

import { EvidenceList } from "./EvidenceList";
import { RunDetails } from "./RunDetails";
import { TraceTimeline } from "./TraceTimeline";
import { Waterfall } from "./Waterfall";

export type InspectorTab = "evidence" | "trace" | "run";

interface Props {
  turn?: Turn;
  view?: AnswerView | null;
  tab: InspectorTab;
  onTab: (tab: InspectorTab) => void;
  highlight?: string | null;
  highlightNonce?: number;
  sessionId: string;
  onEvidence: (id: string) => void;
  className?: string;
}

const TABS: { value: InspectorTab; icon: typeof Layers; label: "inspector.evidence" | "inspector.trace" | "inspector.run" }[] = [
  { value: "evidence", icon: Layers, label: "inspector.evidence" },
  { value: "trace", icon: Activity, label: "inspector.trace" },
  { value: "run", icon: Gauge, label: "inspector.run" },
];

export function Inspector({ turn, view, tab, onTab, highlight, highlightNonce, sessionId, onEvidence, className }: Props) {
  const { t } = useI18n();
  return (
    <Tabs.Root value={tab} onValueChange={(value) => onTab(value as InspectorTab)} className={cn("inspector flex min-h-0 flex-col", className)}>
      <Tabs.List aria-label={t("inspector.title")} className="flex shrink-0 gap-1 rounded-lg bg-surface-2 p-1">
        {TABS.map(({ value, icon: Icon, label }) => (
          <Tabs.Trigger
            key={value}
            value={value}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-[13px] font-medium text-muted transition-colors hover:text-ink data-[state=active]:bg-surface data-[state=active]:text-ink data-[state=active]:shadow-sm"
          >
            <Icon className="size-3.5" aria-hidden />
            {t(label)}
            {value === "evidence" && view ? <span className="text-faint tabular-nums">{view.evidence.length}</span> : null}
          </Tabs.Trigger>
        ))}
      </Tabs.List>
      {turn && (
        <p className="mt-2.5 truncate px-0.5 text-[12px] text-faint" title={turnLabel(turn)}>
          {t("inspector.forTurn", { q: turnLabel(turn) })}
        </p>
      )}
      <div className="scrollbar-thin mt-2 min-h-0 flex-1 overflow-y-auto pr-0.5 pb-4">
        {!view || !turn ? (
          <div className="flex flex-col items-center gap-3 px-6 py-16 text-center text-[13px] text-faint">
            <ScanSearch className="size-8 text-line" aria-hidden />
            {t("inspector.empty")}
          </div>
        ) : (
          <>
            <Tabs.Content value="evidence" className="outline-none">
              <EvidenceList sources={view.evidence} cited={view.cited} highlight={highlight} highlightNonce={highlightNonce} />
            </Tabs.Content>
            <Tabs.Content value="trace" className="space-y-4 outline-none">
              <Waterfall nodes={view.trace} />
              {view.trace.length > 0 ? (
                <TraceTimeline nodes={view.trace} response={view.agent} onEvidence={onEvidence} />
              ) : (
                <p className="py-6 text-center text-[13px] text-faint">{t("mode.classic.hint")}</p>
              )}
            </Tabs.Content>
            <Tabs.Content value="run" className="outline-none">
              <RunDetails view={view} turn={turn} sessionId={sessionId} />
            </Tabs.Content>
          </>
        )}
      </div>
    </Tabs.Root>
  );
}

/** A clarification reply is shown with the question it answered, e.g. "它的市盈率呢 → 宁德时代". */
function turnLabel(turn: { query: string; via?: string; agent?: { query?: string | null } }): string {
  const original = turn.agent?.query;
  return turn.via === "resume" && original && original !== turn.query ? `${original} → ${turn.query}` : turn.query;
}
