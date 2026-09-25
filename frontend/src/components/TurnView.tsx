import { AnimatePresence, m as motion } from "motion/react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Copy,
  HelpCircle,
  Info,
  Layers,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Square,
} from "lucide-react";
import { useId, useMemo, useState } from "react";

import type { Turn } from "@/hooks/useChat";
import { evidenceIndex, stripCitations } from "@/lib/citations";
import { cn } from "@/lib/cn";
import { formatMs } from "@/lib/format";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { liveTrace, traceStats } from "@/lib/trace";
import type { AnswerView } from "@/lib/view";

import { DataPanel } from "./DataPanel";
import { RichText, type CiteHandler } from "./RichText";
import { SentimentBar } from "./SentimentBar";
import { TraceTimeline } from "./TraceTimeline";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Tooltip } from "./ui/tooltip";

export interface TurnActions {
  onCite: (turnId: string, evidenceId: string) => void;
  onInspect: (turnId: string, tab: "evidence" | "trace" | "run") => void;
  onAsk: (query: string) => void;
  onRetry: (turn: Turn) => void;
}

interface Props extends TurnActions {
  turn: Turn;
  view: AnswerView | null;
  isLast: boolean;
  activeEvidence?: string | null;
  themeKey: string;
}

function UserBubble({ turn }: { turn: Turn }) {
  const { t } = useI18n();
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] space-y-1 text-right">
        {turn.via === "resume" && (
          <span className="inline-flex items-center gap-1 text-[11.5px] text-muted">
            <HelpCircle className="size-3" aria-hidden />
            {t("composer.replying")}
          </span>
        )}
        <p
          className="user-message rounded-2xl rounded-br-md bg-cobalt px-4 py-2.5 text-left text-[15px] leading-relaxed whitespace-pre-wrap text-cobalt-ink"
          aria-label={t("a11y.userSaid")}
        >
          {turn.query}
        </p>
      </div>
    </div>
  );
}

function RouteBadge({ view }: { view: AnswerView }) {
  const { t } = useI18n();
  const route = view.route;
  if (!route) return null;
  const label: Record<string, string> = {
    workflow: t("mode.workflow"),
    agent: t("mode.agent"),
    classic: t("mode.classic"),
    clarify: t("answer.clarify"),
    refuse: t("answer.refused"),
  };
  return (
    <Badge tone={route === "agent" ? "gilt" : route === "refuse" ? "warn" : "cobalt"} className="route-badge">
      {route === "agent" ? <Sparkles /> : <Layers />}
      {label[route] ?? route}
    </Badge>
  );
}

function VerificationBadge({ view }: { view: AnswerView }) {
  const { t } = useI18n();
  const passed = view.verification?.passed;
  if (passed === undefined || passed === null) return null;
  const repaired = (view.agent?.degraded ?? []).some((item) => item.startsWith("verification_failed"));
  return (
    <Tooltip content={t("trace.checkedNumbers", { n: view.verification?.checked_numbers ?? 0 })}>
      <Badge tone={passed ? "down" : "warn"} tabIndex={0} className="verification-badge">
        {passed ? <ShieldCheck /> : <ShieldAlert />}
        {passed ? t("trace.verifyPassed") : repaired ? t("trace.verifyRepaired") : t("trace.verifyFailed")}
      </Badge>
    </Tooltip>
  );
}

function TraceDisclosure({
  view,
  turn,
  onEvidence,
}: {
  view?: AnswerView | null;
  turn: Turn;
  onEvidence: (id: string) => void;
}) {
  const { t } = useI18n();
  const live = turn.status === "running";
  const nodes = view ? view.trace : liveTrace(turn.steps);
  const [openState, setOpen] = useState<boolean | null>(null);
  const open = openState ?? live;
  const panelId = useId();
  if (!nodes.length && !live) return null;
  const stats = traceStats(nodes);
  const ms = view?.serverMs ?? view?.wallMs;
  return (
    <div className="trace rounded-xl border border-line bg-bg/60">
      <button
        type="button"
        className="trace-toggle flex w-full items-center gap-2 px-3 py-2 text-left text-[13px]"
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={open ? t("trace.collapse") : t("trace.expand")}
        onClick={() => setOpen(!open)}
      >
        <span className={cn("size-1.5 rounded-full", live ? "animate-pulse bg-gilt" : "bg-cobalt")} aria-hidden />
        <span className="font-medium">{live ? <span className="shimmer">{t("trace.live")}…</span> : t("trace.title")}</span>
        <span className="text-faint tabular-nums">
          {t("trace.summary", { steps: stats.steps, tools: stats.tools, ms: ms !== undefined ? formatMs(ms) : "…" })}
        </span>
        <ChevronDown className={cn("ml-auto size-4 text-faint transition-transform", open && "rotate-180")} aria-hidden />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={panelId}
            key="trace"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.2, 0.7, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="border-t border-line px-3 pt-2 pb-2.5">
              <TraceTimeline nodes={nodes} live={live} response={view?.agent} onEvidence={onEvidence} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  return (
    <Tooltip content={copied ? t("answer.copied") : t("answer.copy")}>
      <Button
        size="icon-sm"
        aria-label={t("answer.copy")}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1400);
          } catch {
            /* clipboard unavailable */
          }
        }}
      >
        {copied ? <Check /> : <Copy />}
      </Button>
    </Tooltip>
  );
}

function AnswerCard({ turn, view, isLast, activeEvidence, themeKey, onCite, onInspect, onAsk }: Props & { view: AnswerView }) {
  const { t } = useI18n();
  const cite: CiteHandler = useMemo(
    () => ({
      index: evidenceIndex(view.evidence.map((source) => source.evidence_id)),
      titles: new Map(view.evidence.map((source) => [source.evidence_id, source.title ?? ""])),
      onCite: (id: string) => onCite(turn.id, id),
      active: activeEvidence,
    }),
    [view.evidence, onCite, turn.id, activeEvidence],
  );
  const copyText = [stripCitations(view.answer), view.disclaimer].filter(Boolean).join("\n\n");
  const warn = view.kind === "classic" && view.llmStatus && view.llmStatus !== "ok";
  return (
    <article className="answer-card space-y-3.5 rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5" aria-label={t("a11y.assistant")}>
      <header className="flex flex-wrap items-center gap-1.5">
        <RouteBadge view={view} />
        <VerificationBadge view={view} />
        {view.evidence.length > 0 && (
          <button
            type="button"
            onClick={() => onInspect(turn.id, "evidence")}
            className="evidence-count inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11.5px] font-medium text-muted hover:bg-surface-2 hover:text-ink"
          >
            <Layers className="size-3" aria-hidden />
            {t("answer.sources", { n: view.evidence.length })}
          </button>
        )}
        <span className="ml-auto flex items-center gap-1">
          {view.wallMs !== undefined && <span className="text-[11.5px] text-faint tabular-nums">{formatMs(view.wallMs)}</span>}
          <CopyButton text={copyText} />
        </span>
      </header>

      {warn && (
        <p className="flex items-center gap-1.5 rounded-lg bg-warn-soft px-3 py-2 text-[13px] text-warn">
          <Info className="size-4 shrink-0" aria-hidden />
          {t("answer.classicFallback")}
        </p>
      )}

      {view.kind === "agent" && <TraceDisclosure view={view} turn={turn} onEvidence={(id) => onCite(turn.id, id)} />}

      <RichText text={view.answer} cite={cite} className="answer-text text-[15px] leading-[1.8] text-ink" />

      {view.data && <DataPanel data={view.data} themeKey={themeKey} onEvidence={(id) => onCite(turn.id, id)} />}

      {view.keyPoints.length > 0 && (
        <section className="key-points">
          <h3 className="mb-1.5 text-[13px] font-medium text-muted">{t("answer.keyPoints")}</h3>
          <RichText text={view.keyPoints.map((point) => `- ${point}`).join("\n")} cite={cite} className="text-[14px] leading-relaxed" />
        </section>
      )}

      {view.limitations.length > 0 && (
        <section className="limitations rounded-lg bg-surface-2/70 px-3 py-2">
          <h3 className="mb-1 flex items-center gap-1.5 text-[12.5px] font-medium text-muted">
            <AlertTriangle className="size-3.5 text-warn" aria-hidden />
            {t(view.kind === "classic" ? "run.warnings" : "answer.limitations")}
          </h3>
          <RichText
            text={view.limitations.map((item) => `- ${item}`).join("\n")}
            cite={cite}
            className="text-[12.5px] leading-relaxed text-muted [&_ul]:space-y-0.5"
          />
        </section>
      )}

      {view.sentiment && <SentimentBar sentiment={view.sentiment} />}

      {isLast && view.next.length > 0 && (
        <section aria-label={t("answer.next")}>
          <h3 className="mb-1.5 text-[13px] font-medium text-muted">{t("answer.next")}</h3>
          <div className="flex flex-wrap gap-1.5">
            {view.next.map((item) => (
              <button
                key={item.question}
                type="button"
                onClick={() => onAsk(item.question)}
                className="next-question rounded-full border border-line bg-surface px-3 py-1.5 text-left text-[13px] text-ink transition-colors hover:border-cobalt hover:bg-cobalt-soft hover:text-cobalt"
              >
                {item.question}
              </button>
            ))}
          </div>
        </section>
      )}

      <footer className="disclaimer flex items-start gap-1.5 border-t border-line pt-3 text-[12px] leading-relaxed text-faint">
        <ShieldCheck className="mt-0.5 size-3.5 shrink-0" aria-hidden />
        <span>{view.disclaimer || t("answer.disclaimerDefault")}</span>
      </footer>
    </article>
  );
}

function ClarificationCard({ turn }: { turn: Turn }) {
  const { t } = useI18n();
  return (
    <article className="clarification-card space-y-3 rounded-2xl border border-gilt/50 bg-gilt-soft/50 p-4 sm:p-5" aria-label={t("answer.clarify")}>
      <Badge tone="gilt">
        <HelpCircle />
        {t("answer.clarify")}
      </Badge>
      <p className="text-[15px] leading-relaxed">{turn.clarification?.question}</p>
      {turn.clarification?.original_query && (
        <p className="text-[12.5px] text-muted">
          “{turn.clarification.original_query}” · {t("answer.clarifyHint")}
        </p>
      )}
    </article>
  );
}

function ErrorCard({ turn, onRetry }: { turn: Turn; onRetry: (turn: Turn) => void }) {
  const { t } = useI18n();
  const kind = turn.error?.kind ?? "generic";
  const message =
    kind === "generic" ? t("error.generic", { e: turn.error?.message ?? "" }) : t(`error.${kind}` as MessageKey);
  return (
    <div role="alert" className="error-box flex items-start gap-2 rounded-xl border border-up/30 bg-up-soft px-4 py-3 text-[14px] text-up">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
      <span className="flex-1">{message}</span>
      <Button size="sm" variant="outline" onClick={() => onRetry(turn)}>
        <RotateCcw />
        {t("error.retry")}
      </Button>
    </div>
  );
}

export function TurnView(props: Props) {
  const { turn, view } = props;
  const { t } = useI18n();
  return (
    <motion.div
      className="turn space-y-3"
      data-turn-id={turn.id}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
    >
      <UserBubble turn={turn} />
      {turn.status === "running" && (
        <div className="answer-card rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5" aria-busy="true">
          {turn.via === "stream" ? (
            <TraceDisclosure turn={turn} onEvidence={() => undefined} />
          ) : (
            <p className="text-[14px]">
              <span className="shimmer font-medium">{t("app.running")}…</span>
            </p>
          )}
        </div>
      )}
      {turn.status === "clarify" && <ClarificationCard turn={turn} />}
      {turn.status === "error" && <ErrorCard turn={turn} onRetry={props.onRetry} />}
      {turn.status === "stopped" && (
        <p className="flex items-center gap-1.5 text-[13px] text-faint">
          <Square className="size-3" aria-hidden />
          {t("answer.stopped")}
        </p>
      )}
      {turn.status === "done" && view && <AnswerCard {...props} view={view} />}
    </motion.div>
  );
}
