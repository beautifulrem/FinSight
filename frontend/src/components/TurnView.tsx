import { AnimatePresence, m as motion, useReducedMotion } from "motion/react";
import {
  AlertTriangle,
  ChevronDown,
  FilePenLine,
  HelpCircle,
  Info,
  Layers,
  RotateCcw,
  ShieldAlert,
  SearchCheck,
  ShieldCheck,
  Sparkles,
  Square,
} from "lucide-react";
import { useEffect, useId, useMemo, useState } from "react";

import type { Turn } from "@/hooks/useChat";
import { useElapsed } from "@/hooks/useElapsed";
import { evidenceIndex, stripCitations } from "@/lib/citations";
import { claimInMessage } from "@/lib/claims";
import { cn } from "@/lib/cn";
import { limitationText } from "@/lib/codes";
import { answerEdited, streamingText } from "@/lib/streaming";
import { formatMs } from "@/lib/format";
import { evidenceFreshness, summarizeFreshness } from "@/lib/freshness";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { turnProgress, type Progress } from "@/lib/progress";
import { liveTrace, traceStats } from "@/lib/trace";
import type { ClaimReport } from "@/lib/types";
import { displayName, type AnswerView } from "@/lib/view";

import { CopyButton, ExportMenu, FeedbackControls, type FeedbackSender } from "./AnswerActions";
import { ClaimReportCard } from "./ClaimCheck";
import { CodeText } from "./CodeLabel";
import { DataPanel } from "./DataPanel";
import { FreshnessBanner } from "./Freshness";
import { RichText, type CiteHandler } from "./RichText";
import { RunProgress } from "./RunProgress";
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
  onFeedback: FeedbackSender;
  /** Open the fact-check view with this claim (the "听说…是真的吗" hint). */
  onCheckClaim?: (claim: string) => void;
  /** Stop the running turn (the composer's Stop, repeated next to the progress). */
  onStop?: () => void;
}

interface Props extends TurnActions {
  turn: Turn;
  view: AnswerView | null;
  isLast: boolean;
  activeEvidence?: string | null;
  themeKey: string;
  /** 1-based turn number: names this turn's regions uniquely ("数据（第 2 轮）"). */
  number?: number;
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

/** "听说茅台市盈率只有15倍，是真的吗": point to the fact-check view, which compares every number. */
function ClaimHint({ claim, onCheck }: { claim: string; onCheck: (claim: string) => void }) {
  const { t } = useI18n();
  return (
    <div className="claim-hint flex justify-end">
      <div className="flex max-w-[85%] flex-wrap items-center justify-end gap-x-2 gap-y-1 text-[12.5px] text-muted">
        <span>
          <span className="font-medium text-ink">{t("claim.hint.title")}</span> · {t("claim.hint.body")}
        </span>
        <Button size="sm" variant="outline" className="claim-hint-action" onClick={() => onCheck(claim)}>
          <SearchCheck aria-hidden />
          {t("claim.hint.action")}
        </Button>
      </div>
    </div>
  );
}

/** The server's check of a hearsay question ("听说…是真的吗"), shown inside the answer. */
function InlineFactCheck({ report, number, onOpen }: { report: ClaimReport; number?: number; onOpen?: (claim: string) => void }) {
  const { t } = useI18n();
  const label = number === undefined ? t("claim.inline.title") : t("a11y.inTurn", { label: t("claim.inline.title"), n: number });
  return (
    <section className="inline-fact-check space-y-2" aria-label={label}>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="flex items-center gap-1.5 text-[13px] font-medium text-muted">
          <SearchCheck className="size-3.5 text-cobalt" aria-hidden />
          {t("claim.inline.title")}
        </h2>
        {onOpen && (
          <Button size="sm" variant="ghost" className="inline-fact-check-open ml-auto" onClick={() => onOpen(report.claim)}>
            {t("claim.inline.open")}
          </Button>
        )}
      </div>
      <ClaimReportCard report={report} turn={number} />
    </section>
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
  live: liveState,
  onEvidence,
}: {
  view?: AnswerView | null;
  turn: Turn;
  /** While the turn runs: what it is doing now and for how long (seconds). */
  live?: { progress: Progress; elapsed: number };
  onEvidence: (id: string) => void;
}) {
  const { t } = useI18n();
  const live = turn.status === "running";
  const nodes = view ? view.trace : liveTrace(turn.steps);
  const [openState, setOpen] = useState<boolean | null>(null);
  // Open while tools run; fold away once answer text starts streaming so the text stays in view.
  const open = openState ?? (live && !turn.draft);
  const panelId = useId();
  if (!nodes.length && !live) return null;
  const stats = traceStats(nodes);
  const ms = view?.serverMs ?? view?.wallMs ?? (liveState ? liveState.elapsed * 1000 : undefined);
  const liveLabel = liveState ? t(`progress.activity.${liveState.progress.activity}`) : t("trace.live");
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
        <span className={cn("size-1.5 shrink-0 rounded-full", live ? "bg-gilt motion-safe:animate-pulse" : "bg-cobalt")} aria-hidden />
        <span className="min-w-0 truncate font-medium">{live ? <span className="shimmer">{liveLabel}…</span> : t("trace.title")}</span>
        <span className={cn("text-faint tabular-nums", liveState && !view && "hidden sm:inline")}>
          {t("trace.summary", {
            steps: stats.steps,
            tools: stats.tools,
            ms: ms === undefined ? "…" : liveState && !view ? t("progress.elapsed", { s: liveState.elapsed }) : formatMs(ms),
          })}
        </span>
        {liveState && !view && (
          // Phones show only the elapsed time next to the step; the counts wrap badly at 390 px.
          <span className="shrink-0 text-faint tabular-nums sm:hidden">{t("progress.elapsed", { s: liveState.elapsed })}</span>
        )}
        <ChevronDown className={cn("ml-auto size-4 shrink-0 text-faint transition-transform", open && "rotate-180")} aria-hidden />
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

function StreamingAnswer({ draft }: { draft: string }) {
  const text = streamingText(draft);
  return (
    // Tokens render silently (aria-busy, no live region): the status region announces the step instead.
    <div className="streaming-answer" data-streaming="true" aria-busy="true">
      {text.trim() ? (
        <RichText text={text} className="answer-text streaming-text text-[15px] leading-[1.8] text-ink" />
      ) : (
        <p className="streaming-text text-[15px] leading-[1.8]" aria-hidden />
      )}
    </div>
  );
}

function EditedNotice() {
  const { t } = useI18n();
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const timer = window.setTimeout(() => setVisible(false), 6000);
    return () => window.clearTimeout(timer);
  }, []);
  return (
    <AnimatePresence>
      {visible && (
        <motion.span
          key="edited"
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="inline-flex"
        >
          <Tooltip content={t("answer.editedHint")}>
            <Badge tone="gilt" tabIndex={0} className="answer-edited">
              <FilePenLine />
              {t("answer.edited")}
            </Badge>
          </Tooltip>
        </motion.span>
      )}
    </AnimatePresence>
  );
}

function Limitations({ view, cite }: { view: AnswerView; cite: CiteHandler }) {
  const { lang, t } = useI18n();
  return (
    <section className="limitations rounded-lg bg-surface-2/70 px-3 py-2">
      <h2 className="mb-1 flex items-center gap-1.5 text-[12.5px] font-medium text-muted">
        <AlertTriangle className="size-3.5 text-warn" aria-hidden />
        {t(view.kind === "classic" ? "run.warnings" : "answer.limitations")}
      </h2>
      <ul className="list-disc space-y-0.5 pl-5 text-[12.5px] leading-relaxed text-muted marker:text-faint">
        {view.limitations.map((item, i) => {
          const { text, code } = limitationText(lang, item);
          return (
            <li key={`${i}-${item}`}>
              {code ? <CodeText code={code} label={text} /> : <RichText text={text} cite={cite} className="inline [&>p]:inline" />}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function AnswerCard({
  turn,
  view,
  isLast,
  activeEvidence,
  themeKey,
  number,
  onCite,
  onInspect,
  onAsk,
  onFeedback,
  onCheckClaim,
}: Props & { view: AnswerView }) {
  const { lang, t } = useI18n();
  const streamed = Boolean(turn.draft?.trim());
  const edited = view.kind === "agent" && answerEdited(turn.draft, view.answer);
  const freshness = useMemo(() => summarizeFreshness(view.evidence, view.cited), [view.evidence, view.cited]);
  const sourceFreshness = useMemo(
    () => new Map(view.evidence.map((source) => [source.evidence_id, evidenceFreshness(source)])),
    [view.evidence],
  );
  const query = view.agent?.query || turn.query;
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
    <article
      className="answer-card space-y-3.5 rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5"
      aria-label={t("a11y.assistant")}
      data-streamed={streamed || undefined}
      data-edited={edited || undefined}
    >
      <header className="flex flex-wrap items-center gap-1.5">
        <RouteBadge view={view} />
        <VerificationBadge view={view} />
        {edited && <EditedNotice />}
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
        {view.wallMs !== undefined && (
          <span className="ml-auto text-[11.5px] text-muted tabular-nums">{formatMs(view.wallMs)}</span>
        )}
      </header>

      {warn && (
        <p className="flex items-center gap-1.5 rounded-lg bg-warn-soft px-3 py-2 text-[13px] text-warn">
          <Info className="size-4 shrink-0" aria-hidden />
          {t("answer.classicFallback")}
        </p>
      )}

      {view.kind === "agent" && <TraceDisclosure view={view} turn={turn} onEvidence={(id) => onCite(turn.id, id)} />}

      <FreshnessBanner summary={freshness} turn={number} onReview={() => onInspect(turn.id, "evidence")} />

      <motion.div
        initial={streamed ? { opacity: 0.4, filter: "blur(2px)" } : false}
        animate={{ opacity: 1, filter: "blur(0px)" }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      >
        <RichText text={view.answer} cite={cite} className="answer-text text-[15px] leading-[1.8] text-ink" />
      </motion.div>

      {view.factCheck && <InlineFactCheck report={view.factCheck} number={number} onOpen={onCheckClaim} />}

      {view.data && (
        <DataPanel
          data={view.data}
          themeKey={themeKey}
          turn={number}
          freshness={sourceFreshness}
          displayName={(name) => displayName(view.englishNames, lang, name)}
          onEvidence={(id) => onCite(turn.id, id)}
        />
      )}

      {view.keyPoints.length > 0 && (
        <section className="key-points">
          <h2 className="mb-1.5 text-[13px] font-medium text-muted">{t("answer.keyPoints")}</h2>
          <RichText text={view.keyPoints.map((point) => `- ${point}`).join("\n")} cite={cite} className="text-[14px] leading-relaxed" />
        </section>
      )}

      {view.limitations.length > 0 && <Limitations view={view} cite={cite} />}

      {view.sentiment && <SentimentBar sentiment={view.sentiment} />}

      {isLast && view.next.length > 0 && (
        <section aria-label={number === undefined ? t("answer.next") : t("a11y.inTurn", { label: t("answer.next"), n: number })}>
          <h2 className="mb-1.5 text-[13px] font-medium text-muted">{t("answer.next")}</h2>
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

      <footer className="space-y-2 border-t border-line pt-3">
        <p className="disclaimer flex items-start gap-1.5 text-[12px] leading-relaxed text-muted">
          <ShieldCheck className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <span>{view.disclaimer || t("answer.disclaimerDefault")}</span>
        </p>
        <div className="answer-actions flex flex-wrap items-start gap-1" role="group" aria-label={t("answer.actions")}>
          {view.agent?.trace_id ? (
            <FeedbackControls traceId={view.agent.trace_id} sessionId={view.agent.session_id ?? null} onSend={onFeedback} />
          ) : null}
          <span className="ml-auto flex items-center gap-0.5">
            <CopyButton text={copyText} />
            <ExportMenu view={view} query={query} />
          </span>
        </div>
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

/**
 * A turn that is still running. Before the first answer token it shows the progress panel (step, tools,
 * elapsed time, answer skeleton); once text streams, the panel folds into the collapsed run trace above
 * the streaming answer. One polite status region announces step changes; nothing else here is live.
 */
function RunningCard({ turn, onStop }: { turn: Turn; onStop?: () => void }) {
  const { t } = useI18n();
  const reduceMotion = useReducedMotion();
  const staged = turn.via === "stream";
  const progress = useMemo(() => turnProgress(turn), [turn]);
  const elapsed = useElapsed(turn.startedAt);
  const status = staged
    ? t("progress.status", { phase: t(`progress.phase.${progress.phase}`), activity: t(`progress.activity.${progress.activity}`) })
    : t("app.running");
  return (
    <div
      className="answer-card running-card rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5"
      data-phase={staged ? progress.phase : undefined}
    >
      <p role="status" className="progress-status sr-only">
        {status}
      </p>
      <div aria-live="off">
        {/* "wait": the panel fades out before the trace and the text take its place, so the two never stack. */}
        <AnimatePresence initial={false} mode="wait">
          {progress.streaming ? (
            <motion.div
              key="stream"
              className="space-y-3.5"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: reduceMotion ? 0 : 0.18 }}
            >
              <TraceDisclosure turn={turn} live={{ progress, elapsed }} onEvidence={() => undefined} />
              <StreamingAnswer draft={turn.draft ?? ""} />
            </motion.div>
          ) : (
            <motion.div key="progress" exit={{ opacity: 0 }} transition={{ duration: reduceMotion ? 0 : 0.15 }}>
              <RunProgress progress={progress} elapsed={elapsed} staged={staged} onStop={onStop} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export function TurnView(props: Props) {
  const { turn, view, onCheckClaim } = props;
  const { t } = useI18n();
  // The server's inline check replaces the hint once the answer carries one.
  const claim = useMemo(
    () => (onCheckClaim && !view?.factCheck ? claimInMessage(turn.query) : null),
    [onCheckClaim, turn.query, view?.factCheck],
  );
  return (
    <motion.div
      className="turn space-y-3"
      data-turn-id={turn.id}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
    >
      <UserBubble turn={turn} />
      {claim && onCheckClaim && <ClaimHint claim={claim} onCheck={onCheckClaim} />}
      {turn.status === "running" && <RunningCard turn={turn} onStop={props.onStop} />}
      {turn.status === "clarify" && <ClarificationCard turn={turn} />}
      {turn.status === "error" && <ErrorCard turn={turn} onRetry={props.onRetry} />}
      {turn.status === "stopped" && (
        <p className="stopped-note flex items-center gap-1.5 text-[13px] text-faint">
          <Square className="size-3" aria-hidden />
          {turn.finishedAt !== undefined
            ? t("progress.stoppedAfter", { s: Math.max(0, Math.round((turn.finishedAt - turn.startedAt) / 1000)) })
            : t("answer.stopped")}
        </p>
      )}
      {turn.status === "done" && view && <AnswerCard {...props} view={view} />}
    </motion.div>
  );
}
