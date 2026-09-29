import { evidenceTitle } from "@/lib/format";
import { m as motion } from "motion/react";
import {
  AlertTriangle,
  CircleCheck,
  CircleQuestionMark,
  CircleX,
  Database,
  LoaderCircle,
  Quote,
  RotateCcw,
  SearchCheck,
  SearchX,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestionMark,
  ShieldX,
} from "lucide-react";
import {
  useEffect,
  useId,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type Ref,
} from "react";

import { checkClaim, classifyError, type ErrorKind } from "@/lib/api";
import { checkEvidence, claimedText, formatClaimValue, noteText, statusCounts, targetName } from "@/lib/claims";
import { cn } from "@/lib/cn";
import { humanizeCode } from "@/lib/codes";
import { evidenceFreshness } from "@/lib/freshness";
import { sourceNameLabel, sourceTypeLabel, useI18n, type MessageKey } from "@/lib/i18n";
import type { ClaimCheckItem, ClaimReport, ClaimStatus, ClaimVerdict } from "@/lib/types";

import { AsOf } from "./Freshness";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Tooltip } from "./ui/tooltip";

const MAX_CLAIM = 2000;
const EXAMPLES: MessageKey[] = ["claim.ex1", "claim.ex2", "claim.ex3"];

const VERDICTS: Record<ClaimVerdict, { icon: typeof ShieldCheck; tone: "down" | "up" | "warn" | "neutral"; frame: string }> = {
  supported: { icon: ShieldCheck, tone: "down", frame: "border-down/35 bg-down-soft/60" },
  contradicted: { icon: ShieldX, tone: "up", frame: "border-up/35 bg-up-soft/60" },
  partially_supported: { icon: ShieldAlert, tone: "warn", frame: "border-warn/35 bg-warn-soft/60" },
  unverifiable: { icon: ShieldQuestionMark, tone: "neutral", frame: "border-line bg-surface-2/60" },
};

const STATUSES: Record<ClaimStatus, { icon: typeof CircleCheck; tone: "down" | "up" | "neutral"; edge: string; actual: string }> = {
  supported: { icon: CircleCheck, tone: "down", edge: "border-l-down", actual: "text-down" },
  contradicted: { icon: CircleX, tone: "up", edge: "border-l-up", actual: "text-up" },
  unverifiable: { icon: CircleQuestionMark, tone: "neutral", edge: "border-l-line", actual: "text-muted" },
};

const COLOR: Record<string, string> = { down: "text-down", up: "text-up", warn: "text-warn", neutral: "text-muted" };

/** Overall verdict: colour, icon and label (never colour alone). */
export function VerdictBadge({ verdict, className }: { verdict: ClaimVerdict; className?: string }) {
  const { t } = useI18n();
  const spec = VERDICTS[verdict] ?? VERDICTS.unverifiable;
  return (
    <Badge tone={spec.tone} className={cn("claim-verdict gap-1.5 px-2.5 py-1 text-[14px] leading-5 [&_svg]:size-4", className)} data-verdict={verdict}>
      <spec.icon aria-hidden />
      {t(`claim.verdict.${verdict}` as MessageKey)}
    </Badge>
  );
}

function StatusBadge({ status }: { status: ClaimStatus }) {
  const { t } = useI18n();
  const spec = STATUSES[status] ?? STATUSES.unverifiable;
  return (
    <Badge tone={spec.tone} className="claim-status shrink-0" data-status={status}>
      <spec.icon aria-hidden />
      {t(`claim.status.${status}` as MessageKey)}
    </Badge>
  );
}

function CheckRow({ check, report }: { check: ClaimCheckItem; report: ClaimReport }) {
  const { lang, t } = useI18n();
  const spec = STATUSES[check.status] ?? STATUSES.unverifiable;
  const evidence = checkEvidence(check, report);
  const info = evidence ? evidenceFreshness(evidence) : null;
  const metric = check.metric ? humanizeCode(lang, check.metric, "metric") : t("claim.unknownMetric");
  const note = noteText(t, check.note, check);
  const hasActual = check.actual !== null && check.actual !== undefined;
  const name = targetName(lang, report);
  const claimed = claimedText(lang, t, check, report.claim, name);
  const source =
    sourceNameLabel(lang, evidence?.provenance ?? null, evidence?.source_name) ||
    (evidence ? sourceTypeLabel(lang, evidence.source_type) : null);
  return (
    <li
      className={cn("claim-check rounded-xl border border-l-4 border-line bg-surface p-3.5 sm:p-4", spec.edge)}
      data-status={check.status}
      data-metric={check.metric ?? ""}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h4 className="claim-metric text-[14.5px] leading-snug font-semibold text-ink">{metric}</h4>
          <p className="claim-target-name text-[12.5px] text-muted">{check.target ? name(check.target) : t("claim.unknownTarget")}</p>
        </div>
        <StatusBadge status={check.status} />
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-2">
        <div className="rounded-lg bg-surface-2/70 px-3 py-2">
          <dt className="text-[11.5px] font-medium text-muted">{t("claim.claimed")}</dt>
          <dd
            className={cn(
              "claim-claimed text-[17px] leading-7 font-medium tabular-nums",
              check.status === "contradicted" ? "text-muted line-through decoration-up/60 decoration-2" : "text-ink",
            )}
            data-comparator={check.comparator ?? "eq"}
            data-direction={check.direction ?? undefined}
          >
            {/* The symbol (">", "≠", "≈") is for the eye; screen readers get the word ("高于 30%"). */}
            {claimed.label === claimed.text ? (
              claimed.text
            ) : (
              <>
                <span aria-hidden>{claimed.text}</span>
                <span className="sr-only">{claimed.label}</span>
              </>
            )}
          </dd>
          {claimed.detail && <dd className="claim-reference text-[12px] text-muted tabular-nums">{claimed.detail}</dd>}
        </div>
        <div className="rounded-lg bg-surface-2/70 px-3 py-2">
          <dt className="text-[11.5px] font-medium text-muted">{t("claim.actual")}</dt>
          <dd className={cn("claim-actual text-[17px] leading-7 font-semibold tabular-nums", hasActual ? spec.actual : "text-muted")}>
            {hasActual ? formatClaimValue(lang, t, check.metric, check.actual!, "actual") : t("claim.noActual")}
          </dd>
        </div>
      </dl>

      {note && <p className="claim-note mt-2.5 text-[12.5px] leading-relaxed text-muted">{note}</p>}

      {evidence && info && (
        <div className="claim-source mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-line/70 pt-2.5 text-[12px]">
          <span className="inline-flex items-center gap-1 text-muted">
            <Database className="size-3" aria-hidden />
            <Tooltip content={evidenceTitle(lang, evidence.title) || null}>
              <span tabIndex={evidence.title ? 0 : undefined} className="font-medium text-ink">
                {source || t("claim.sourceUnknown")}
              </span>
            </Tooltip>
          </span>
          <AsOf info={info} />
          {check.as_of_basis && (
            <span className="claim-basis text-muted">({t(`claim.basis.${check.as_of_basis}` as MessageKey)})</span>
          )}
          <code className="ml-auto min-w-0 truncate font-mono text-[11px] text-muted" title={t("claim.evidence", { id: evidence.evidence_id })}>
            {evidence.evidence_id}
          </code>
        </div>
      )}
    </li>
  );
}

/**
 * The report for one claim: verdict, targets, one card per number, and the disclaimer. `turn` (in the chat)
 * gives its checks region a name unique to the turn (axe landmark-unique).
 */
export function ClaimReportCard({ report, turn }: { report: ClaimReport; turn?: number }) {
  const { lang, t } = useI18n();
  const headingId = useId();
  const spec = VERDICTS[report.verdict] ?? VERDICTS.unverifiable;
  const counts = statusCounts(report);
  const targets = (report.targets ?? []).filter((target) => target.name || target.symbol);
  const summary = [
    t("claim.count.total", { n: report.checks.length }),
    ...(["supported", "contradicted", "unverifiable"] as const)
      .filter((status) => counts[status] > 0)
      .map((status) => t(`claim.count.${status}`, { n: counts[status] })),
  ].join(" · ");
  return (
    <article className="claim-report space-y-4 rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5" aria-labelledby={headingId}>
      <h2 id={headingId} className="sr-only">
        {t("claim.result")}
      </h2>
      <header className={cn("space-y-2 rounded-xl border px-3.5 py-3", spec.frame)}>
        <div className="flex flex-wrap items-center gap-2">
          <span className="sr-only">{t("claim.verdict")}: </span>
          <VerdictBadge verdict={report.verdict} />
          {report.checks.length > 0 && <span className="claim-counts text-[12.5px] text-muted tabular-nums">{summary}</span>}
        </div>
        <p className={cn("text-[13px] leading-relaxed", COLOR[spec.tone])}>{t(`claim.verdictHint.${report.verdict}` as MessageKey)}</p>
      </header>

      <figure className="flex gap-2 text-[15px] leading-relaxed text-ink">
        <Quote className="mt-1 size-4 shrink-0 text-faint" aria-hidden />
        <blockquote className="claim-text min-w-0 break-words">{report.claim}</blockquote>
      </figure>

      {targets.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-[12.5px]">
          <span className="text-muted">{t("claim.targets")}:</span>
          {targets.map((target) => (
            <Badge key={`${target.symbol}-${target.name}`} tone="cobalt" className="claim-target">
              {lang === "en" ? (target.name_en ?? target.name) : target.name}
              {target.symbol && <span className="font-mono text-[11px]">{target.symbol}</span>}
            </Badge>
          ))}
        </div>
      )}

      {report.checks.length > 0 ? (
        <section
          className="space-y-2"
          aria-label={turn === undefined ? t("claim.checks") : t("a11y.inTurn", { label: t("claim.checks"), n: turn })}
        >
          <h3 className="text-[13px] font-medium text-muted">{t("claim.checks")}</h3>
          <ul className="claim-checks space-y-2.5">
            {report.checks.map((check, i) => (
              <CheckRow key={`${i}-${check.metric}-${check.claimed}`} check={check} report={report} />
            ))}
          </ul>
        </section>
      ) : (
        <div className="claim-empty flex items-start gap-2.5 rounded-xl border border-dashed border-line px-3.5 py-3">
          <SearchX className="mt-0.5 size-4 shrink-0 text-faint" aria-hidden />
          <div className="space-y-1">
            <p className="text-[14px] font-medium">{t("claim.emptyTitle")}</p>
            <p className="text-[13px] leading-relaxed text-muted">{t("claim.emptyBody")}</p>
          </div>
        </div>
      )}

      <footer className="border-t border-line pt-3">
        <p className="claim-disclaimer disclaimer flex items-start gap-1.5 text-[12px] leading-relaxed text-muted">
          <ShieldCheck className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <span>{t("claim.disclaimer")}</span>
        </p>
      </footer>
    </article>
  );
}

function LoadingCard({ claim }: { claim: string }) {
  const { t } = useI18n();
  return (
    <div className="claim-loading space-y-4 rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5" aria-busy="true">
      <p className="flex items-center gap-2 text-[14px]">
        <LoaderCircle className="size-4 animate-spin text-cobalt motion-reduce:animate-none" aria-hidden />
        <span className="shimmer font-medium">{t("claim.checking")}…</span>
      </p>
      <p className="text-[14px] leading-relaxed break-words text-muted">“{claim}”</p>
      <div className="space-y-2.5" aria-hidden>
        {[0, 1].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl bg-surface-2 motion-reduce:animate-none" />
        ))}
      </div>
    </div>
  );
}

function ErrorBox({ kind, message, onRetry }: { kind: ErrorKind; message: string; onRetry: () => void }) {
  const { t } = useI18n();
  const text =
    kind === "timeout"
      ? t("claim.errorTimeout")
      : kind === "invalid"
        ? t("claim.errorInvalid")
        : kind === "generic"
          ? t("error.generic", { e: message })
          : t(`error.${kind}` as MessageKey);
  return (
    <div role="alert" className="claim-error error-box flex items-start gap-2 rounded-xl border border-up/30 bg-up-soft px-4 py-3 text-[14px] text-up" data-kind={kind}>
      <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
      <span className="min-w-0 flex-1 break-words">{text}</span>
      <Button size="sm" variant="outline" onClick={onRetry}>
        <RotateCcw />
        {t("error.retry")}
      </Button>
    </div>
  );
}

type State =
  | { status: "idle" }
  | { status: "loading"; claim: string }
  | { status: "error"; claim: string; kind: ErrorKind; message: string }
  | { status: "done"; report: ClaimReport };

/** Lets the chat hand a claim over ("核查这句话"): fill the box and check it. */
export interface ClaimCheckHandle {
  check: (claim: string) => void;
}

/** "核查 / Fact-check": paste a market claim and see each number checked against the data. */
export function ClaimCheckView({ apiKey, ref }: { apiKey: string; ref?: Ref<ClaimCheckHandle> }) {
  const { lang, t } = useI18n();
  const [value, setValue] = useState("");
  const [hint, setHint] = useState(false);
  const [state, setState] = useState<State>({ status: "idle" });
  const textarea = useRef<HTMLTextAreaElement>(null);
  const controller = useRef<AbortController | null>(null);
  const results = useRef<HTMLDivElement>(null);
  const busy = state.status === "loading";

  useEffect(() => () => controller.current?.abort(), []);

  // The form and examples fill a phone screen: bring the report (or error) into view when it arrives.
  useEffect(() => {
    if (state.status !== "done" && state.status !== "error") return;
    const element = results.current;
    if (!element || element.getBoundingClientRect().top < window.innerHeight * 0.6) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    element.scrollIntoView?.({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }, [state]);

  useLayoutEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);

  const run = async (claim: string) => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setState({ status: "loading", claim });
    try {
      const report = await checkClaim(claim, lang, { apiKey, signal: current.signal });
      if (!current.signal.aborted) setState({ status: "done", report });
    } catch (error) {
      if (current.signal.aborted) return;
      setState({ status: "error", claim, ...classifyError(error) });
    }
  };

  useImperativeHandle(ref, () => ({
    check: (claim: string) => {
      setValue(claim);
      setHint(false);
      void run(claim);
    },
  }));

  const submit = (text = value) => {
    if (busy) return;
    const claim = text.trim();
    if (claim.length < 2) {
      setHint(true);
      textarea.current?.focus();
      return;
    }
    setHint(false);
    void run(claim);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter checks; Shift+Enter adds a line; ignore Enter while an IME (pinyin) is composing.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      submit();
    }
  };

  const announcement =
    state.status === "done"
      ? t("claim.done", { v: t(`claim.verdict.${state.report.verdict}` as MessageKey) })
      : state.status === "loading"
        ? `${t("claim.checking")}…`
        : "";

  return (
    <div className="claim-check-view mx-auto w-full max-w-3xl space-y-6 px-3 py-5 sm:px-5 sm:py-8">
      <div className="space-y-2.5">
        <h1 className="flex items-center gap-2 text-[24px] leading-tight font-semibold tracking-tight text-balance sm:text-[28px]">
          <SearchCheck className="size-6 shrink-0 text-cobalt" aria-hidden />
          {t("claim.title")}
        </h1>
        <p className="max-w-[40rem] text-[14.5px] leading-relaxed text-muted">{t("claim.body")}</p>
      </div>

      <form
        id="claim-form"
        className="rounded-2xl border border-line bg-surface shadow-card transition-shadow focus-within:border-cobalt/60 focus-within:ring-4 focus-within:ring-cobalt/10"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <label htmlFor="claim-input" className="block px-4 pt-3 text-[12.5px] font-medium text-muted">
          {t("claim.label")}
        </label>
        <textarea
          id="claim-input"
          ref={textarea}
          rows={2}
          value={value}
          maxLength={MAX_CLAIM}
          onChange={(event) => {
            setValue(event.target.value);
            if (hint) setHint(false);
          }}
          onKeyDown={onKeyDown}
          placeholder={t("claim.placeholder")}
          aria-describedby="claim-hint"
          aria-invalid={hint || undefined}
          className="block w-full resize-none bg-transparent px-4 pt-1 pb-1 text-[15px] leading-relaxed text-ink outline-none placeholder:text-faint"
          autoComplete="off"
          enterKeyHint="go"
        />
        <div className="flex items-center gap-2 px-2.5 pt-1 pb-2.5">
          <span id="claim-hint" className={hint ? "px-1.5 text-[12px] text-up" : "hidden px-1.5 text-[12px] text-faint sm:inline"}>
            {hint ? t("claim.tooShort") : t("claim.hint")}
          </span>
          <Button id="claim-submit" type="submit" variant="primary" size="sm" className="ml-auto" disabled={busy}>
            {busy ? <LoaderCircle className="animate-spin motion-reduce:animate-none" aria-hidden /> : <SearchCheck aria-hidden />}
            <span>{busy ? t("claim.checking") : t("claim.submit")}</span>
          </Button>
        </div>
      </form>

      <section className="space-y-2.5" aria-labelledby="claim-examples-title">
        <h2 id="claim-examples-title" className="text-[13px] font-medium text-muted">
          {t("claim.examples")}
        </h2>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((key) => (
            <button
              key={key}
              type="button"
              disabled={busy}
              onClick={() => {
                const text = t(key);
                setValue(text);
                submit(text);
              }}
              className="claim-example rounded-full border border-line bg-surface px-3.5 py-1.5 text-left text-[13.5px] leading-snug text-ink transition-colors hover:border-cobalt hover:bg-cobalt-soft hover:text-cobalt disabled:opacity-50"
            >
              {t(key)}
            </button>
          ))}
        </div>
      </section>

      <p role="status" className="sr-only">
        {announcement}
      </p>

      {/* No exit animation: the report must be in place when it is scrolled into view. */}
      <div ref={results} className="scroll-mt-4">
        {state.status !== "idle" && (
          <motion.div
            key={state.status === "done" ? `done-${state.report.claim}` : state.status}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            {state.status === "loading" && <LoadingCard claim={state.claim} />}
            {state.status === "error" && <ErrorBox kind={state.kind} message={state.message} onRetry={() => void run(state.claim)} />}
            {state.status === "done" && <ClaimReportCard report={state.report} />}
          </motion.div>
        )}
      </div>
    </div>
  );
}
