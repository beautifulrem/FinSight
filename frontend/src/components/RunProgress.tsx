import { Check, CircleDashed, Square, X } from "lucide-react";

import { cn } from "@/lib/cn";
import { formatMs } from "@/lib/format";
import { toolLabel, useI18n } from "@/lib/i18n";
import { PHASES, phaseState, type Progress } from "@/lib/progress";

import { Button } from "./ui/button";

/** Tool rows shown before older ones fold into a count (keeps the panel short on phones). */
const MAX_TOOLS = 5;
/** After this many seconds without answer text, say how long model reasoning usually takes. */
export const SLOW_AFTER_S = 8;

interface Props {
  progress: Progress;
  elapsed: number;
  /** Streamed turns report graph steps; classic / resume requests only have the elapsed time. */
  staged: boolean;
  onStop?: () => void;
}

function StepIndicator({ progress }: { progress: Progress }) {
  const { t } = useI18n();
  return (
    <ol className="progress-steps grid grid-cols-4 gap-1.5" aria-label={t("progress.title")}>
      {PHASES.map((phase) => {
        const state = phaseState(progress.phase, phase);
        return (
          <li key={phase} className="min-w-0" data-phase={phase} data-state={state} aria-current={state === "active" ? "step" : undefined}>
            <span
              aria-hidden
              className={cn(
                "block h-1 rounded-full transition-colors duration-300 motion-reduce:transition-none",
                state === "done" && "bg-cobalt",
                state === "active" && "bg-gilt",
                state === "pending" && "bg-line",
              )}
            />
            <span
              className={cn(
                "mt-1.5 flex items-center gap-1 text-[12px] leading-tight",
                state === "done" && "text-muted",
                state === "active" && "font-medium text-ink",
                state === "pending" && "text-faint",
              )}
            >
              {state === "done" && <Check className="size-3 shrink-0 text-cobalt" strokeWidth={3} aria-hidden />}
              <span className="truncate">{t(`progress.phase.${phase}`)}</span>
              <span className="sr-only">({t(`progress.state.${state}`)})</span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function ToolList({ progress }: { progress: Progress }) {
  const { lang, t } = useI18n();
  const hidden = Math.max(0, progress.tools.length - MAX_TOOLS);
  const shown = progress.tools.slice(hidden);
  return (
    <ul className="progress-tools space-y-0.5" aria-label={t("progress.tools")}>
      {hidden > 0 && <li className="pl-6 text-[12px] text-faint">{t("progress.moreTools", { n: hidden })}</li>}
      {shown.map((tool) => (
        <li key={tool.id} className="progress-tool flex min-w-0 items-center gap-2 py-0.5 text-[13px]" data-status={tool.status}>
          <span
            aria-hidden
            className={cn(
              "grid size-4 shrink-0 place-items-center rounded-full",
              tool.status === "ok" && "bg-down-soft text-down",
              tool.status === "error" && "bg-up-soft text-up",
              tool.status === "running" && "text-cobalt",
            )}
          >
            {tool.status === "ok" ? (
              <Check className="size-3" strokeWidth={3} />
            ) : tool.status === "error" ? (
              <X className="size-3" strokeWidth={3} />
            ) : (
              <CircleDashed className="size-3.5 motion-safe:animate-spin" />
            )}
          </span>
          <span className="shrink-0 font-medium whitespace-nowrap text-ink">{toolLabel(lang, tool.tool)}</span>
          <code className="hidden shrink-0 font-mono text-[11.5px] text-faint sm:inline">{tool.tool}</code>
          {tool.target && <span className="progress-target min-w-0 truncate text-muted">· {tool.target}</span>}
          {tool.latencyMs !== undefined && (
            <span className="ml-auto shrink-0 pl-1 text-[11.5px] text-faint tabular-nums">{formatMs(tool.latencyMs)}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

/**
 * The "working" state of a turn before its first answer token: the current step in plain language, the
 * tools being called with their target, the elapsed time and a skeleton where the answer will appear.
 * Screen readers get the step from the turn's status region (TurnView), not from this panel.
 */
export function RunProgress({ progress, elapsed, staged, onStop }: Props) {
  const { t } = useI18n();
  const activity = staged ? t(`progress.activity.${progress.activity}`) : t("app.running");
  const slow = staged && progress.usesLlm && !progress.streaming && elapsed >= SLOW_AFTER_S;
  return (
    <div className="run-progress space-y-3" data-phase={progress.phase} data-activity={progress.activity}>
      <div className="flex min-h-8 items-center gap-2 text-[13.5px]">
        <span aria-hidden className="size-2 shrink-0 rounded-full bg-gilt motion-safe:animate-pulse" />
        <span className="progress-activity min-w-0 flex-1 font-medium text-ink">{activity}…</span>
        <span className="progress-elapsed shrink-0 text-[12px] text-muted tabular-nums">
          <span className="sr-only">{t("progress.elapsedLabel")} </span>
          {t("progress.elapsed", { s: elapsed })}
        </span>
        {onStop && (
          <Button size="sm" variant="ghost" className="progress-stop -mr-1.5 shrink-0 [&_svg]:size-3" onClick={onStop}>
            <Square aria-hidden />
            {t("composer.stop")}
          </Button>
        )}
      </div>
      {staged && <StepIndicator progress={progress} />}
      {progress.tools.length > 0 && <ToolList progress={progress} />}
      {slow && <p className="progress-hint text-[12.5px] leading-relaxed text-muted">{t("progress.slow")}</p>}
      <div aria-hidden className="answer-skeleton space-y-2.5 pt-1 motion-safe:animate-pulse">
        <div className="h-2.5 w-[92%] rounded-full bg-surface-2" />
        <div className="h-2.5 w-[78%] rounded-full bg-surface-2" />
        <div className="h-2.5 w-[56%] rounded-full bg-surface-2" />
      </div>
    </div>
  );
}
