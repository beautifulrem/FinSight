import { AnimatePresence, m as motion } from "motion/react";
import { AlertTriangle, Check, ChevronRight, CircleDashed, ShieldCheck, Sparkles, Wrench, X } from "lucide-react";
import { useId, useState } from "react";

import { cn } from "@/lib/cn";
import { formatInt, formatMs } from "@/lib/format";
import { humanizeCode } from "@/lib/codes";
import { nodeLabel, toolLabel, useI18n } from "@/lib/i18n";
import { formatArgs, type TraceNode, type TraceTool } from "@/lib/trace";
import type { AgentResponse } from "@/lib/types";

import { CodeBadge } from "./CodeLabel";
import { Badge } from "./ui/badge";

interface Props {
  nodes: TraceNode[];
  live?: boolean;
  response?: AgentResponse;
  onEvidence?: (id: string) => void;
}

function ToolRow({ tool, onEvidence }: { tool: TraceTool; onEvidence?: (id: string) => void }) {
  const { lang, t } = useI18n();
  const [open, setOpen] = useState(false);
  const detailId = useId();
  const args = formatArgs(tool.args);
  return (
    <li className="tool-row">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls={detailId}
        className="group flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-[13px] hover:bg-surface-2"
      >
        <span
          className={cn(
            "grid size-4 shrink-0 place-items-center rounded-full",
            tool.status === "ok" && "bg-down-soft text-down",
            tool.status === "error" && "bg-up-soft text-up",
            tool.status === "running" && "text-cobalt",
          )}
          aria-hidden
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
        <code className="hidden min-w-0 truncate font-mono text-[11.5px] text-faint sm:inline">
          {tool.tool}({args.length > 48 ? `${args.slice(0, 48)}…` : args})
        </code>
        <span className="ml-auto flex shrink-0 items-center gap-1.5 text-[11.5px] text-faint tabular-nums">
          {tool.cached && <Badge tone="cobalt">{t("trace.cached")}</Badge>}
          {tool.source === "llm" && <Badge tone="gilt">{t("trace.byLlm")}</Badge>}
          {tool.error && <Badge tone="up" data-code={tool.error.code}>{humanizeCode(lang, tool.error.code, "toolError")}</Badge>}
          {tool.latencyMs !== undefined && formatMs(tool.latencyMs)}
          <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} aria-hidden />
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={detailId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <dl className="ml-7 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 py-1.5 text-[12px]">
              <dt className="text-faint">{t("trace.args")}</dt>
              <dd className="min-w-0 font-mono break-all text-muted">{args || "{}"}</dd>
              {tool.reason && (
                <>
                  <dt className="text-faint">{t("trace.reasons")}</dt>
                  <dd className="text-muted">{tool.reason}</dd>
                </>
              )}
              {tool.attempts && tool.attempts > 1 ? (
                <>
                  <dt className="text-faint">retry</dt>
                  <dd className="text-muted">{t("trace.attempts", { n: tool.attempts })}</dd>
                </>
              ) : null}
              {tool.error && (
                <>
                  <dt className="text-up">{humanizeCode(lang, tool.error.code, "toolError")}</dt>
                  <dd className="text-muted">
                    {tool.error.message} <code className="font-mono text-[11px]">({tool.error.code})</code>
                  </dd>
                </>
              )}
              {tool.evidenceIds.length > 0 && (
                <>
                  <dt className="text-faint">{t("trace.evidence")}</dt>
                  <dd className="flex flex-wrap gap-1">
                    {tool.evidenceIds.map((id) => (
                      <button
                        key={id}
                        type="button"
                        onClick={() => onEvidence?.(id)}
                        className="rounded bg-cobalt-soft px-1.5 font-mono text-[11px] text-cobalt hover:underline"
                      >
                        {id}
                      </button>
                    ))}
                  </dd>
                </>
              )}
            </dl>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

function NodeDetails({ node, response, finalVerify }: { node: TraceNode; response?: AgentResponse; finalVerify: boolean }) {
  const { lang, t } = useI18n();
  if (!response) return null;
  if (node.node === "guard_in" && response.route) {
    return (
      <div className="flex flex-wrap items-center gap-1 text-[12px]">
        <Badge tone="cobalt" data-code={response.route}>
          {t("trace.route")}: {humanizeCode(lang, response.route, "route")}
        </Badge>
        {(response.route_reasons ?? []).map((reason) => (
          <CodeBadge key={reason} code={reason} kind="reason" className="font-normal" />
        ))}
      </div>
    );
  }
  if (node.node === "verify" || node.node === "revise") {
    const v = response.verification ?? {};
    // The response carries only the final verification result, so show it on the last check.
    if (node.node === "revise" || !finalVerify || v.passed === undefined || v.passed === null) return null;
    const repaired = (response.degraded ?? []).some((item) => item.startsWith("verification_failed"));
    return (
      <div className="flex flex-wrap items-center gap-1 text-[12px]">
        <Badge tone={v.passed ? "down" : repaired ? "warn" : "up"}>
          <ShieldCheck />
          {v.passed ? t("trace.verifyPassed") : repaired ? t("trace.verifyRepaired") : t("trace.verifyFailed")}
        </Badge>
        {v.checked_numbers ? <Badge>{t("trace.checkedNumbers", { n: v.checked_numbers })}</Badge> : null}
        {(v.invalid_citations ?? []).length > 0 && (
          <Badge tone="up">
            {t("trace.invalidCitations")}: {(v.invalid_citations ?? []).join(", ")}
          </Badge>
        )}
        {(v.misattributed_numbers ?? []).length > 0 && (
          <Badge tone="warn">
            {t("trace.misattributed", { n: String((v.misattributed_numbers ?? []).length) })}: {(v.misattributed_numbers ?? []).join(", ")}
          </Badge>
        )}
        {(v.unsupported_numbers ?? []).length > 0 && (
          <Badge tone="up">
            {t("trace.unsupported", { n: String((v.unsupported_numbers ?? []).length) })}: {(v.unsupported_numbers ?? []).join(", ")}
          </Badge>
        )}
      </div>
    );
  }
  if (node.node === "compliance" && (response.compliance_notes ?? []).length) {
    return (
      <div className="flex flex-wrap gap-1 text-[12px]">
        {(response.compliance_notes ?? []).map((note) => (
          <CodeBadge key={note} code={note} kind="compliance" tone="warn" className="font-normal" />
        ))}
      </div>
    );
  }
  if (node.llm) {
    const entry = node.llm;
    return (
      <div className="flex flex-wrap gap-1 text-[12px]">
        <Badge tone="gilt">
          <Sparkles />
          {entry.model ?? t("trace.llmCall")}
        </Badge>
        <Badge>
          {formatInt(lang, entry.prompt_tokens)} → {formatInt(lang, entry.completion_tokens)} tok
        </Badge>
        {entry.latency_ms !== undefined && <Badge>{formatMs(entry.latency_ms)}</Badge>}
        {(entry.tool_calls ?? []).length > 0 && (
          <Badge tone="cobalt">
            <Wrench />
            {(entry.tool_calls ?? []).join(", ")}
          </Badge>
        )}
      </div>
    );
  }
  if (node.node === "compose" && response.answer_source) {
    return (
      <Badge className="font-normal" data-code={response.answer_source}>
        {t("run.answerSource")}: {humanizeCode(lang, response.answer_source, "answerSource")}
      </Badge>
    );
  }
  return null;
}

/** Vertical run timeline: graph nodes in order, with the tools each one executed. */
export function TraceTimeline({ nodes, live, response, onEvidence }: Props) {
  const { lang, t } = useI18n();
  const degraded = response?.degraded ?? [];
  const lastVerify = nodes.map((node) => node.node).lastIndexOf("verify");
  return (
    // Not a live region: a running turn announces its steps through one status region (TurnView).
    <ol className="relative space-y-0.5">
      <AnimatePresence initial={false}>
        {nodes.map((node, index) => (
          <motion.li
            key={node.id}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="relative pl-6"
          >
            <span aria-hidden className="absolute top-0 bottom-0 left-[7px] w-px bg-line" />
            <span
              aria-hidden
              className={cn(
                "absolute top-[9px] left-[3px] size-[9px] rounded-full border-2 border-surface",
                node.tools.some((tool) => tool.status === "error") ? "bg-warn" : "bg-cobalt",
              )}
            />
            <div className="flex items-baseline gap-2 py-1">
              <span className="text-[13px] font-medium text-ink">{nodeLabel(lang, node.node)}</span>
              <code className="font-mono text-[11px] text-muted">{node.node}</code>
              {node.durationMs !== undefined && (
                <span className="ml-auto text-[11.5px] text-faint tabular-nums">{formatMs(node.durationMs)}</span>
              )}
            </div>
            <div className="space-y-1 pb-1">
              <NodeDetails node={node} response={response} finalVerify={index === lastVerify} />
              {node.tools.length > 0 && (
                <ul className="-ml-1.5 space-y-0.5">
                  {node.tools.map((tool) => (
                    <ToolRow key={tool.id} tool={tool} onEvidence={onEvidence} />
                  ))}
                </ul>
              )}
            </div>
          </motion.li>
        ))}
        {live && (
          <motion.li key="live" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative pl-6">
            <span aria-hidden className="absolute top-[9px] left-[3px] size-[9px] rounded-full bg-gilt motion-safe:animate-pulse" />
            <div className="py-1 text-[13px]">
              <span className="shimmer font-medium">{t("trace.live")}…</span>
            </div>
          </motion.li>
        )}
      </AnimatePresence>
      {degraded.length > 0 && (
        <li className="relative flex flex-wrap items-center gap-1 pt-1 pl-6 text-[12px]">
          <AlertTriangle aria-hidden className="absolute top-1.5 left-0.5 size-3.5 text-warn" />
          <span className="text-warn">{t("trace.degraded")}:</span>
          {degraded.map((item) => (
            <CodeBadge key={item} code={item} kind="degraded" tone="warn" className="font-normal" />
          ))}
        </li>
      )}
    </ol>
  );
}
