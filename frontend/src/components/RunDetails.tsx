import { Check, Copy } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { Turn } from "@/hooks/useChat";
import { formatCost, formatInt, formatMs } from "@/lib/format";
import { humanizeCode } from "@/lib/codes";
import { useI18n } from "@/lib/i18n";
import type { AnswerView } from "@/lib/view";

import { CodeBadge } from "./CodeLabel";

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[7.5rem_1fr] gap-2 border-b border-line/70 py-2 text-[13px] last:border-0">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 text-ink">{children}</dd>
    </div>
  );
}

function CopyId({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="group inline-flex max-w-full items-center gap-1 font-mono text-[12px] break-all text-ink hover:text-cobalt"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        } catch {
          /* ignore */
        }
      }}
    >
      <span className="trace-id">{value}</span>
      {copied ? <Check className="size-3 shrink-0" aria-hidden /> : <Copy className="size-3 shrink-0 opacity-50 group-hover:opacity-100" aria-hidden />}
    </button>
  );
}

export function RunDetails({ view, turn, sessionId }: { view: AnswerView; turn: Turn; sessionId: string }) {
  const { lang, t } = useI18n();
  const agent = view.agent;
  const llm = agent?.llm;
  const usage = llm?.usage;
  const cost = formatCost(llm?.cost, llm?.currency);
  const nlu = agent?.nlu_summary;
  const tools = agent?.tool_calls ?? [];
  const failed = tools.filter((call) => !call.ok).length;
  const classic = turn.classic;
  return (
    <dl className="run-details">
      {agent?.trace_id && (
        <Row label={t("run.traceId")}>
          <CopyId value={agent.trace_id} />
        </Row>
      )}
      <Row label={t("run.route")}>
        <span className="flex flex-wrap items-center gap-1">
          {view.route ? <CodeBadge code={view.route} kind="route" tone="cobalt" /> : "–"}
          {(agent?.route_reasons ?? []).map((reason) => (
            <CodeBadge key={reason} code={reason} kind="reason" className="font-normal" />
          ))}
        </span>
      </Row>
      {view.answerSource && (
        <Row label={t("run.answerSource")}>
          <CodeBadge code={view.answerSource} kind="answerSource" className="font-normal" />
        </Row>
      )}
      {agent && (
        <Row label={t("run.model")}>
          {llm?.model ? <span className="font-mono text-[12px]">{llm.model}</span> : <span className="text-muted">{t("run.noModel")}</span>}
        </Row>
      )}
      {classic?.llm && (
        <Row label={t("run.llmStatus")}>
          <span className="font-mono text-[12px]">{classic.llm.model}</span>
          {classic.llm.status && (
            <span className="ml-1 text-[12px]">· {humanizeCode(lang, classic.llm.status, "answerSource")}</span>
          )}
          {classic.llm.error && <span className="block text-[12px] text-muted">{classic.llm.error}</span>}
        </Row>
      )}
      {llm && (
        <Row label={t("run.llmCalls")}>
          {llm.calls}
          {llm.steps ? <span className="ml-1 text-muted">· {t("run.steps", { n: llm.steps })}</span> : null}
        </Row>
      )}
      {usage && (
        <Row label={t("run.tokens")}>
          <span className="font-medium tabular-nums">{formatInt(lang, usage.total_tokens)}</span>
          <span className="block text-[12px] text-muted tabular-nums">
            {t("run.prompt")} {formatInt(lang, usage.prompt_tokens)} · {t("run.completion")} {formatInt(lang, usage.completion_tokens)}
            {usage.prompt_cache_hit_tokens ? ` · ${t("run.cacheHit")} ${formatInt(lang, usage.prompt_cache_hit_tokens)}` : ""}
            {usage.reasoning_tokens ? ` · ${t("run.reasoning")} ${formatInt(lang, usage.reasoning_tokens)}` : ""}
          </span>
        </Row>
      )}
      {agent && (
        <Row label={t("run.cost")}>
          {cost ? (
            <span className="tabular-nums">
              {cost}
              {llm?.cost_source && (
                <span className="ml-1 text-[12px] text-muted" data-code={llm.cost_source}>
                  ({humanizeCode(lang, llm.cost_source)})
                </span>
              )}
            </span>
          ) : (
            <span className="text-muted">{llm?.calls ? t("run.costUnset") : "0"}</span>
          )}
        </Row>
      )}
      {view.serverMs !== undefined && <Row label={t("run.latency")}>{formatMs(view.serverMs)}</Row>}
      {view.wallMs !== undefined && <Row label={t("run.wall")}>{formatMs(view.wallMs)}</Row>}
      {agent && (
        <Row label={t("run.tools")}>
          {tools.length}
          {failed ? <span className="ml-1 text-up">({failed} ✕)</span> : null}
        </Row>
      )}
      {nlu && (
        <>
          <Row label={t("run.entities")}>
            {(nlu.entities ?? []).length
              ? (nlu.entities ?? []).map((entity) => `${entity.name ?? ""}${entity.symbol ? ` (${entity.symbol})` : ""}`).join("、")
              : "–"}
          </Row>
          <Row label={t("run.style")}>
            {nlu.question_style ? humanizeCode(lang, nlu.question_style, "style") : "–"} ·{" "}
            {nlu.product_type ? humanizeCode(lang, nlu.product_type, "product") : "–"}
          </Row>
          {(nlu.risk_flags ?? []).length > 0 && (
            <Row label={t("run.riskFlags")}>
              <span className="flex flex-wrap gap-1">
                {(nlu.risk_flags ?? []).map((flag) => (
                  <CodeBadge key={flag} code={flag} kind="risk" tone="warn" className="font-normal" />
                ))}
              </span>
            </Row>
          )}
        </>
      )}
      {classic?.nlu_result && (
        <>
          <Row label={t("run.entities")}>
            {(classic.nlu_result.entities ?? []).map((entity) => `${entity.canonical_name ?? ""}${entity.symbol ? ` (${entity.symbol})` : ""}`).join("、") || "–"}
          </Row>
          <Row label={t("run.intents")}>
            {(classic.nlu_result.intent_labels ?? []).map((intent) => humanizeCode(lang, intent.label, "intent")).join(lang === "zh" ? "、" : ", ") ||
              "–"}
          </Row>
          <Row label={t("run.sources")}>
            <span className="font-mono text-[12px]">{(classic.retrieval_result?.executed_sources ?? []).join(", ") || "–"}</span>
          </Row>
        </>
      )}
      <Row label={t("run.session")}>
        <span className="font-mono text-[12px] break-all">{sessionId}</span>
        {agent?.turn_index != null && <span className="ml-1 text-[12px] text-muted">{t("run.turn", { n: agent.turn_index + 1 })}</span>}
      </Row>
    </dl>
  );
}
