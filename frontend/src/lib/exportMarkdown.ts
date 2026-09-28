import { evidenceIndex, splitCitations } from "./citations";
import { fallbackReasonText, humanizeCode, limitationText } from "./codes";
import { formatDate } from "./format";
import { evidenceFreshness } from "./freshness";
import { sourceNameLabel, sourceTypeLabel, type Lang } from "./i18n";
import type { AnswerView } from "./view";

const T = {
  zh: {
    question: "问题",
    answer: "回答",
    keyPoints: "要点",
    limitations: "局限",
    evidence: "证据",
    disclaimer: "风险提示",
    generated: "生成时间",
    route: "路由",
    model: "模型",
    trace: "Trace ID",
    verified: "核验",
    passed: "通过",
    failed: "未通过",
    asOf: "截至",
    cited: "已引用",
    notCited: "未引用",
    live: "实时",
    fallback: "备用源",
    cached: "缓存",
    snapshot: "离线快照",
    stale: "可能过时",
    noLink: "结构化数据，无网页链接",
    from: "来自",
    reason: "降级原因",
    none: "（无）",
  },
  en: {
    question: "Question",
    answer: "Answer",
    keyPoints: "Key points",
    limitations: "Limitations",
    evidence: "Evidence",
    disclaimer: "Risk disclaimer",
    generated: "Generated",
    route: "Route",
    model: "Model",
    trace: "Trace ID",
    verified: "Verification",
    passed: "passed",
    failed: "failed",
    asOf: "as of",
    cited: "cited",
    notCited: "not cited",
    live: "live",
    fallback: "fallback source",
    cached: "cached",
    snapshot: "offline snapshot",
    stale: "may be stale",
    noLink: "structured data, no web link",
    from: "from",
    reason: "fallback reason",
    none: "(none)",
  },
} as const;

/** Citation markers `[price_600519.SH]` become `[E1]`, matching the numbered evidence list. */
function withRefs(text: string, index: Map<string, number>): string {
  let out = "";
  for (const segment of splitCitations(text, index)) {
    if (segment.type === "text") out += segment.text;
    // Ids that are not in the run were already flagged by the verifier; they are dropped here.
    else if (segment.type === "cite") out += `${/[A-Za-z0-9%)]$/.test(out) ? " " : ""}[E${segment.index}]`;
  }
  return out;
}

function escapeInline(text: string): string {
  return text.replace(/([\\`*_[\]<>|])/g, "\\$1");
}

export interface ExportMeta {
  query: string;
  lang: Lang;
  now?: Date;
  disclaimerFallback: string;
}

/** One answer with its evidence ledger and risk disclaimer, as a self-contained Markdown document. */
export function answerToMarkdown(view: AnswerView, meta: ExportMeta): string {
  const { lang } = meta;
  const t = T[lang];
  const now = meta.now ?? new Date();
  const index = evidenceIndex(view.evidence.map((source) => source.evidence_id));
  const agent = view.agent;
  const lines: string[] = [];

  lines.push(`# ${meta.query.replace(/\s+/g, " ").trim()}`, "");
  const facts = [`${t.generated}: ${now.toISOString().replace(/\.\d+Z$/, "Z")}`];
  if (view.route) facts.push(`${t.route}: ${humanizeCode(lang, view.route, "route")}`);
  if (agent?.llm?.model) facts.push(`${t.model}: ${agent.llm.model}`);
  if (view.verification?.passed != null) facts.push(`${t.verified}: ${view.verification.passed ? t.passed : t.failed}`);
  if (agent?.trace_id) facts.push(`${t.trace}: \`${agent.trace_id}\``);
  lines.push(`> FinSight · ${facts.join(" · ")}`, "");

  lines.push(`## ${t.answer}`, "", withRefs(view.answer, index).trim(), "");

  if (view.keyPoints.length) {
    lines.push(`## ${t.keyPoints}`, "", ...view.keyPoints.map((point) => `- ${withRefs(point, index)}`), "");
  }
  if (view.limitations.length) {
    lines.push(
      `## ${t.limitations}`,
      "",
      ...view.limitations.map((item) => {
        const { text, code } = limitationText(lang, item);
        return `- ${withRefs(text, index)}${code ? ` (\`${code}\`)` : ""}`;
      }),
      "",
    );
  }

  lines.push(`## ${t.evidence}`, "");
  if (!view.evidence.length) lines.push(t.none);
  view.evidence.forEach((source, i) => {
    const info = evidenceFreshness(source, now);
    const status: string[] = [view.cited.has(source.evidence_id) ? t.cited : t.notCited];
    if (info.mode !== "unknown") status.push(t[info.mode]);
    if (info.stale) status.push(t.stale);
    const title = escapeInline(source.title || source.evidence_id);
    const parts = [`**E${i + 1}** ${title}`, `\`${source.evidence_id}\``];
    parts.push(sourceTypeLabel(lang, source.source_type));
    const origin = sourceNameLabel(lang, info.provenance, source.source_name);
    if (origin) parts.push(`${t.from} ${escapeInline(origin)}`);
    if (info.asOf) parts.push(`${t.asOf} ${formatDate(lang, info.asOf)}`);
    parts.push(status.join(", "));
    lines.push(`${i + 1}. ${parts.join(" · ")}`);
    const url = source.source_url && /^https?:\/\//i.test(source.source_url) ? source.source_url : null;
    lines.push(`   - ${url ? `<${url}>` : t.noLink}`);
    const reason = info.provenance?.fallback_reason;
    if (reason) lines.push(`   - ${t.reason}: ${fallbackReasonText(lang, reason)}`);
  });
  lines.push("");

  lines.push(`## ${t.disclaimer}`, "", `> ${view.disclaimer || meta.disclaimerFallback}`, "");
  return lines.join("\n");
}

/** `finsight-2026-09-26-贵州茅台最近走势.md`: local date plus a short, filesystem-safe slug of the question. */
export function exportFilename(query: string, now = new Date()): string {
  const slug = query
    .replace(/[\\/:*?"<>|\s]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
  const pad = (n: number) => String(n).padStart(2, "0");
  const day = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  return `finsight-${day}${slug ? `-${slug}` : ""}.md`;
}

export function downloadText(filename: string, text: string, type = "text/markdown;charset=utf-8"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
