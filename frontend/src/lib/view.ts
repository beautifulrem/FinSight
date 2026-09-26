import type { Turn } from "@/hooks/useChat";

import { hasMarketData, marketDataFromAgent, marketDataFromClassic, type MarketData } from "./marketData";
import { finalTrace, liveTrace, serverDurationMs, type TraceNode } from "./trace";
import type {
  AgentResponse,
  ClassicResponse,
  EvidenceSource,
  NextQuestion,
  Provenance,
  SentimentSummary,
  StructuredItem,
  Verification,
} from "./types";

/** One shape for agent and classic answers so the card and inspector render both. */
export interface AnswerView {
  kind: "agent" | "classic";
  answer: string;
  keyPoints: string[];
  limitations: string[];
  disclaimer?: string | null;
  evidence: EvidenceSource[];
  cited: Set<string>;
  route?: string | null;
  answerSource?: string | null;
  verification?: Verification | null;
  sentiment?: SentimentSummary | null;
  next: NextQuestion[];
  data?: MarketData;
  trace: TraceNode[];
  agent?: AgentResponse;
  serverMs?: number;
  wallMs?: number;
  llmStatus?: string;
}

function normalize(text: string): string {
  return text.replace(/\[[^\]]*\]/g, "").replace(/[\s。．.，,；;：:]/g, "");
}

/** Template answers are the key points joined together; hide key points that repeat the answer. */
export function distinctKeyPoints(answer: string, points: string[]): string[] {
  const body = normalize(answer);
  const distinct = points.filter((point) => !body.includes(normalize(point)));
  return distinct.length === 0 ? [] : points;
}

/** Classic responses list documents in `evidence_sources`; structured rows live in retrieval_result. */
function classicEvidence(response: ClassicResponse): EvidenceSource[] {
  const structured = new Map(
    (response.retrieval_result?.structured_data ?? []).filter((item) => item.evidence_id).map((item) => [item.evidence_id!, item]),
  );
  const describe = (item: StructuredItem): Partial<EvidenceSource> => {
    const payload = item.payload ?? {};
    const subject = payload.canonical_name ?? payload.name ?? payload.industry_name ?? payload.indicator_code ?? payload.symbol;
    const date = payload.trade_date ?? payload.report_date ?? payload.metric_date;
    return {
      kind: "structured",
      source_type: item.source_type ?? null,
      title: subject ? String(subject) : item.evidence_id,
      source_name: item.source_name ?? (typeof payload.source_name === "string" ? payload.source_name : null),
      as_of: item.as_of ?? (date ? String(date) : null),
      // Live / snapshot provenance drives the freshness badges and banner, as in agent mode.
      provenance: payload.provenance && typeof payload.provenance === "object" ? (payload.provenance as Provenance) : null,
    };
  };
  // Structured rows arrive with the evidence id as title and no date; fill them in from the payload.
  const sources: EvidenceSource[] = (response.evidence_sources ?? []).map((source) => {
    const item = structured.get(source.evidence_id);
    if (!item) return source;
    const info = describe(item);
    return {
      ...source,
      kind: source.kind ?? "structured",
      title: !source.title || source.title === source.evidence_id ? info.title : source.title,
      as_of: source.as_of ?? info.as_of,
      provenance: source.provenance ?? info.provenance,
    };
  });
  const seen = new Set(sources.map((source) => source.evidence_id));
  for (const [id, item] of structured) {
    if (seen.has(id)) continue;
    sources.push({ evidence_id: id, ...describe(item) });
  }
  const used = new Set(response.evidence_used ?? []);
  // Cited sources first, like the agent response.
  return [...sources.filter((s) => used.has(s.evidence_id)), ...sources.filter((s) => !used.has(s.evidence_id))];
}

export function answerView(turn: Turn): AnswerView | null {
  const wallMs = turn.finishedAt !== undefined ? turn.finishedAt - turn.startedAt : undefined;
  if (turn.agent) {
    const response = turn.agent;
    const data = marketDataFromAgent(response);
    const answer = response.answer ?? "";
    return {
      kind: "agent",
      answer,
      keyPoints: distinctKeyPoints(answer, response.key_points ?? []),
      limitations: response.limitations ?? [],
      disclaimer: response.risk_disclaimer,
      evidence: response.evidence_sources ?? [],
      cited: new Set(response.evidence_used ?? []),
      route: response.route,
      answerSource: response.answer_source,
      verification: response.verification,
      sentiment: response.sentiment,
      next: response.next_questions ?? [],
      data: hasMarketData(data) ? data : undefined,
      trace: response.spans?.length || response.tool_calls?.length ? finalTrace(response) : liveTrace(turn.steps),
      agent: response,
      serverMs: serverDurationMs(response),
      wallMs,
    };
  }
  if (turn.classic) {
    const response = turn.classic;
    const data = marketDataFromClassic(response);
    const answer = response.answer ?? "";
    return {
      kind: "classic",
      answer,
      keyPoints: distinctKeyPoints(answer, response.key_points ?? []),
      limitations: response.retrieval_result?.warnings ?? [],
      disclaimer: response.risk_disclaimer,
      evidence: classicEvidence(response),
      cited: new Set(response.evidence_used ?? []),
      route: "classic",
      answerSource: response.llm?.status === "ok" ? "llm" : response.llm?.status,
      next: [],
      data: hasMarketData(data) ? data : undefined,
      trace: [],
      wallMs,
      llmStatus: response.llm?.status,
    };
  }
  return null;
}
