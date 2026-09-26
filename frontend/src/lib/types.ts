// Types mirror `schemas/agent_chat_response.schema.json` and `query_intelligence/contracts.py`.
// Fields the UI does not rely on are optional so older/newer servers keep working.

export type AgentMode = "auto" | "agent" | "workflow";
export type UiMode = AgentMode | "classic";
export type AgentRoute = "refuse" | "clarify" | "workflow" | "agent";

export interface ToolError {
  code: string;
  message: string;
  retryable?: boolean;
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  error?: ToolError | null;
  latency_ms: number;
  started_at?: number;
  attempts?: number;
  cached?: boolean;
  evidence_ids: string[];
  source?: string;
  reason?: string;
  step?: number | null;
  instruction_like_text_removed?: boolean;
}

export interface EvidenceSource {
  evidence_id: string;
  kind?: string;
  source_type?: string | null;
  title?: string | null;
  source_name?: string | null;
  source_url?: string | null;
  as_of?: string | null;
  produced_by?: string | null;
  /** Structured evidence carries its payload (numbers, price series, `provenance`). */
  payload?: Record<string, unknown> | null;
  /** Some servers may lift provenance to the top level; the UI accepts either place. */
  provenance?: Provenance | null;
}

/** `payload.provenance` on live and snapshot records (query_intelligence/integrations/sources/provenance.py). */
export interface Provenance {
  source?: string | null;
  source_label?: string | null;
  endpoint?: string | null;
  is_live?: boolean;
  /** live | live_fallback | last_known_good | snapshot */
  mode?: string | null;
  fetched_at?: string | null;
  as_of?: string | null;
  /** fresh | stale | unknown */
  freshness?: string | null;
  fallback_reason?: string | null;
  attempts?: string[];
  cache_hit?: boolean;
  note?: string | null;
  original_source?: string | null;
}

export interface Verification {
  passed?: boolean | null;
  cited_ids?: string[];
  invalid_citations?: string[];
  unsupported_numbers?: number[];
  misattributed_numbers?: number[];
  checked_numbers?: number;
  missing_citations?: boolean;
  [key: string]: unknown;
}

export interface LlmUsage {
  prompt_tokens: number;
  completion_tokens: number;
  prompt_cache_hit_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
}

export interface LlmLogEntry {
  node?: string;
  step?: number | null;
  model?: string | null;
  started_at?: number;
  latency_ms?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  tool_calls?: string[];
  finish_reason?: string | null;
}

export interface LlmInfo {
  model?: string | null;
  calls: number;
  steps: number;
  usage: LlmUsage;
  cost?: number | null;
  currency?: string | null;
  cost_source?: string | null;
  log?: LlmLogEntry[];
}

export interface Span {
  node: string;
  started_at: number;
  duration_ms: number;
  [key: string]: unknown;
}

export interface NextQuestion {
  question: string;
  score?: number;
  reason?: string;
}

export interface SentimentSummary {
  targets?: string[];
  overall_label?: string;
  mean_score?: number;
  label_counts?: Record<string, number>;
  backend?: string;
  evidence_id?: string;
}

export interface Clarification {
  type?: "clarification";
  question: string;
  missing_slots?: string[];
  original_query?: string | null;
  [key: string]: unknown;
}

export interface NluSummary {
  question_style?: string | null;
  product_type?: string | null;
  entities?: { name?: string | null; symbol?: string | null }[];
  risk_flags?: string[];
}

export interface AgentResponse {
  status: "ok" | "needs_clarification";
  session_id: string;
  clarification?: Clarification | null;
  trace_id?: string | null;
  run_id?: string | null;
  query?: string | null;
  language?: "zh" | "en" | null;
  route?: AgentRoute | null;
  route_reasons?: string[];
  answer?: string | null;
  key_points?: string[];
  evidence_used?: string[];
  limitations?: string[];
  risk_disclaimer?: string | null;
  evidence_sources?: EvidenceSource[];
  tool_calls?: ToolCall[];
  verification?: Verification | null;
  compliance_notes?: string[];
  degraded?: string[];
  answer_source?: string | null;
  llm?: LlmInfo | null;
  nlu_summary?: NluSummary | null;
  spans?: Span[];
  turn_index?: number | null;
  sentiment?: SentimentSummary | null;
  next_questions?: NextQuestion[];
}

export interface StructuredItem {
  evidence_id?: string;
  source_type?: string;
  source_name?: string | null;
  as_of?: string | null;
  payload?: Record<string, unknown> | null;
}

/** `POST /chat` without a mode: the original NLU → retrieval → one LLM call pipeline. */
export interface ClassicResponse {
  answer: string;
  key_points?: string[];
  risk_disclaimer?: string;
  evidence_used?: string[];
  evidence_sources?: EvidenceSource[];
  llm?: { provider?: string; model?: string; status?: string; error?: string | null };
  nlu_result?: {
    query_id?: string;
    question_style?: string;
    product_type?: { label?: string };
    intent_labels?: { label: string; score?: number }[];
    entities?: { canonical_name?: string; symbol?: string }[];
    source_plan?: string[];
    risk_flags?: string[];
  };
  retrieval_result?: {
    executed_sources?: string[];
    structured_data?: StructuredItem[];
    documents?: unknown[];
    warnings?: string[];
    analysis_summary?: Record<string, unknown> | null;
  };
}

export interface SessionTurn {
  query: string;
  route?: string;
  answer?: string;
  entities?: { name?: string | null; symbol?: string | null }[];
  evidence_used?: string[];
}

export interface SessionInfo {
  session_id: string;
  turns: SessionTurn[];
  pending_clarification: Clarification | null;
}

export type StreamEvent =
  | { event: "session"; data: { session_id: string } }
  | { event: "step"; data: { node: string; label: string } }
  | { event: "tool_call"; data: { tool: string; arguments: unknown } }
  | {
      event: "tool_result";
      data: { tool: string; ok: boolean; latency_ms: number; evidence_ids: string[]; error?: ToolError | null };
    }
  | { event: "clarification"; data: Clarification & { session_id: string } }
  /** Streamed answer text while the LLM writes; the final `answer` event replaces it. */
  | { event: "answer_delta"; data: { text: string } }
  | { event: "answer"; data: AgentResponse }
  | { event: "error"; data: { message: string } }
  | { event: "done"; data: { session_id: string } };

export type FeedbackRating = "up" | "down";

export interface FeedbackRequest {
  trace_id: string;
  session_id: string | null;
  rating: FeedbackRating;
  comment: string | null;
}
