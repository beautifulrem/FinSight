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
  /** English name of the payload's company or industry (agent/names.py), for the English UI's tiles. */
  name_en?: string | null;
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
  /** `name_en`: the English name from the alias table (data/synonym_dict.json), for the English UI. */
  entities?: { name?: string | null; symbol?: string | null; name_en?: string | null }[];
  risk_flags?: string[];
  /** Metric keys the question asks about, in order ("roe", "pe", "net_margin"…): the KPI tiles lead with them. */
  asked_metrics?: string[];
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
  /** A hearsay question ("听说茅台市盈率只有15倍，是真的吗"): the claim inside it, checked against the data. */
  fact_check?: ClaimReport | null;
}

export interface StructuredItem {
  evidence_id?: string;
  source_type?: string;
  source_name?: string | null;
  as_of?: string | null;
  /** Agent evidence titles name the company: "五粮液 (000858.SZ) fundamentals". */
  title?: string | null;
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
    entities?: { canonical_name?: string; symbol?: string; name_en?: string | null }[];
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
  /** As on the agent path: the inline check of a hearsay question. */
  fact_check?: ClaimReport | null;
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
  /** A graph node is about to run (sent before its `step`, which reports it finished). */
  | { event: "node_start"; data: { node: string; label: string } }
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

/** `POST /agent/claim-check` (query_intelligence/agent/claim_check.py `ClaimReport`). */
export type ClaimStatus = "supported" | "contradicted" | "unverifiable";
export type ClaimVerdict = ClaimStatus | "partially_supported";

/** What the card's headline says: the verdict, or "partly checked" for a supported verdict with unchecked parts. */
export type ClaimHeadline = ClaimVerdict | "partly_checked";

/** How a claimed number relates to the value: "超过30%" is `gt`, "不是15倍" is `ne`, "20到30倍" is `range`. */
export type ClaimComparator = "eq" | "ne" | "gt" | "ge" | "lt" | "le" | "approx" | "range";

/** Why a check is unverifiable (machine-readable; `note` carries the English text). */
export type ClaimReason =
  | "no_target"
  | "no_metric"
  | "no_data"
  | "growth_unavailable"
  | "unit_mismatch"
  | "no_unit"
  | "forecast"
  | "period_mismatch"
  | "multi_day"
  | "no_reference";

export interface ClaimCheckItem {
  target?: string | null;
  metric?: string | null;
  /** The claimed number, signed by its move word ("跌超1%" is -1); null for a relation ("比五粮液高"). */
  claimed: number | null;
  /** Upper bound of a range claim ("20到30倍"). */
  claimed_high?: number | null;
  /** Unit written after the number in the claim: "亿", "%", "倍", "billion", ... */
  claimed_unit?: string | null;
  comparator?: ClaimComparator;
  negated?: boolean;
  /** The direction a move word states ("跌超1%" is down): a bound then applies to the size of the move. */
  direction?: "up" | "down" | null;
  /** A relational claim's other side ("茅台PE比五粮液高": 五粮液, or "白酒行业平均"), with its value. */
  reference?: string | null;
  reference_value?: number | null;
  reference_evidence_id?: string | null;
  /** A multiple claim ("市净率是五粮液的1.5倍"): the target's value divided by the reference's; `claimed` is the multiple. */
  ratio?: number | null;
  /**
   * A difference claim ("茅台ROE比五粮液高出3.6个百分点"): the target's value minus the reference's (a percentage of
   * the reference's value for `relative_difference`); `claimed` is the stated difference, signed by its direction.
   */
  difference?: number | null;
  /**
   * What the check compares (round 10): the target's own value; `stated_reference`, a value the claim states for the
   * compared side ("比白酒行业平均的30倍低": the average is 30x); a relation; a multiple; a stated difference.
   */
  kind?: "value" | "stated_reference" | "relation" | "ratio" | "difference" | "relative_difference";
  actual?: number | null;
  status: ClaimStatus;
  reason?: ClaimReason | null;
  evidence_id?: string | null;
  source?: string | null;
  as_of?: string | null;
  /** What `as_of` is: the trade date of a price, the valuation date of P/E and P/B, or the report period. */
  as_of_basis?: "trade_date" | "valuation_date" | "report_date" | "indicator_date" | null;
  note?: string;
}

export interface ClaimReport {
  claim: string;
  verdict: ClaimVerdict;
  checks: ClaimCheckItem[];
  targets?: { name?: string | null; symbol?: string | null; name_en?: string | null }[];
  /** English for the Chinese target / reference labels of the checks ("白酒行业平均" → "baijiu (liquor) industry average"). */
  labels_en?: Record<string, string>;
  evidence_sources?: {
    evidence_id: string;
    source_name?: string | null;
    as_of?: string | null;
    title?: string | null;
    provenance?: Provenance | null;
  }[];
  /** Clauses that name a target or metric but had nothing to check ("ROE很高"): shown as "not checked" rows. */
  unchecked?: ClaimUnchecked[];
  /**
   * How much of the claim the verdict covers: "partial" when a part is unchecked or unverifiable. A "supported"
   * verdict with partial coverage is shown as "partly checked", never as "the numbers agree" (round 9, E2).
   */
  coverage?: "full" | "partial" | "none";
  disclaimer: string;
}

export interface ClaimUnchecked {
  /** The clause as the checker read it ("十五倍" is written "15倍"). */
  text: string;
  reason?: "no_claim";
  note?: string;
}

export type FeedbackRating = "up" | "down";

export interface FeedbackRequest {
  trace_id: string;
  session_id: string | null;
  rating: FeedbackRating;
  comment: string | null;
}
