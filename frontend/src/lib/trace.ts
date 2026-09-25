import type { LiveStep } from "@/hooks/useChat";

import type { AgentResponse, LlmLogEntry, ToolCall, ToolError } from "./types";

export interface TraceTool {
  id: string;
  tool: string;
  args: unknown;
  status: "running" | "ok" | "error";
  latencyMs?: number;
  evidenceIds: string[];
  error?: ToolError | null;
  cached?: boolean;
  attempts?: number;
  source?: string;
  reason?: string;
  startedAt?: number;
}

export interface TraceNode {
  id: string;
  node: string;
  durationMs?: number;
  startedAt?: number;
  tools: TraceTool[];
  llm?: LlmLogEntry;
}

export function liveTrace(steps: LiveStep[]): TraceNode[] {
  return steps.map((step) => ({
    id: step.id,
    node: step.node,
    tools: step.tools.map((tool) => ({
      id: tool.id,
      tool: tool.tool,
      args: tool.args,
      status: tool.status,
      latencyMs: tool.latencyMs,
      evidenceIds: tool.evidenceIds ?? [],
      error: tool.error,
    })),
  }));
}

function toTraceTool(call: ToolCall, i: number): TraceTool {
  return {
    id: `tool-${i}`,
    tool: call.tool,
    args: call.arguments,
    status: call.ok ? "ok" : "error",
    latencyMs: call.latency_ms,
    evidenceIds: call.evidence_ids ?? [],
    error: call.error,
    cached: call.cached,
    attempts: call.attempts,
    source: call.source,
    reason: call.reason,
    startedAt: call.started_at,
  };
}

/**
 * Rebuild the run as an ordered list of graph nodes from the final response: `spans` give node
 * order and timing; planner tools attach to `execute_plan`, LLM-chosen tools to the matching
 * `agent_tools` round, and LLM log entries to their node.
 */
export function finalTrace(response: AgentResponse): TraceNode[] {
  const spans = response.spans ?? [];
  const calls = (response.tool_calls ?? []).map(toTraceTool);
  const planner = calls.filter((call) => call.source !== "llm");
  const llmRounds = new Map<number, TraceTool[]>();
  (response.tool_calls ?? []).forEach((call, i) => {
    if (call.source !== "llm") return;
    const round = call.step ?? 0;
    llmRounds.set(round, [...(llmRounds.get(round) ?? []), calls[i]!]);
  });
  const rounds = [...llmRounds.entries()].sort(([a], [b]) => a - b).map(([, tools]) => tools);
  const llmLog = response.llm?.log ?? [];
  const logByNode = new Map<string, LlmLogEntry[]>();
  for (const entry of llmLog) {
    const node = entry.node ?? "agent_llm";
    logByNode.set(node, [...(logByNode.get(node) ?? []), entry]);
  }
  let plannerUsed = false;
  const nodes: TraceNode[] = spans.map((span, i) => {
    const node: TraceNode = { id: `span-${i}`, node: span.node, durationMs: span.duration_ms, startedAt: span.started_at, tools: [] };
    if (span.node === "execute_plan" && !plannerUsed) {
      node.tools = planner;
      plannerUsed = true;
    } else if (span.node === "agent_tools") {
      node.tools = rounds.shift() ?? [];
    }
    const log = logByNode.get(span.node);
    if (log?.length) node.llm = log.shift();
    return node;
  });
  const leftover = [...(plannerUsed ? [] : planner), ...rounds.flat()];
  if (leftover.length) nodes.push({ id: "tools-extra", node: "agent_tools", tools: leftover });
  return nodes;
}

export function traceStats(nodes: TraceNode[]): { steps: number; tools: number } {
  return { steps: nodes.length, tools: nodes.reduce((sum, node) => sum + node.tools.length, 0) };
}

/** Server-side run time from spans (first start to last end). */
export function serverDurationMs(response: AgentResponse): number | undefined {
  const spans = response.spans ?? [];
  if (!spans.length) return undefined;
  const start = Math.min(...spans.map((span) => span.started_at));
  const end = Math.max(...spans.map((span) => span.started_at + span.duration_ms / 1000));
  return (end - start) * 1000;
}

export function formatArgs(args: unknown): string {
  if (args == null) return "";
  if (typeof args === "string") {
    try {
      return JSON.stringify(JSON.parse(args));
    } catch {
      return args;
    }
  }
  return JSON.stringify(args);
}
