import type { Turn } from "@/hooks/useChat";

/**
 * Live progress of a streamed turn before (and while) its answer arrives, derived from the SSE events
 * the turn has received: `node_start` / `step` give the running graph node, `tool_call` / `tool_result`
 * the tools, and the first `answer_delta` the start of writing.
 *
 * The four phases follow the order the graph actually runs in (query_intelligence/agent/graph.py):
 * guard_in → (agent_llm ⇄ agent_tools | execute_plan) → drafting (agent_llm / compose) → verify → compliance.
 */
export type Phase = "understand" | "fetch" | "write" | "verify";

export const PHASES: readonly Phase[] = ["understand", "fetch", "write", "verify"];

/** What the run is doing right now, in words a reader understands. */
export type Activity =
  | "starting"
  | "routing"
  | "clarifying"
  | "planning"
  | "fetching"
  | "reading"
  | "composing"
  | "writing"
  | "verifying"
  | "revising"
  | "finalizing";

export interface ProgressTool {
  id: string;
  tool: string;
  target?: string;
  status: "running" | "ok" | "error";
  latencyMs?: number;
}

export interface Progress {
  phase: Phase;
  activity: Activity;
  /** Every tool called so far, in call order. */
  tools: ProgressTool[];
  /** True once an LLM node has run: the wait before the first token is model time. */
  usesLlm: boolean;
  /** True once answer text has started streaming. */
  streaming: boolean;
}

const TARGET_KEYS = ["target", "targets", "text", "topics", "query", "symbol", "name"] as const;
const MAX_TARGET = 32;

function parseArgs(args: unknown): unknown {
  if (typeof args !== "string") return args;
  try {
    return JSON.parse(args);
  } catch {
    return args;
  }
}

/** The security, topic or query a tool call is about: `get_fundamentals({"target": "贵州茅台"})` → "贵州茅台". */
export function toolTarget(args: unknown): string | undefined {
  const value = parseArgs(args);
  let found: string | undefined;
  if (typeof value === "string") found = value.trim();
  else if (value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    for (const key of TARGET_KEYS) {
      const item = record[key];
      if (typeof item === "string" && item.trim()) found = item.trim();
      else if (Array.isArray(item)) {
        const parts = item.filter((part): part is string => typeof part === "string" && Boolean(part.trim()));
        if (parts.length) found = parts.map((part) => part.trim()).join(", ");
      }
      if (found) break;
    }
  }
  if (!found) return undefined;
  return found.length > MAX_TARGET ? `${found.slice(0, MAX_TARGET - 1)}…` : found;
}

function phaseOf(activity: Activity): Phase {
  switch (activity) {
    case "starting":
    case "routing":
    case "clarifying":
    case "planning":
      return "understand";
    case "fetching":
      return "fetch";
    case "reading":
    case "composing":
    case "writing":
      return "write";
    default:
      return "verify";
  }
}

type ProgressInput = Pick<Turn, "steps" | "current" | "draft">;

/** Event sequence → what the progress panel shows. Pure, so the state machine is unit-tested. */
export function turnProgress(turn: ProgressInput): Progress {
  const tools: ProgressTool[] = turn.steps.flatMap((step) =>
    step.tools.map((tool) => ({
      id: tool.id,
      tool: tool.tool,
      target: toolTarget(tool.args),
      status: tool.status,
      latencyMs: tool.latencyMs,
    })),
  );
  const streaming = Boolean(turn.draft);
  const node = turn.current?.node;
  const usesLlm = node === "agent_llm" || turn.steps.some((step) => step.node === "agent_llm");
  let activity: Activity;
  switch (node) {
    case undefined:
      activity = "starting";
      break;
    case "guard_in":
      activity = "routing";
      break;
    case "clarify":
      activity = "clarifying";
      break;
    case "execute_plan":
    case "agent_tools":
      activity = "fetching";
      break;
    case "agent_llm":
      // The first LLM round decides which data to fetch; later rounds read the results and write.
      activity = streaming ? "writing" : tools.length ? "reading" : "planning";
      // A finished round that asked for tools hands over to agent_tools: show the fetch already.
      if (!turn.current?.running && tools.some((tool) => tool.status === "running")) activity = "fetching";
      break;
    case "compose":
      activity = streaming ? "writing" : "composing";
      break;
    case "verify":
      activity = "verifying";
      break;
    case "revise":
      activity = "revising";
      break;
    default:
      // refuse, compliance, finalize
      activity = "finalizing";
  }
  return { phase: phaseOf(activity), activity, tools, usesLlm, streaming };
}

/** Status of each phase in the step indicator. */
export function phaseState(current: Phase, phase: Phase): "done" | "active" | "pending" {
  const a = PHASES.indexOf(phase);
  const b = PHASES.indexOf(current);
  return a < b ? "done" : a === b ? "active" : "pending";
}
