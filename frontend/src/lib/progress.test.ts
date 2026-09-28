import { applyEvent, type Turn } from "@/hooks/useChat";

import { phaseState, toolTarget, turnProgress } from "./progress";
import type { StreamEvent } from "./types";

function start(): Turn {
  return { kind: "turn", id: "t", query: "贵州茅台的市盈率是多少", mode: "auto", via: "stream", status: "running", startedAt: 0, steps: [] };
}

const nodeStart = (node: string): StreamEvent => ({ event: "node_start", data: { node, label: node } });
const step = (node: string): StreamEvent => ({ event: "step", data: { node, label: node } });
const call = (tool: string, args: unknown): StreamEvent => ({ event: "tool_call", data: { tool, arguments: args } });
const result = (tool: string, ok = true): StreamEvent => ({
  event: "tool_result",
  data: { tool, ok, latency_ms: 120, evidence_ids: [], error: ok ? null : { code: "upstream_error", message: "x" } },
});

/** Feed events one by one and record what the panel shows after each. */
function play(events: (StreamEvent | { draft: string })[], turn = start()) {
  const shown: string[] = [];
  for (const event of events) {
    turn = "draft" in event ? { ...turn, draft: (turn.draft ?? "") + event.draft } : applyEvent(turn, event);
    const progress = turnProgress(turn);
    shown.push(`${progress.phase}/${progress.activity}`);
  }
  return { shown, turn };
}

describe("turnProgress state machine", () => {
  it("starts in 'understand' before any event", () => {
    expect(turnProgress(start())).toMatchObject({ phase: "understand", activity: "starting", tools: [], streaming: false });
  });

  it("walks the agent path: plan → fetch → read → write → verify → finalize", () => {
    const { shown, turn } = play([
      nodeStart("guard_in"),
      step("guard_in"),
      nodeStart("agent_llm"),
      step("agent_llm"),
      call("resolve_entity", '{"text": "贵州茅台"}'),
      call("get_fundamentals", { target: "贵州茅台" }),
      nodeStart("agent_tools"),
      step("agent_tools"),
      result("resolve_entity"),
      result("get_fundamentals", false),
      nodeStart("agent_llm"),
      { draft: "贵州茅台市盈率" },
      step("agent_llm"),
      nodeStart("verify"),
      step("verify"),
      nodeStart("compliance"),
    ]);
    expect(shown).toEqual([
      "understand/routing",
      "understand/routing",
      "understand/planning",
      "understand/planning",
      // A finished LLM round that asked for tools already shows the fetch.
      "fetch/fetching",
      "fetch/fetching",
      "fetch/fetching",
      "fetch/fetching",
      "fetch/fetching",
      "fetch/fetching",
      "write/reading",
      "write/writing",
      "write/writing",
      "verify/verifying",
      "verify/verifying",
      "verify/finalizing",
    ]);
    const progress = turnProgress(turn);
    expect(progress.usesLlm).toBe(true);
    expect(progress.streaming).toBe(true);
    expect(progress.tools.map((tool) => [tool.tool, tool.target, tool.status])).toEqual([
      ["resolve_entity", "贵州茅台", "ok"],
      ["get_fundamentals", "贵州茅台", "error"],
    ]);
  });

  it("walks the deterministic workflow path without an LLM", () => {
    const { shown, turn } = play([
      nodeStart("guard_in"),
      step("guard_in"),
      nodeStart("execute_plan"),
      step("execute_plan"),
      call("get_price_history", { target: "600519.SH" }),
      result("get_price_history"),
      nodeStart("compose"),
      step("compose"),
      nodeStart("verify"),
      nodeStart("revise"),
    ]);
    expect(shown).toEqual([
      "understand/routing",
      "understand/routing",
      "fetch/fetching",
      "fetch/fetching",
      "fetch/fetching",
      "fetch/fetching",
      "write/composing",
      "write/composing",
      "verify/verifying",
      "verify/revising",
    ]);
    expect(turnProgress(turn).usesLlm).toBe(false);
  });

  it("goes back to fetching when the model asks for more tools", () => {
    const { shown } = play([
      nodeStart("agent_llm"),
      step("agent_llm"),
      call("get_price_history", { target: "贵州茅台" }),
      nodeStart("agent_tools"),
      step("agent_tools"),
      result("get_price_history"),
      nodeStart("agent_llm"),
      step("agent_llm"),
      call("search_news", { query: "白酒", targets: [] }),
      nodeStart("agent_tools"),
    ]);
    expect(shown.slice(-5)).toEqual(["fetch/fetching", "write/reading", "write/reading", "fetch/fetching", "fetch/fetching"]);
  });
});

describe("toolTarget", () => {
  it("finds the security, topic or query of a call", () => {
    expect(toolTarget({ target: "贵州茅台" })).toBe("贵州茅台");
    expect(toolTarget('{"target": "600519.SH"}')).toBe("600519.SH");
    expect(toolTarget({ query: "白酒", targets: ["贵州茅台", "五粮液"] })).toBe("贵州茅台, 五粮液");
    expect(toolTarget({ topics: ["cpi", "pmi"] })).toBe("cpi, pmi");
    expect(toolTarget({ text: "  茅台  " })).toBe("茅台");
    expect(toolTarget({ targets: [], query: "新能源" })).toBe("新能源");
    expect(toolTarget({})).toBeUndefined();
    expect(toolTarget(null)).toBeUndefined();
    expect(toolTarget({ query: "x".repeat(60) })).toHaveLength(32);
  });
});

describe("phaseState", () => {
  it("marks earlier phases done and later ones pending", () => {
    expect(phaseState("write", "understand")).toBe("done");
    expect(phaseState("write", "write")).toBe("active");
    expect(phaseState("write", "verify")).toBe("pending");
  });
});
