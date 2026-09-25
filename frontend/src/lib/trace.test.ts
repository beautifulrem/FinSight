import { finalTrace, serverDurationMs, traceStats } from "./trace";
import type { AgentResponse, ToolCall } from "./types";

const call = (tool: string, source: string, step: number): ToolCall => ({
  tool,
  arguments: { target: "600519.SH" },
  ok: true,
  latency_ms: 10,
  evidence_ids: [],
  source,
  step,
});

describe("finalTrace", () => {
  it("attaches LLM tool rounds to agent_tools spans in order", () => {
    const response: AgentResponse = {
      status: "ok",
      session_id: "s",
      spans: [
        { node: "guard_in", started_at: 100, duration_ms: 5 },
        { node: "agent_llm", started_at: 100.01, duration_ms: 1000 },
        { node: "agent_tools", started_at: 101.01, duration_ms: 20 },
        { node: "agent_llm", started_at: 101.03, duration_ms: 900 },
        { node: "agent_tools", started_at: 101.93, duration_ms: 20 },
        { node: "verify", started_at: 101.95, duration_ms: 1 },
      ],
      tool_calls: [call("get_price_history", "llm", 1), call("get_fundamentals", "llm", 1), call("search_news", "llm", 2)],
      llm: { calls: 2, steps: 2, usage: { prompt_tokens: 1, completion_tokens: 1, prompt_cache_hit_tokens: 0, reasoning_tokens: 0, total_tokens: 2 }, log: [{ node: "agent_llm", prompt_tokens: 10 }, { node: "agent_llm", prompt_tokens: 20 }] },
    };
    const nodes = finalTrace(response);
    expect(nodes.map((node) => node.node)).toEqual(["guard_in", "agent_llm", "agent_tools", "agent_llm", "agent_tools", "verify"]);
    expect(nodes[2]?.tools.map((tool) => tool.tool)).toEqual(["get_price_history", "get_fundamentals"]);
    expect(nodes[4]?.tools.map((tool) => tool.tool)).toEqual(["search_news"]);
    expect(nodes[3]?.llm?.prompt_tokens).toBe(20);
    expect(traceStats(nodes)).toEqual({ steps: 6, tools: 3 });
    expect(serverDurationMs(response)).toBeCloseTo(1951, 0);
  });

  it("puts planner tools under execute_plan", () => {
    const response: AgentResponse = {
      status: "ok",
      session_id: "s",
      spans: [{ node: "execute_plan", started_at: 1, duration_ms: 3 }],
      tool_calls: [call("get_price_history", "planner", 0)],
    };
    expect(finalTrace(response)[0]?.tools).toHaveLength(1);
  });
});
