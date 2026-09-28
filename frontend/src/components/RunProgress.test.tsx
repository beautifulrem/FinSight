import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { applyEvent, type Turn } from "@/hooks/useChat";
import { I18nContext, makeTranslate } from "@/lib/i18n";
import type { StreamEvent } from "@/lib/types";

import { TurnView } from "./TurnView";
import { TooltipProvider } from "./ui/tooltip";

const EVENTS: StreamEvent[] = [
  { event: "node_start", data: { node: "guard_in", label: "" } },
  { event: "step", data: { node: "guard_in", label: "" } },
  { event: "node_start", data: { node: "agent_llm", label: "" } },
  { event: "step", data: { node: "agent_llm", label: "" } },
  { event: "tool_call", data: { tool: "get_fundamentals", arguments: '{"target": "贵州茅台"}' } },
  { event: "node_start", data: { node: "agent_tools", label: "" } },
];

function running(events: StreamEvent[], extra: Partial<Turn> = {}): Turn {
  const base: Turn = {
    kind: "turn",
    id: "turn-live",
    query: "贵州茅台的市盈率是多少",
    mode: "auto",
    via: "stream",
    status: "running",
    startedAt: performance.now(),
    steps: [],
  };
  return { ...events.reduce(applyEvent, base), ...extra };
}

function renderTurn(turn: Turn, { lang = "zh" as "zh" | "en", onStop = vi.fn() } = {}) {
  const ui = (item: Turn) => (
    <I18nContext.Provider value={{ lang, t: makeTranslate(lang) }}>
      <TooltipProvider>
        <TurnView
          turn={item}
          view={null}
          isLast
          themeKey="light"
          onCite={vi.fn()}
          onInspect={vi.fn()}
          onAsk={vi.fn()}
          onRetry={vi.fn()}
          onFeedback={vi.fn()}
          onStop={onStop}
        />
      </TooltipProvider>
    </I18nContext.Provider>
  );
  const result = render(ui(turn));
  return { ...result, onStop, update: (item: Turn) => result.rerender(ui(item)) };
}

describe("running turn progress", () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "performance"] }));
  afterEach(() => vi.useRealTimers());

  it("shows the current step, the tools with their target, the elapsed time and a skeleton", () => {
    const { update } = renderTurn(running(EVENTS.slice(0, 3)));
    // One polite status region carries the step for screen readers; the panel itself is not live.
    expect(screen.getByRole("status")).toHaveTextContent("理解问题：模型正在规划要查询的数据");
    expect(document.querySelector(".run-progress")?.closest('[aria-live="off"]')).not.toBeNull();
    expect(document.querySelector('.progress-steps [aria-current="step"]')).toHaveTextContent("理解问题");
    expect(document.querySelector(".answer-skeleton")).toHaveAttribute("aria-hidden", "true");
    expect(document.querySelector(".progress-elapsed")).toHaveTextContent("0 秒");

    act(() => vi.advanceTimersByTime(3000));
    expect(document.querySelector(".progress-elapsed")).toHaveTextContent("3 秒");

    update(running(EVENTS, { startedAt: 0 }));
    expect(screen.getByRole("status")).toHaveTextContent("查询数据：正在调用数据工具");
    const steps = [...document.querySelectorAll(".progress-steps li")].map((li) => li.getAttribute("data-state"));
    expect(steps).toEqual(["done", "active", "pending", "pending"]);
    const tool = document.querySelector(".progress-tool")!;
    expect(tool).toHaveAttribute("data-status", "running");
    expect(tool).toHaveTextContent("基本面");
    expect(tool).toHaveTextContent("get_fundamentals");
    expect(tool).toHaveTextContent("· 贵州茅台");
  });

  it("explains a long model wait and offers Stop", async () => {
    vi.useRealTimers();
    const user = userEvent.setup();
    const turn = running(EVENTS.slice(0, 3), { startedAt: performance.now() - 9000 });
    const { onStop } = renderTurn(turn, { lang: "en" });
    expect(document.querySelector(".progress-hint")).toHaveTextContent("Model reasoning usually takes 10–20 s");
    expect(document.querySelector(".progress-elapsed")).toHaveTextContent("9s");
    await user.click(screen.getByRole("button", { name: "Stop" }));
    expect(onStop).toHaveBeenCalledOnce();
  });

  it("folds into the run trace once the first answer token arrives", () => {
    const { update } = renderTurn(running(EVENTS));
    expect(document.querySelector(".run-progress")).not.toBeNull();
    expect(document.querySelector(".trace-toggle")).toBeNull();

    const writing: StreamEvent[] = [
      ...EVENTS,
      { event: "step", data: { node: "agent_tools", label: "" } },
      { event: "tool_result", data: { tool: "get_fundamentals", ok: true, latency_ms: 80, evidence_ids: [] } },
      { event: "node_start", data: { node: "agent_llm", label: "" } },
    ];
    update(running(writing, { draft: "贵州茅台市盈率约 24.6 倍" }));
    act(() => vi.advanceTimersByTime(500)); // exit animation
    expect(document.querySelector(".run-progress")).toBeNull();
    expect(document.querySelector(".streaming-answer")).toHaveAttribute("aria-busy", "true");
    const toggle = document.querySelector(".trace-toggle")!;
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveTextContent("正在撰写回答…");
    expect(screen.getByRole("status")).toHaveTextContent("撰写回答：正在撰写回答");
  });

  it("shows only the elapsed time for requests without step events", () => {
    renderTurn(running([], { via: "classic" }));
    expect(document.querySelector(".progress-steps")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent("分析中");
    expect(document.querySelector(".answer-skeleton")).not.toBeNull();
  });

  it("says how long a stopped turn ran", () => {
    renderTurn(running(EVENTS, { status: "stopped", startedAt: 0, finishedAt: 12_400 }));
    expect(document.querySelector(".stopped-note")).toHaveTextContent("已停止（用时 12 秒）");
  });
});
