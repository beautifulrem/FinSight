import { render, screen } from "@testing-library/react";

import type { Turn } from "@/hooks/useChat";
import { answerView } from "@/lib/view";

import { RunDetails } from "./RunDetails";
import { TooltipProvider } from "./ui/tooltip";

const REASONS = [
  "simple:single_lookup",
  "ellipsis:target->贵州茅台",
  "ellipsis:aspect->市盈率/走势",
  "coreference:它->贵州茅台",
  "dropped_fuzzy_concept:有色金属",
  "input_guard:instruction_like_text_removed",
];

function turn(reasons: string[]): Turn {
  return {
    kind: "turn",
    id: "t1",
    query: "ROE呢",
    mode: "auto",
    via: "stream",
    status: "done",
    startedAt: 0,
    finishedAt: 10,
    steps: [],
    agent: { status: "ok", session_id: "s1", trace_id: "a".repeat(32), route: "workflow", route_reasons: reasons, answer: "…" },
  };
}

describe("RunDetails route reasons", () => {
  it("shows follow-up rewrites in words and keeps the raw code for the tooltip", () => {
    const item = turn(REASONS);
    render(
      <TooltipProvider>
        <RunDetails view={answerView(item)!} turn={item} sessionId="s1" />
      </TooltipProvider>,
    );
    for (const label of [
      "简单单点查询",
      "沿用上一轮的标的 贵州茅台",
      "沿用上一轮的问题：市盈率、走势",
      "将“它”理解为 贵州茅台",
      "忽略了模糊匹配到的概念“有色金属”",
      "已移除问题中疑似指令的文本",
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText("沿用上一轮的标的 贵州茅台")).toHaveAttribute("data-code", "ellipsis:target->贵州茅台");
    const details = document.querySelector(".run-details")!;
    for (const code of ["ellipsis:", "coreference:", "->", "dropped_fuzzy_concept", "input_guard"]) {
      expect(details).not.toHaveTextContent(code);
    }
  });
});
