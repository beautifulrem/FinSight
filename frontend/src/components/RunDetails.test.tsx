import { render, screen } from "@testing-library/react";

import type { Turn } from "@/hooks/useChat";
import { I18nContext, makeTranslate } from "@/lib/i18n";
import { answerView } from "@/lib/view";

import { RunDetails } from "./RunDetails";
import { TooltipProvider } from "./ui/tooltip";

const REASONS = [
  "simple:single_lookup",
  "ellipsis:target->贵州茅台",
  "ellipsis:aspect->市盈率+走势",
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

describe("RunDetails frame reasons (round 12, H14)", () => {
  const frame = ["frame:difference:roe:五粮液|贵州茅台", "frame:style_compare", "model_policy:composition_for_slow_model"];

  function withEntities(): Turn {
    const item = turn(frame);
    return {
      ...item,
      agent: {
        ...item.agent!,
        nlu_summary: {
          entities: [
            { name: "五粮液", name_en: "Wuliangye", symbol: "000858.SZ" },
            { name: "贵州茅台", name_en: "Kweichow Moutai", symbol: "600519.SH" },
          ],
        },
      },
    };
  }

  it.each([
    ["zh", ["计算ROE差值：五粮液 对比 贵州茅台", "按对比计算处理", "慢速推理模型：改用固定流程，由 LLM 撰写回答"]],
    [
      "en",
      [
        "Computed the ROE gap: Wuliangye vs Kweichow Moutai",
        "Treated as a computed comparison",
        "Slow reasoning model: used the workflow with an LLM-written answer",
      ],
    ],
  ] as const)("shows the computed comparison in %s, never the raw chip", (lang, labels) => {
    const item = withEntities();
    render(
      <I18nContext.Provider value={{ lang, t: makeTranslate(lang) }}>
        <TooltipProvider>
          <RunDetails view={answerView(item)!} turn={item} sessionId="s1" />
        </TooltipProvider>
      </I18nContext.Provider>,
    );
    for (const label of labels) expect(screen.getByText(label)).toBeInTheDocument();
    const details = document.querySelector(".run-details")!;
    for (const raw of ["Frame difference", "frame:", "|", "model_policy"]) expect(details).not.toHaveTextContent(raw);
  });
});

describe("RunDetails timings", () => {
  it("shows the client-measured time to first token and the total time", () => {
    const item: Turn = { ...turn([]), startedAt: 100, firstTokenAt: 9_900, finishedAt: 14_300 };
    render(
      <TooltipProvider>
        <RunDetails view={answerView(item)!} turn={item} sessionId="s1" />
      </TooltipProvider>,
    );
    expect(document.querySelector(".run-ttft")).toHaveTextContent("9.80 s");
    expect(document.querySelector(".run-wall")).toHaveTextContent("14.2 s");
    expect(screen.getAllByText("· 浏览器计时")).toHaveLength(2);
  });

  it("says when an agent answer was not streamed", () => {
    const item = turn([]);
    render(
      <TooltipProvider>
        <RunDetails view={answerView(item)!} turn={item} sessionId="s1" />
      </TooltipProvider>,
    );
    expect(document.querySelector(".run-ttft")).toHaveTextContent("未流式输出");
  });
});
