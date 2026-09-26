import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { Turn } from "@/hooks/useChat";
import { answerView } from "@/lib/view";

import { TurnView } from "./TurnView";
import { TooltipProvider } from "./ui/tooltip";

function doneTurn(overrides: Partial<Turn> = {}): Turn {
  return {
    kind: "turn",
    id: "turn-1",
    query: "贵州茅台和白酒行业",
    mode: "auto",
    via: "stream",
    status: "done",
    startedAt: 0,
    finishedAt: 1200,
    steps: [],
    draft: "贵州茅台收于 1413 元 [price_600519.SH]。",
    agent: {
      status: "ok",
      session_id: "s1",
      trace_id: "trace-1",
      route: "refuse",
      answer: "条件性判断：贵州茅台收于 1413 元 [price_600519.SH]。",
      limitations: ["out_of_scope_query", "查询内容不属于金融范畴"],
      evidence_used: ["price_600519.SH"],
      evidence_sources: [
        { evidence_id: "price_600519.SH", kind: "structured", source_type: "market_api", as_of: "2026-09-24", payload: { provenance: { mode: "live" } } },
        {
          evidence_id: "industry_白酒",
          kind: "structured",
          source_type: "industry_sql",
          as_of: "2026-04-21",
          payload: { provenance: { mode: "snapshot", is_live: false, freshness: "stale" } },
        },
      ],
    },
    ...overrides,
  };
}

function renderTurn(turn: Turn, onFeedback = vi.fn().mockResolvedValue("ok")) {
  const view = answerView(turn);
  render(
    <TooltipProvider>
      <TurnView
        turn={turn}
        view={view}
        isLast
        themeKey="light"
        onCite={vi.fn()}
        onInspect={vi.fn()}
        onAsk={vi.fn()}
        onRetry={vi.fn()}
        onFeedback={onFeedback}
      />
    </TooltipProvider>,
  );
  return onFeedback;
}

describe("TurnView answer card", () => {
  beforeEach(() => window.localStorage.clear());

  it("flags an answer edited after verification, humanises codes and shows the freshness banner", () => {
    renderTurn(doneTurn());
    const card = screen.getByRole("article");
    expect(card).toHaveAttribute("data-edited", "true");
    expect(screen.getByText("已按核验结果修订")).toBeInTheDocument();
    const limitation = within(card).getByText("问题不属于金融范畴");
    expect(limitation).toHaveAttribute("data-code", "out_of_scope_query");
    expect(card).not.toHaveTextContent("out_of_scope_query");
    expect(screen.getByRole("region", { name: "部分数据不是实时的，或可能已过时" })).toHaveTextContent("1 条离线快照");
  });

  it("does not flag an answer that matches its draft", () => {
    const turn = doneTurn();
    renderTurn({ ...turn, draft: turn.agent!.answer! });
    expect(screen.getByRole("article")).not.toHaveAttribute("data-edited");
  });

  it("renders the streamed draft with citations hidden while running", () => {
    renderTurn(doneTurn({ status: "running", agent: undefined, draft: "茅台 [price_600519.SH] 收于 14" }));
    const streaming = document.querySelector(".streaming-answer");
    expect(streaming).toHaveTextContent("茅台 收于 14");
    expect(streaming).not.toHaveTextContent("price_600519");
  });

  it("sends thumbs-down feedback with a comment and remembers it per trace", async () => {
    const user = userEvent.setup();
    const onFeedback = renderTurn(doneTurn());
    await user.click(screen.getByRole("button", { name: "没帮助" }));
    expect(onFeedback).toHaveBeenCalledWith({ trace_id: "trace-1", session_id: "s1", rating: "down", comment: null });
    expect(screen.getByRole("button", { name: "没帮助" })).toHaveAttribute("aria-pressed", "true");
    await user.type(screen.getByLabelText("补充说明（可选）"), "数字不对");
    await user.click(screen.getByRole("button", { name: "提交" }));
    expect(onFeedback).toHaveBeenLastCalledWith({ trace_id: "trace-1", session_id: "s1", rating: "down", comment: "数字不对" });
    expect(await screen.findByText("感谢反馈")).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem("finsight.feedback")!)["trace-1"]).toMatchObject({ rating: "down", comment: "数字不对", state: "sent" });
  });

  it("keeps the rating locally when the server has no feedback endpoint", async () => {
    const user = userEvent.setup();
    renderTurn(doneTurn(), vi.fn().mockResolvedValue("unsupported"));
    await user.click(screen.getByRole("button", { name: "有帮助" }));
    expect(await screen.findByText("已保存在本浏览器（服务端暂不接收反馈）")).toBeInTheDocument();
  });
});
