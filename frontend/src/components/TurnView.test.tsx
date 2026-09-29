import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { Turn } from "@/hooks/useChat";
import { I18nContext, makeTranslate } from "@/lib/i18n";
import type { ClaimReport } from "@/lib/types";
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

function renderTurn(turn: Turn, onFeedback = vi.fn().mockResolvedValue("ok"), onCheckClaim?: (claim: string) => void) {
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
        onCheckClaim={onCheckClaim}
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

describe("claim hint", () => {
  it("suggests the fact-check view for hearsay questions and hands over the claim", async () => {
    const user = userEvent.setup();
    const onCheckClaim = vi.fn();
    renderTurn(doneTurn({ query: "听说茅台市盈率只有15倍，是真的吗？" }), undefined, onCheckClaim);
    expect(screen.getByText("这像一条待核实的说法")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "核查这句话" }));
    expect(onCheckClaim).toHaveBeenCalledWith("茅台市盈率只有15倍");
  });

  it("stays hidden for ordinary questions", () => {
    renderTurn(doneTurn(), undefined, vi.fn());
    expect(screen.queryByRole("button", { name: "核查这句话" })).not.toBeInTheDocument();
  });
});

describe("inline fact check, region names and English names (round 4)", () => {
  const factCheck: ClaimReport = {
    claim: "五粮液昨天跌了超过1%",
    verdict: "contradicted",
    checks: [
      {
        target: "五粮液",
        metric: "pct_change_1d",
        claimed: -1,
        comparator: "gt",
        direction: "down",
        actual: -0.5337,
        status: "contradicted",
        evidence_id: "price_000858.SZ",
        as_of: "2026-04-22",
      },
    ],
    targets: [{ name: "五粮液", symbol: "000858.SZ", name_en: "Wuliangye" }],
    disclaimer: "不构成投资建议。",
  };

  function withData(overrides: Partial<Turn> = {}): Turn {
    const turn = doneTurn(overrides);
    return {
      ...turn,
      agent: {
        ...turn.agent!,
        route: "workflow",
        nlu_summary: { entities: [{ name: "贵州茅台", symbol: "600519.SH", name_en: "Kweichow Moutai" }] },
        evidence_sources: [
          {
            evidence_id: "price_600519.SH",
            kind: "structured",
            source_type: "market_api",
            as_of: "2026-09-24",
            payload: { symbol: "600519.SH", name: "贵州茅台", close: 1413, pct_change_1d: 0.42, provenance: { mode: "live" } },
          },
        ],
        ...(overrides.agent ?? {}),
      },
    };
  }

  function renderTurns(turns: Turn[], lang: "zh" | "en" = "zh", onCheckClaim = vi.fn()) {
    render(
      <I18nContext.Provider value={{ lang, t: makeTranslate(lang) }}>
        <TooltipProvider>
          {turns.map((turn, index) => (
            <TurnView
              key={turn.id}
              turn={turn}
              view={answerView(turn)}
              isLast={index === turns.length - 1}
              number={index + 1}
              themeKey="light"
              onCite={vi.fn()}
              onInspect={vi.fn()}
              onAsk={vi.fn()}
              onRetry={vi.fn()}
              onFeedback={vi.fn().mockResolvedValue("ok")}
              onCheckClaim={onCheckClaim}
            />
          ))}
        </TooltipProvider>
      </I18nContext.Provider>,
    );
    return onCheckClaim;
  }

  it("shows the server's check of a hearsay question inside the answer, with the move as a fall", async () => {
    const user = userEvent.setup();
    const turn = withData({ query: "听说五粮液昨天跌了超过1%，是真的吗", agent: { status: "ok", session_id: "s1", fact_check: factCheck } });
    const onCheckClaim = renderTurns([turn]);

    const inline = screen.getByRole("region", { name: "核查这句说法（第 1 轮）" });
    expect(inline.querySelector(".claim-verdict")).toHaveAttribute("data-verdict", "contradicted");
    expect(inline.querySelector(".claim-claimed")).toHaveTextContent("跌幅 > 1%");
    expect(inline.querySelector(".claim-claimed")).toHaveAttribute("data-direction", "down");
    expect(inline.querySelector(".claim-actual")).toHaveTextContent("-0.53%");
    // the inline check replaces the hint; its button opens the full view with the claim
    expect(screen.queryByRole("button", { name: "核查这句话" })).not.toBeInTheDocument();
    await user.click(within(inline).getByRole("button", { name: "在「核查」中查看" }));
    expect(onCheckClaim).toHaveBeenCalledWith("五粮液昨天跌了超过1%");
  });

  it("names each turn's regions uniquely (C17)", () => {
    renderTurns([withData({ id: "a" }), withData({ id: "b" })]);

    expect(screen.getByRole("region", { name: "数据（第 1 轮）" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "数据（第 2 轮）" })).toBeInTheDocument();
    const names = screen.getAllByRole("region").map((region) => region.getAttribute("aria-label"));
    expect(new Set(names).size).toBe(names.length);
  });

  it("shows English company names in the English UI", () => {
    renderTurns([withData({ agent: { status: "ok", session_id: "s1", fact_check: factCheck } })], "en");

    expect(document.querySelector(".kpi-tile")).toHaveTextContent("Kweichow Moutai ·");
    expect(document.querySelector(".claim-target")).toHaveTextContent("Wuliangye");
    expect(document.querySelector(".claim-target-name")).toHaveTextContent("Wuliangye");
    expect(document.querySelector(".claim-claimed")).toHaveTextContent("Fall > 1%");
  });
});
