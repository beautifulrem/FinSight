import { act, render, renderHook, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { type Turn, useChat } from "@/hooks/useChat";
import type { AgentResponse } from "@/lib/types";
import { answerView } from "@/lib/view";

import { TurnView } from "./TurnView";
import { TooltipProvider } from "./ui/tooltip";

/**
 * Round 12: three-turn gap sessions through the streaming client. The server's session comparison frame computes the
 * gap (or ratio) on the third turn and cites both operands; the UI must show that sentence with both citation chips,
 * never a raw evidence id, and keep the earlier turns' answers. Answers are the offline backend's own text for these
 * sessions (captured from the template path).
 */

type Session = Record<string, AgentResponse>;

function fundamental(symbol: string) {
  return { evidence_id: `fundamental_${symbol}`, kind: "structured", source_type: "fundamental_sql", as_of: "2025-12-31" };
}

function price(symbol: string) {
  return { evidence_id: `price_${symbol}`, kind: "structured", source_type: "market_api", as_of: "2026-04-22" };
}

function answer(query: string, text: string, ids: string[], reasons: string[], language: "zh" | "en"): AgentResponse {
  const sources = ids.map((id) => (id.startsWith("price_") ? price(id.slice(6)) : fundamental(id.slice(12))));
  return {
    status: "ok",
    session_id: "s-gap",
    trace_id: `trace-${query}`,
    query,
    language,
    route: "workflow",
    route_reasons: reasons,
    answer: text,
    evidence_used: ids,
    evidence_sources: sources,
    limitations: [],
    verification: { passed: true } as AgentResponse["verification"],
  };
}

const MT = "600519.SH";
const PA = "601318.SH";

const ZH_ROE: Session = {
  贵州茅台的净资产收益率: answer(
    "贵州茅台的净资产收益率",
    "根据本次检索到的证据：贵州茅台 基本面（报告期 2025-12-31，2025年年报）：PE(TTM) 24.6 倍，PB 8.1 倍，ROE 33%，营业收入 1688.38 亿元，净利润 823.2 亿元 [fundamental_600519.SH]。",
    [`fundamental_${MT}`],
    ["simple:single_lookup"],
    "zh",
  ),
  中国平安呢: answer(
    "中国平安呢",
    "根据本次检索到的证据：中国平安 基本面（报告期 2025-12-31，2025年年报）：PE(TTM) 8.7 倍，PB 1.1 倍，ROE 15.2%，营业收入 12180 亿元，净利润 1210 亿元 [fundamental_601318.SH]。",
    [`fundamental_${PA}`],
    ["simple:single_lookup", "ellipsis:aspect->净资产收益率"],
    "zh",
  ),
  高了几个点: answer(
    "高了几个点",
    "根据本次检索到的证据：贵州茅台ROE 33%，中国平安 15.2%，两者相差 17.8 个百分点（贵州茅台更高） [fundamental_600519.SH][fundamental_601318.SH]。",
    [`fundamental_${MT}`, `fundamental_${PA}`],
    ["frame:difference:roe:贵州茅台|中国平安", "frame:style_compare"],
    "zh",
  ),
};

const EN_ROE: Session = {
  "What's Ping An's ROE?": answer(
    "What's Ping An's ROE?",
    "Based on the evidence retrieved for this question: Ping An Insurance fundamentals (period 2025-12-31, FY2025): PE(TTM) 8.7x, PB 1.1x, ROE 15.2% [fundamental_601318.SH].",
    [`fundamental_${PA}`],
    ["simple:single_lookup"],
    "en",
  ),
  "and Moutai's?": answer(
    "and Moutai's?",
    "Based on the evidence retrieved for this question: Kweichow Moutai fundamentals (period 2025-12-31, FY2025): PE(TTM) 24.6x, PB 8.1x, ROE 33% [fundamental_600519.SH].",
    [`fundamental_${MT}`],
    ["simple:single_lookup", "ellipsis:aspect->ROE"],
    "en",
  ),
  "by how many points is the latter higher?": answer(
    "by how many points is the latter higher?",
    "Based on the evidence retrieved for this question: Kweichow Moutai ROE 33%, Ping An Insurance 15.2%: a difference of 17.8 percentage points (Kweichow Moutai is higher) [fundamental_600519.SH][fundamental_601318.SH].",
    [`fundamental_${MT}`, `fundamental_${PA}`],
    ["frame:difference:roe:贵州茅台|中国平安", "frame:style_compare"],
    "en",
  ),
};

const TURNOVER_RATIO: Session = {
  沪深300ETF今天成交了多少钱: answer(
    "沪深300ETF今天成交了多少钱",
    "根据本次检索到的证据：沪深300ETF（510300.SH）最新可用收盘价为 4.811 元（2026-04-22） [price_510300.SH]。沪深300ETF（2026-04-22）：成交额 48.52 亿元 [price_510300.SH]。",
    ["price_510300.SH"],
    ["simple:single_lookup"],
    "zh",
  ),
  证券ETF的呢: answer(
    "证券ETF的呢",
    "根据本次检索到的证据：证券ETF（512880.SH）最新可用收盘价为 1.021 元（2026-04-22），当日涨跌幅 0.59% [price_512880.SH]。证券ETF（2026-04-22）：成交额 4.41 亿元 [price_512880.SH]。",
    ["price_512880.SH"],
    ["simple:single_lookup", "ellipsis:aspect->成交额"],
    "zh",
  ),
  前者是后者的几倍: answer(
    "前者是后者的几倍",
    "根据本次检索到的证据：沪深300ETF成交额 48.52 亿元，证券ETF 4.41 亿元，前者约为后者的 11 倍 [price_510300.SH][price_512880.SH]。",
    ["price_510300.SH", "price_512880.SH"],
    ["frame:ratio:amount:沪深300ETF|证券ETF", "frame:style_compare"],
    "zh",
  ),
};

function sse(events: [string, unknown][]): string {
  return events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("");
}

/** `fetch` answering `/agent/chat/stream` from the session table, as the server streams it. */
function mockStream(session: Session) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
    const { query } = JSON.parse(String(init?.body)) as { query: string };
    const response = session[query];
    if (!response) return new Response(JSON.stringify({ detail: "unexpected query" }), { status: 500 });
    const body = sse([
      ["session", { session_id: "s-gap" }],
      ["step", { node: "guard_in" }],
      ["answer", response],
      ["done", {}],
    ]);
    return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
  });
}

async function runSession(session: Session): Promise<Turn[]> {
  const fetchSpy = mockStream(session);
  const { result } = renderHook(() => useChat({ sessionId: "s-gap", mode: "auto", apiKey: "" }));
  for (const query of Object.keys(session)) {
    await act(async () => {
      await result.current.send(query);
    });
  }
  // one request per turn, all in the same session
  expect(fetchSpy).toHaveBeenCalledTimes(3);
  for (const call of fetchSpy.mock.calls) expect(JSON.parse(String(call[1]?.body)).session_id).toBe("s-gap");
  const turns = result.current.items.filter((item): item is Turn => item.kind === "turn");
  expect(turns.map((turn) => turn.status)).toEqual(["done", "done", "done"]);
  return turns;
}

function renderTurns(turns: Turn[], onCite = vi.fn()) {
  render(
    <TooltipProvider>
      {turns.map((turn, index) => (
        <TurnView
          key={turn.id}
          turn={turn}
          view={answerView(turn)}
          isLast={index === turns.length - 1}
          themeKey="light"
          onCite={onCite}
          onInspect={vi.fn()}
          onAsk={vi.fn()}
          onRetry={vi.fn()}
          onFeedback={vi.fn().mockResolvedValue("ok")}
        />
      ))}
    </TooltipProvider>,
  );
  return { cards: screen.getAllByRole("article"), onCite };
}

async function expectBothOperandsCited(card: HTMLElement, onCite: ReturnType<typeof vi.fn>, turn: Turn, ids: string[]) {
  const user = userEvent.setup();
  expect(card).not.toHaveTextContent(/\[(?:fundamental|price)_/);
  for (const [position, id] of ids.entries()) {
    await user.click(within(card).getAllByRole("button", { name: new RegExp(`E${position + 1}\\b`) })[0]!);
    expect(onCite).toHaveBeenLastCalledWith(turn.id, id);
  }
}

describe("three-turn gap sessions (round 12)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows the frame's computed gap in percentage points with both operands cited (Chinese)", async () => {
    const turns = await runSession(ZH_ROE);
    expect(turns[2]!.agent?.route_reasons).toContain("frame:difference:roe:贵州茅台|中国平安");
    const { cards, onCite } = renderTurns(turns);
    expect(cards).toHaveLength(3);
    expect(cards[0]).toHaveTextContent("ROE 33%");
    expect(cards[1]).toHaveTextContent("ROE 15.2%");
    expect(cards[2]).toHaveTextContent("两者相差 17.8 个百分点（贵州茅台更高）");
    await expectBothOperandsCited(cards[2]!, onCite, turns[2]!, [`fundamental_${MT}`, `fundamental_${PA}`]);
  });

  it("shows the English gap after an ellipsis chain with 'the latter'", async () => {
    const turns = await runSession(EN_ROE);
    const { cards, onCite } = renderTurns(turns);
    expect(cards[2]).toHaveTextContent("a difference of 17.8 percentage points (Kweichow Moutai is higher)");
    expect(cards[2]).not.toHaveTextContent(/which one is better/i);
    await expectBothOperandsCited(cards[2]!, onCite, turns[2]!, [`fundamental_${MT}`, `fundamental_${PA}`]);
  });

  it("states the turnover on each turn and the ratio on the third", async () => {
    const turns = await runSession(TURNOVER_RATIO);
    const { cards, onCite } = renderTurns(turns);
    expect(cards[0]).toHaveTextContent("成交额 48.52 亿元");
    expect(cards[1]).toHaveTextContent("成交额 4.41 亿元");
    expect(cards[2]).toHaveTextContent("前者约为后者的 11 倍");
    await expectBothOperandsCited(cards[2]!, onCite, turns[2]!, ["price_510300.SH", "price_512880.SH"]);
  });
});
