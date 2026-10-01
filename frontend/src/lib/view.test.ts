import type { Turn } from "@/hooks/useChat";

import type { AgentResponse } from "./types";
import { answerView, distinctKeyPoints } from "./view";

describe("distinctKeyPoints", () => {
  it("hides key points that only repeat the answer", () => {
    const answer = "收盘价 1409.5 [price_1]。PE 24.6 [fundamental_1]。";
    expect(distinctKeyPoints(answer, ["收盘价 1409.5 [price_1]。", "PE 24.6 [fundamental_1]。"])).toEqual([]);
  });

  it("keeps key points that add information", () => {
    expect(distinctKeyPoints("走势偏弱。", ["RSI 10.9"])).toEqual(["RSI 10.9"]);
  });
});

describe("answerView asked metrics (round 13, I15)", () => {
  const turn = (nlu: Record<string, unknown>): Turn => ({
    kind: "turn",
    id: "t",
    query: "宁德时代下周五收盘会在多少",
    mode: "auto",
    via: "stream",
    status: "done",
    startedAt: 0,
    steps: [],
    agent: { status: "ok", session_id: "s", answer: "…", nlu_summary: nlu } as AgentResponse,
  });

  it("marks no tile as the asked metric on a forecast question", () => {
    expect(answerView(turn({ question_style: "forecast", asked_metrics: ["close"] }))?.askedMetrics).toEqual([]);
  });

  it("keeps the asked metrics of a factual question", () => {
    expect(answerView(turn({ question_style: "fact", asked_metrics: ["roe"] }))?.askedMetrics).toEqual(["roe"]);
  });
});
