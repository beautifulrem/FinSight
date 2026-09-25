import { marketDataFromAgent, marketDataFromClassic } from "./marketData";
import type { AgentResponse, ClassicResponse } from "./types";

describe("marketDataFromClassic", () => {
  const response: ClassicResponse = {
    answer: "",
    retrieval_result: {
      structured_data: [
        {
          evidence_id: "price_600519.SH",
          source_type: "market_api",
          payload: {
            symbol: "600519.SH",
            canonical_name: "贵州茅台",
            source_name: "tushare",
            trade_date: "20260424",
            close: 1420,
            pct_change_1d: 0.74,
            amount: 3_793_827.5,
            // Providers return latest-first.
            history: [
              { trade_date: "20260424", close: 1420 },
              { trade_date: "20260423", close: 1409.5 },
              { trade_date: "20260422", close: 1400 },
            ],
          },
        },
        { evidence_id: "fundamental_600519.SH", source_type: "fundamental_sql", payload: { symbol: "600519.SH", pe_ttm: 24.6, roe: 0.33 } },
      ],
      analysis_summary: { market_signal: { trend_signal: "bullish", rsi_14: 55.2 } },
    },
  };

  it("builds an oldest-first price series and KPI tiles", () => {
    const data = marketDataFromClassic(response);
    expect(data.series[0]?.points.map((point) => point.time)).toEqual(["2026-04-22", "2026-04-23", "2026-04-24"]);
    expect(data.series[0]?.name).toBe("贵州茅台");
    const labels = data.kpis.map((kpi) => kpi.label);
    expect(labels).toEqual(expect.arrayContaining(["kpi.close", "kpi.change", "kpi.amount", "kpi.pe", "kpi.roe", "kpi.rsi"]));
    expect(data.kpis.find((kpi) => kpi.label === "kpi.change")?.tone).toBe("up");
    expect(data.kpis.find((kpi) => kpi.label === "kpi.amount")?.format).toBe("volume");
    expect(data.trend).toBe("bullish");
  });
});

describe("marketDataFromAgent", () => {
  it("returns nothing when evidence carries no payload (current server)", () => {
    const response: AgentResponse = {
      status: "ok",
      session_id: "s",
      evidence_sources: [{ evidence_id: "price_600519.SH", source_type: "market_api" }],
    };
    expect(marketDataFromAgent(response)).toEqual({ series: [], kpis: [] });
  });

  it("uses recent_closes when a payload is present", () => {
    const response: AgentResponse = {
      status: "ok",
      session_id: "s",
      evidence_sources: [
        {
          evidence_id: "price_600519.SH",
          source_type: "market_api",
          payload: { name: "贵州茅台", close: 1409.5, recent_closes: [{ date: "2026-04-21", close: 1400 }, { date: "2026-04-22", close: 1409.5 }] },
        },
      ],
    };
    const data = marketDataFromAgent(response);
    expect(data.series).toHaveLength(1);
    expect(data.kpis[0]?.label).toBe("kpi.close");
  });
});
