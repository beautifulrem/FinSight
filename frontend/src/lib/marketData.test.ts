import { formatKpi } from "./format";
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

  it("uses the unit fields of normalised agent payloads (B7: 成交额 was 1000x too small)", () => {
    const response: AgentResponse = {
      status: "ok",
      session_id: "s",
      evidence_sources: [
        {
          evidence_id: "price_600519.SH",
          source_type: "market_api",
          source_name: "tushare",
          payload: { name: "贵州茅台", close: 1409.5, amount: 3_793_827_534, amount_unit: "CNY" },
        },
        {
          evidence_id: "fundamental_600519.SH",
          source_type: "fundamental_sql",
          payload: { roe: 33, pe_ttm: 24.6, metric_units: { roe: "%", pe_ttm: "x" } },
        },
      ],
    };
    const kpis = marketDataFromAgent(response).kpis;
    const amount = kpis.find((kpi) => kpi.label === "kpi.amount");
    expect(amount?.format).toBe("money");
    expect(formatKpi("zh", amount!.value, amount!.format)).toBe("37.94 亿");
    const roe = kpis.find((kpi) => kpi.label === "kpi.roe");
    expect(roe && formatKpi("zh", roe.value, roe.format)).toBe("33%");
  });

  it("finds the Tushare convention on the evidence item or in the provenance of raw rows", () => {
    const raw = (extra: Record<string, unknown>, sourceName?: string): AgentResponse => ({
      status: "ok",
      session_id: "s",
      evidence_sources: [
        { evidence_id: "price_600519.SH", source_type: "market_api", source_name: sourceName, payload: { close: 1409.5, amount: 3_793_827.534, ...extra } },
      ],
    });
    for (const response of [raw({}, "tushare"), raw({ provenance: { original_source: "tushare.daily" } })]) {
      const amount = marketDataFromAgent(response).kpis.find((kpi) => kpi.label === "kpi.amount");
      expect(formatKpi("zh", amount!.value, amount!.format)).toBe("37.94 亿");
    }
  });
});
