import { formatKpi } from "./format";
import { marketDataFromAgent, marketDataFromClassic, selectKpis } from "./marketData";
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

  it("formats fund prices with three decimals (round 9, E13: 1.021 was shown as 1.02)", () => {
    const response: AgentResponse = {
      status: "ok",
      session_id: "s",
      evidence_sources: [
        { evidence_id: "price_159915.SZ", source_type: "market_api", payload: { name: "创业板ETF", symbol: "159915.SZ", close: 1.021 } },
        { evidence_id: "price_512880.SH", source_type: "market_api", payload: { product_type: "etf", close: 0.955, high: 0.961 } },
        { evidence_id: "price_600519.SH", source_type: "market_api", payload: { product_type: "stock", close: 1409.5 } },
      ],
    };
    const kpis = marketDataFromAgent(response).kpis.filter((kpi) => kpi.label === "kpi.close" || kpi.label === "kpi.high");
    expect(kpis.map((kpi) => [kpi.format, formatKpi("zh", kpi.value, kpi.format)])).toEqual([
      ["fundPrice", "1.021"],
      ["fundPrice", "0.955"],
      ["fundPrice", "0.961"],
      ["price", "1,409.5"],
    ]);
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

describe("selectKpis (C18: compare answers)", () => {
  const company = (symbol: string, name: string, pe: number): AgentResponse["evidence_sources"] => [
    {
      evidence_id: `price_${symbol}`,
      source_type: "market_api",
      payload: { symbol, name, close: 100, pct_change_1d: 1, high: 101, low: 99, amount: 5e9, amount_unit: "CNY" },
    },
    // fundamentals payloads may carry only the metrics: the tiles are named from the evidence id
    {
      evidence_id: `fundamental_${symbol}`,
      source_type: "fundamental_sql",
      // a symbol but no name, as the offline fundamentals serve it
      payload: symbol.startsWith("6") ? { symbol, pe_ttm: pe, pb: 5, roe: 30, revenue: 1e11 } : { pe_ttm: pe, pb: 5, roe: 30, revenue: 1e11 },
    },
  ];

  it("gives every compared company the same tiles", () => {
    const response = {
      status: "ok",
      session_id: "s",
      evidence_sources: [
        ...company("600519.SH", "贵州茅台", 24.6)!,
        ...company("000858.SZ", "五粮液", 20.9)!,
        { evidence_id: "industry_白酒", source_type: "industry_sql", payload: { industry_name: "白酒", pe: 27.3 } },
      ],
    } as AgentResponse;
    const tiles = selectKpis(marketDataFromAgent(response).kpis, 8);

    expect(tiles.map((kpi) => `${kpi.subject}:${kpi.label}`)).toEqual([
      "贵州茅台:kpi.close",
      "贵州茅台:kpi.change",
      "贵州茅台:kpi.pe",
      "贵州茅台:kpi.pb",
      "五粮液:kpi.close",
      "五粮液:kpi.change",
      "五粮液:kpi.pe",
      "五粮液:kpi.pb",
    ]);
  });

  it("keeps the order for one company", () => {
    const response = { status: "ok", session_id: "s", evidence_sources: company("600519.SH", "贵州茅台", 24.6) } as AgentResponse;
    const kpis = marketDataFromAgent(response).kpis;

    expect(selectKpis(kpis, 8)).toEqual(kpis.slice(0, 8));
    expect(kpis.every((kpi) => kpi.subject === "贵州茅台")).toBe(true);
  });

  it("leads with the metric the question asked about, for every company (round 11, G11)", () => {
    const response = {
      status: "ok",
      session_id: "s",
      evidence_sources: [...company("600519.SH", "贵州茅台", 24.6)!, ...company("000858.SZ", "五粮液", 20.9)!],
    } as AgentResponse;
    const tiles = selectKpis(marketDataFromAgent(response).kpis, 8, ["roe"]);

    expect(tiles.filter((kpi) => kpi.featured).map((kpi) => `${kpi.subject}:${kpi.label}`)).toEqual([
      "贵州茅台:kpi.roe",
      "五粮液:kpi.roe",
    ]);
    expect(tiles[0]?.label).toBe("kpi.roe");
    expect(tiles[4]?.label).toBe("kpi.roe");
  });

  it("derives a net-margin tile only to show it when asked", () => {
    const response = {
      status: "ok",
      session_id: "s",
      evidence_sources: [
        { evidence_id: "fundamental_000858.SZ", source_type: "fundamental_sql", payload: { symbol: "000858.SZ", name: "五粮液", pe_ttm: 20.9, revenue: 108.5e9, net_profit: 37.8e9 } },
      ],
    } as AgentResponse;
    const kpis = marketDataFromAgent(response).kpis;

    expect(selectKpis(kpis, 8, ["net_margin"])[0]).toMatchObject({ label: "kpi.netMargin", value: 34.84, featured: true });
    expect(selectKpis(kpis, 8)[0]?.label).toBe("kpi.pe");
  });
});
