import { ageInDays, evidenceTitle, formatCost, formatKpi, formatMs } from "./format";

describe("format helpers", () => {
  it("formats durations", () => {
    expect(formatMs(0.4)).toBe("<1 ms");
    expect(formatMs(42.3)).toBe("42 ms");
    expect(formatMs(2500)).toBe("2.50 s");
  });

  it("formats KPI values in the reader's units", () => {
    expect(formatKpi("zh", 174_120_000_000, "money")).toBe("1,741.2 亿");
    expect(formatKpi("en", 174_120_000_000, "money")).toBe("174.12B");
    expect(formatKpi("zh", 0.33, "fraction")).toBe("33%");
    expect(formatKpi("zh", -0.1778, "percent")).toBe("-0.18%");
    expect(formatKpi("zh", 3_793_827.534, "volume")).toBe("37.94 亿");
    expect(formatKpi("zh", 3_793_827_534, "money")).toBe("37.94 亿");
    expect(formatKpi("zh", 33, "percentLevel")).toBe("33%");
    expect(formatKpi("en", 0.8, "percentLevel")).toBe("0.8%");
    // fund prices quote in 0.001 CNY (round 9, E13)
    expect(formatKpi("zh", 1.021, "fundPrice")).toBe("1.021");
    expect(formatKpi("zh", 1.021, "price")).toBe("1.02");
  });

  it("formats cost only when priced", () => {
    expect(formatCost(null, null)).toBeNull();
    expect(formatCost(0.010058, "USD")).toBe("$0.0101");
  });

  it("computes evidence age", () => {
    expect(ageInDays("2026-09-20", new Date("2026-09-25T12:00:00"))).toBe(5);
    expect(ageInDays(null)).toBeUndefined();
  });
});

describe("evidenceTitle", () => {
  it("names structured evidence kinds in Chinese and leaves documents alone", async () => {
    const { evidenceTitle } = await import("./format");
    expect(evidenceTitle("zh", "五粮液 (000858.SZ) daily market data")).toBe("五粮液 (000858.SZ) 日行情");
    expect(evidenceTitle("zh", "贵州茅台 (600519.SH) fundamentals")).toBe("贵州茅台 (600519.SH) 基本面");
    expect(evidenceTitle("zh", "Document sentiment for 600519.SH")).toBe("600519.SH 文档情绪");
    expect(evidenceTitle("en", "五粮液 (000858.SZ) daily market data")).toBe("五粮液 (000858.SZ) daily market data");
    expect(evidenceTitle("zh", "茅台2025年年报发布")).toBe("茅台2025年年报发布");
    const names = new Map([["贵州茅台", "Kweichow Moutai"]]);
    expect(evidenceTitle("en", "贵州茅台 (600519.SH) fundamentals", names)).toBe("Kweichow Moutai (600519.SH) fundamentals");
  });
});

describe("evidenceTitle keeps symbols", () => {
  it("translates the name but not the symbol in an English title", () => {
    const names = new Map([
      ["贵州茅台", "Kweichow Moutai"],
      ["600519.SH", "Kweichow Moutai"],
    ]);
    expect(evidenceTitle("en", "贵州茅台 (600519.SH) daily market data", names)).toBe(
      "Kweichow Moutai (600519.SH) daily market data",
    );
    expect(evidenceTitle("en", "白酒 industry snapshot")).toBe("Baijiu (liquor) industry snapshot");
    const server = new Map([["白酒", "baijiu (liquor)"]]); // the server's name_en is lower case
    expect(evidenceTitle("en", "白酒 industry snapshot", server)).toBe("Baijiu (liquor) industry snapshot");
  });
});
