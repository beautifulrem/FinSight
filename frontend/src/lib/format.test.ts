import { ageInDays, formatCost, formatKpi, formatMs } from "./format";

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
