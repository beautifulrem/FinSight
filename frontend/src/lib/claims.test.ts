import { checkEvidence, claimedAmount, formatClaimValue, noteText, statusCounts } from "./claims";
import { makeTranslate } from "./i18n";
import type { ClaimReport } from "./types";

const zh = makeTranslate("zh");
const en = makeTranslate("en");

const report: ClaimReport = {
  claim: "茅台市盈率只有15倍，营收1700亿",
  verdict: "contradicted",
  checks: [
    { target: "贵州茅台", metric: "pe_ttm", claimed: 15, actual: 24.6, status: "contradicted", evidence_id: "fundamental_600519.SH", source: null, as_of: "2025-12-31" },
    { target: "贵州茅台", metric: null, claimed: 3, status: "unverifiable", note: "metric not recognised" },
  ],
  targets: [{ name: "贵州茅台", symbol: "600519.SH" }],
  evidence_sources: [{ evidence_id: "fundamental_600519.SH", source_name: null, as_of: "2025-12-31", title: "贵州茅台 (600519.SH) fundamentals" }],
  disclaimer: "不构成投资建议。",
};

describe("formatClaimValue", () => {
  it("writes each metric in its unit", () => {
    expect(formatClaimValue("zh", zh, "pe_ttm", 24.6, "actual")).toBe("24.6 倍");
    expect(formatClaimValue("en", en, "pe_ttm", 15, "claimed")).toBe("15×");
    expect(formatClaimValue("zh", zh, "close", 1409.5, "actual")).toBe("1,409.5 元");
    expect(formatClaimValue("zh", zh, "pct_change_1d", -5, "claimed")).toBe("-5%");
    expect(formatClaimValue("en", en, "pct_change_1d", -0.1778, "actual")).toBe("-0.18%");
    expect(formatClaimValue("zh", zh, "roe", 50, "claimed")).toBe("50%");
    // the claim-check server normalises ROE to percent (tools/units.py), so the actual value is not rescaled
    expect(formatClaimValue("zh", zh, "roe", 33, "actual")).toBe("33%");
    expect(formatClaimValue("zh", zh, "roe", 0.8, "actual")).toBe("0.8%");
    expect(formatClaimValue("zh", zh, "revenue", 174_120_000_000, "actual")).toBe("1,741.2 亿元");
    expect(formatClaimValue("en", en, "revenue", 174_120_000_000, "actual")).toBe("174.12B CNY");
    expect(formatClaimValue("zh", zh, null, 3, "claimed")).toBe("3");
  });

  it("reads the unit of a claimed amount back from the claim text", () => {
    expect(claimedAmount("茅台营收1700亿", 1700)).toBe(170_000_000_000);
    expect(claimedAmount("净利润约 862.3 亿元", 862.3)).toBeCloseTo(86_230_000_000);
    expect(claimedAmount("revenue of 174 billion", 174)).toBe(174_000_000_000);
    expect(claimedAmount("营收增长 20", 20)).toBe(20);
    expect(formatClaimValue("zh", zh, "revenue", 1700, "claimed", "茅台营收1,700亿")).toBe("1,700 亿元");
  });
});

describe("claim report helpers", () => {
  it("localises the checker's notes and keeps unknown ones", () => {
    expect(noteText(zh, "metric not recognised")).toBe("未能判断这个数字指的是哪个指标");
    expect(noteText(en, "no data for this metric")).toBe("The data sources have no value for this metric");
    expect(noteText(zh, "something new")).toBe("something new");
    expect(noteText(zh, "")).toBe("");
  });

  it("counts statuses and builds an evidence record for freshness", () => {
    expect(statusCounts(report)).toEqual({ supported: 0, contradicted: 1, unverifiable: 1 });
    expect(checkEvidence(report.checks[0]!, report)).toMatchObject({
      evidence_id: "fundamental_600519.SH",
      source_type: "fundamental_sql",
      kind: "structured",
      as_of: "2025-12-31",
      title: "贵州茅台 (600519.SH) fundamentals",
    });
    expect(checkEvidence(report.checks[1]!, report)).toBeNull();
  });
});
