import { checkEvidence, claimedAmount, claimedText, claimInMessage, formatClaimValue, noteText, statusCounts } from "./claims";
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
    expect(formatClaimValue("zh", zh, "roe", 0.33, "actual")).toBe("33%");
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

describe("comparators, reasons and claims in chat messages", () => {
  it("writes the claimed side with its comparator, and in words for screen readers", () => {
    const roe = { metric: "roe", claimed: 30, claimed_unit: "%", comparator: "gt", status: "supported" } as const;
    expect(claimedText("zh", zh, roe)).toEqual({ text: "> 30%", label: "高于 30%" });
    expect(claimedText("en", en, { ...roe, comparator: "le" })).toEqual({ text: "≤ 30%", label: "at most 30%" });
    const pe = { metric: "pe_ttm", claimed: 15, comparator: "ne", negated: true, status: "supported" } as const;
    expect(claimedText("zh", zh, pe).text).toBe("≠ 15 倍");
    expect(claimedText("zh", zh, { ...pe, comparator: "approx" }).text).toBe("≈ 15 倍");
    expect(claimedText("zh", zh, { ...pe, comparator: "eq" }).text).toBe("15 倍");
    const range = { metric: "pe_ttm", claimed: 20, claimed_high: 30, comparator: "range", status: "supported" } as const;
    expect(claimedText("zh", zh, range)).toEqual({ text: "20 倍 – 30 倍", label: "介于 20 倍 – 30 倍" });
    expect(claimedText("zh", zh, { ...range, negated: true }).text).toBe("∉ 20 倍 – 30 倍");
    // a number-less move ("昨天下跌了") is a check against 0
    expect(claimedText("zh", zh, { metric: "pct_change_1d", claimed: 0, comparator: "lt", status: "supported" }).text).toBe(
      "< 0%",
    );
  });

  it("formats growth and margins in percent", () => {
    expect(formatClaimValue("zh", zh, "revenue_yoy", 1.4699, "actual")).toBe("+1.47%");
    expect(formatClaimValue("zh", zh, "netprofit_yoy", -2, "claimed")).toBe("-2%");
    expect(formatClaimValue("zh", zh, "gross_margin", 76.1, "actual")).toBe("76.1%");
  });

  it("localises reason codes with the data date", () => {
    const check = { claimed: 16, status: "unverifiable", reason: "period_mismatch", as_of: "2026-06-30" } as const;
    expect(noteText(zh, "the claim is about 2019; the data is for 2026-06-30", check)).toBe(
      "说法所指的期间与数据（2026-06-30）不同",
    );
    expect(noteText(en, "", { claimed: 16, status: "unverifiable", reason: "growth_unavailable" })).toBe(
      "The source has no year-on-year growth for this item",
    );
    expect(
      noteText(zh, "compared with the report for the period ending 2026-06-30 (year to date)", {
        claimed: 16,
        status: "supported",
        as_of: "2026-06-30",
      }),
    ).toBe("按截至 2026-06-30 的累计报告比对");
  });

  it("finds the claim in hearsay questions typed in the chat", () => {
    expect(claimInMessage("听说茅台市盈率只有15倍，是真的吗")).toBe("茅台市盈率只有15倍");
    expect(claimInMessage("据说五粮液ROE超过三成？")).toBe("五粮液ROE超过三成");
    expect(claimInMessage("茅台昨天跌了5%，真的吗？")).toBe("茅台昨天跌了5%");
    expect(claimInMessage("I heard that Moutai's P/E is only 15x, is that true?")).toBe("Moutai's P/E is only 15x");
    expect(claimInMessage("Is it true that Moutai fell 5% yesterday?")).toBe("Moutai fell 5% yesterday");
    // ordinary questions and hearsay without a number are left alone
    expect(claimInMessage("贵州茅台最近走势怎么样？")).toBeNull();
    expect(claimInMessage("茅台市盈率是多少")).toBeNull();
    expect(claimInMessage("听说茅台管理层很好，是真的吗")).toBeNull();
  });
});
