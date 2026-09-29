import {
  checkEvidence,
  claimedAmount,
  claimedText,
  claimInMessage,
  formatClaimValue,
  noteText,
  statusCounts,
  targetName,
} from "./claims";
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

describe("moves, relations and macro claims (round 4)", () => {
  it("writes a bound on a move as the size of the move in its direction", () => {
    // "五粮液昨天跌了超过1%": claimed -1, gt, down -> "跌幅 > 1%", not "> -1%"
    const fall = { metric: "pct_change_1d", claimed: -1, comparator: "gt", direction: "down", status: "contradicted" } as const;
    expect(claimedText("zh", zh, fall)).toEqual({ text: "跌幅 > 1%", label: "跌幅高于 1%" });
    expect(claimedText("en", en, fall)).toEqual({ text: "Fall > 1%", label: "Fall more than 1%" });
    expect(claimedText("zh", zh, { ...fall, comparator: "lt" }).text).toBe("跌幅 < 1%");
    expect(claimedText("zh", zh, { ...fall, comparator: "le", negated: true }).text).toBe("跌幅 ≤ 1%");
    const rise = { metric: "pct_change_1d", claimed: 0.1, comparator: "gt", direction: "up", status: "contradicted" } as const;
    expect(claimedText("zh", zh, rise).text).toBe("涨幅 > 0.1%");
    const range = { ...fall, claimed: -0.1, claimed_high: -0.3, comparator: "range" } as const;
    expect(claimedText("zh", zh, range)).toEqual({ text: "跌幅 0.1% – 0.3%", label: "跌幅介于 0.1% – 0.3%" });
    // an exact move keeps its sign
    expect(claimedText("zh", zh, { ...fall, comparator: "eq", claimed: -0.53 }).text).toBe("-0.53%");
  });

  it("writes a relation as the other side, with its value", () => {
    const relation = {
      target: "贵州茅台",
      metric: "pe_ttm",
      claimed: null,
      comparator: "gt",
      reference: "五粮液",
      reference_value: 20.9,
      actual: 24.6,
      status: "supported",
    } as const;
    expect(claimedText("zh", zh, relation)).toEqual({ text: "> 五粮液", label: "高于 五粮液", detail: "五粮液 20.9 倍" });
    const names = (name: string) => (name === "五粮液" ? "Wuliangye" : name);
    expect(claimedText("en", en, relation, "", names)).toEqual({
      text: "> Wuliangye",
      label: "more than Wuliangye",
      detail: "Wuliangye 20.9×",
    });
  });

  it("formats macro readings and names targets in English", () => {
    expect(formatClaimValue("zh", zh, "cpi_yoy", 0.8, "actual")).toBe("+0.8%");
    expect(formatClaimValue("zh", zh, "cn10y", 2.31, "actual")).toBe("2.31%");
    expect(formatClaimValue("en", en, "pmi", 50.6, "actual")).toBe("50.6");
    expect(noteText(en, "", { claimed: null, status: "unverifiable", reason: "no_reference" })).toBe(
      "The data sources have nothing to compare with (e.g. a market average)",
    );
    const named: ClaimReport = { ...report, targets: [{ name: "贵州茅台", symbol: "600519.SH", name_en: "Kweichow Moutai" }] };
    expect(targetName("en", named)("贵州茅台")).toBe("Kweichow Moutai");
    expect(targetName("zh", named)("贵州茅台")).toBe("贵州茅台");
    expect(checkEvidence({ claimed: 0.8, status: "supported", evidence_id: "macro_CPI_CN" }, report)?.source_type).toBe("macro_sql");
  });

  it("finds relational hearsay", () => {
    expect(claimInMessage("听说茅台的市盈率比五粮液高，对吗")).toBe("茅台的市盈率比五粮液高");
  });
});
