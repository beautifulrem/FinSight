import { evidenceIndex, splitCitations, stripCitations } from "./citations";

const index = evidenceIndex(["price_600519.SH", "fundamental_600519.SH", "industry_白酒"]);

describe("splitCitations", () => {
  it("turns known evidence ids into numbered citations", () => {
    const segments = splitCitations("收盘价 1409.5 [price_600519.SH]。行业 [industry_白酒]", index);
    expect(segments).toEqual([
      { type: "text", text: "收盘价 1409.5" },
      { type: "cite", id: "price_600519.SH", index: 1 },
      { type: "text", text: "。行业" },
      { type: "cite", id: "industry_白酒", index: 3 },
    ]);
  });

  it("splits comma-separated ids and flags ids that are not in the run", () => {
    const segments = splitCitations("见 [price_600519.SH, news_404]", index);
    expect(segments).toEqual([
      { type: "text", text: "见" },
      { type: "cite", id: "price_600519.SH", index: 1 },
      { type: "invalid", id: "news_404" },
    ]);
  });

  it("resolves a case-mismatched id to its evidence item (B5: lower-cased English answers)", () => {
    expect(splitCitations("PE 24.6 [fundamental_600519.sh]", index)).toEqual([
      { type: "text", text: "PE 24.6" },
      { type: "cite", id: "fundamental_600519.SH", index: 2 },
    ]);
  });

  it("leaves ordinary bracketed text alone", () => {
    expect(splitCitations("[注意] 风险", index)).toEqual([{ type: "text", text: "[注意] 风险" }]);
  });
});

describe("stripCitations", () => {
  it("removes citation markers before punctuation", () => {
    expect(stripCitations("PE 24.6 [fundamental_600519.SH]。")).toBe("PE 24.6。");
  });
});
