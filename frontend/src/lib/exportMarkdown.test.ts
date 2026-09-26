import { answerToMarkdown, exportFilename } from "./exportMarkdown";
import type { AnswerView } from "./view";

const view: AnswerView = {
  kind: "agent",
  answer: "贵州茅台收于 1413 元 [price_600519.SH]。",
  keyPoints: ["PE 24.6 [fundamental_600519.SH]"],
  limitations: ["out_of_scope_query", "白酒行业快照较旧"],
  disclaimer: "不构成投资建议。",
  evidence: [
    {
      evidence_id: "price_600519.SH",
      kind: "structured",
      source_type: "market_api",
      title: "贵州茅台 daily",
      as_of: "2026-09-24",
      payload: { provenance: { mode: "live_fallback", source_label: "新浪财经行情", fallback_reason: "eastmoney.quote:timeout" } },
    },
    { evidence_id: "news_1", kind: "document", source_type: "news", title: "年报 | 摘要", source_url: "https://example.com/a", as_of: "2026-09-01" },
  ],
  cited: new Set(["price_600519.SH"]),
  route: "agent",
  next: [],
  trace: [],
  verification: { passed: true },
  agent: { status: "ok", session_id: "s1", trace_id: "abc123", llm: { model: "m", calls: 1, steps: 1, usage: { prompt_tokens: 1, completion_tokens: 1, prompt_cache_hit_tokens: 0, reasoning_tokens: 0, total_tokens: 2 } } },
};

describe("answerToMarkdown", () => {
  it("exports the answer with numbered evidence, provenance and the disclaimer", () => {
    const md = answerToMarkdown(view, { query: "茅台走势", lang: "zh", now: new Date("2026-09-26T00:00:00Z"), disclaimerFallback: "x" });
    expect(md).toContain("# 茅台走势");
    expect(md).toContain("Trace ID: `abc123`");
    expect(md).toContain("收于 1413 元[E1]。");
    expect(md).toContain("- PE 24.6\n");
    expect(md).not.toContain("fundamental_600519.SH");
    expect(md).toContain("- 问题不属于金融范畴 (`out_of_scope_query`)");
    expect(md).toMatch(/1\. \*\*E1\*\* 贵州茅台 daily · `price_600519.SH` · 行情 · 来自 新浪财经行情 · 截至 .*2026.* · 已引用, 备用源/);
    expect(md).toContain("降级原因: eastmoney.quote 超时");
    expect(md).toContain("年报 \\| 摘要");
    expect(md).toContain("<https://example.com/a>");
    expect(md.trim().endsWith("> 不构成投资建议。")).toBe(true);
  });

  it("builds a safe file name", () => {
    expect(exportFilename("贵州茅台/最近 走势?", new Date(2026, 8, 26, 1, 0))).toBe("finsight-2026-09-26-贵州茅台-最近-走势.md");
  });
});
