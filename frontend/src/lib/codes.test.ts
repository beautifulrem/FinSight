import { fallbackReasonText, humanizeCode, isCode, limitationText, localizeNames } from "./codes";

describe("humanizeCode", () => {
  it("labels the codes the backend emits in both languages", () => {
    expect(humanizeCode("zh", "nlu:out_of_scope_query")).toBe("不属于金融问答范围");
    expect(humanizeCode("en", "out_of_scope_query")).toBe("Question is outside finance");
    expect(humanizeCode("en", "verification_failed:repaired")).toBe("Verification failed; answer repaired");
    expect(humanizeCode("zh", "language_mismatch_fallback_to_template")).toContain("模板");
    expect(humanizeCode("en", "instruction_like_text_removed_from_tool_output")).toMatch(/instruction-like/i);
    expect(humanizeCode("en", "instruction_like_text_removed_in_future_place")).toMatch(/instruction-like/i);
    expect(humanizeCode("zh", "multi_entity:2")).toBe("涉及 2 只证券");
    expect(humanizeCode("en", "question_style:why")).toBe("Question type: why / cause");
    expect(humanizeCode("zh", "intent:peer_compare")).toBe("意图：同业对比");
    expect(humanizeCode("zh", "repeated_tool_calls:2")).toBe("拦截了 2 次重复的工具调用");
    expect(humanizeCode("en", "lexical:judgment_or_timing")).toBe("Judgment or timing wording");
    expect(humanizeCode("en", "out_of_coverage")).toMatch(/A-shares, funds, ETFs, indices/);
    expect(humanizeCode("zh", "coverage:crypto")).toBe("加密资产不在数据覆盖范围内");
    expect(humanizeCode("zh", "dangling_why:target->五粮液")).toBe("追问原因，沿用上一轮的标的 五粮液");
    expect(isCode("out_of_coverage")).toBe(true);
  });

  it("parses parameterised codes without leaking exception text", () => {
    expect(humanizeCode("zh", "budget:step budget of 6 reached")).toBe("达到推理步数上限（6 步），提前作答");
    expect(humanizeCode("en", "budget:run deadline of 45s reached")).toBe("Run deadline (45 s) reached; answered early");
    expect(humanizeCode("en", "llm_error:HTTP 503 upstream")).toBe("LLM call failed; fell back");
    expect(humanizeCode("zh", "no_llm_configured:something_new")).toContain("未配置 LLM");
    expect(humanizeCode("zh", "stock_zh_a_hist_skipped:circuit_open:eastmoney.quote")).toBe("stock_zh_a_hist 熔断中，已跳过");
  });

  it("uses kind-specific tables and prettifies unknown codes instead of showing them raw", () => {
    expect(humanizeCode("zh", "refuse", "route")).toBe("超出范围（拒答）");
    expect(humanizeCode("en", "llm_agent", "answerSource")).toBe("Written by the LLM agent");
    expect(humanizeCode("zh", "timeout", "toolError")).toBe("超时");
    expect(humanizeCode("en", "brand_new_reason")).toBe("Brand new reason");
  });
});

describe("follow-up rewrite reasons", () => {
  it("explains carried-over targets and questions with the suffix as data", () => {
    expect(humanizeCode("zh", "ellipsis:target->贵州茅台")).toBe("沿用上一轮的标的 贵州茅台");
    expect(humanizeCode("zh", "ellipsis:target->宁德时代和比亚迪")).toBe("沿用上一轮的标的 宁德时代和比亚迪");
    expect(humanizeCode("en", "ellipsis:target->Kweichow Moutai")).toBe("Kept the security from the last turn: Kweichow Moutai");
    expect(humanizeCode("zh", "ellipsis:aspect->市盈率+走势")).toBe("沿用上一轮的问题：市盈率、走势");
    // The server joins aspects with "+"; "P/E" keeps its slash.
    expect(humanizeCode("en", "ellipsis:aspect->P/E+trend")).toBe("Kept the question from the last turn: P/E, trend");
    expect(humanizeCode("zh", "ellipsis:aspect->市净率+PB")).toBe("沿用上一轮的问题：市净率、PB");
    expect(humanizeCode("zh", "ellipsis:something_new")).toBe("补全了省略的追问");
  });

  it("explains resolved pronouns, clarifications and dropped fuzzy concepts", () => {
    expect(humanizeCode("zh", "coreference:它->贵州茅台")).toBe("将“它”理解为 贵州茅台");
    expect(humanizeCode("en", "coreference:its->BYD")).toBe("Read “its” as BYD");
    expect(humanizeCode("zh", "coreference:这两家->宁德时代和比亚迪")).toBe("将“这两家”理解为 宁德时代和比亚迪");
    expect(humanizeCode("zh", "clarified:600519.SH")).toBe("按澄清回复补全为 600519.SH");
    expect(humanizeCode("zh", "dropped_fuzzy_concept:有色金属")).toBe("忽略了模糊匹配到的概念“有色金属”");
    expect(humanizeCode("en", "dropped_fuzzy_concept:Nonferrous metals")).toBe("Ignored the loosely matched concept “Nonferrous metals”");
  });

  it("labels the router's exact codes for follow-ups", () => {
    expect(humanizeCode("zh", "metric_without_target")).toBe("只问了指标，没有指明证券");
    expect(humanizeCode("en", "metric_without_target")).toBe("Asked for a metric without naming a security");
    expect(humanizeCode("zh", "input_guard:instruction_like_text_removed")).toBe("已移除问题中疑似指令的文本");
    for (const code of ["ellipsis:target->贵州茅台", "coreference:它->贵州茅台", "dropped_fuzzy_concept:有色金属", "metric_without_target"]) {
      expect(isCode(code)).toBe(true);
      expect(humanizeCode("en", code)).not.toContain(code);
    }
  });

  it("localises claim-check metric names", () => {
    expect(humanizeCode("zh", "pe_ttm", "metric")).toBe("市盈率(TTM)");
    expect(humanizeCode("en", "pct_change_1d", "metric")).toBe("Daily change");
    expect(humanizeCode("zh", "net_profit", "metric")).toBe("净利润");
    expect(humanizeCode("en", "dividend_yield", "metric")).toBe("Dividend yield");
  });
});

describe("isCode / limitationText", () => {
  it("separates machine codes from prose limitations", () => {
    expect(isCode("out_of_scope_query")).toBe(true);
    expect(isCode("budget:step budget of 6 reached")).toBe(true);
    expect(isCode("查询内容不属于金融范畴")).toBe(false);
    expect(isCode("Retrieval confidence is low")).toBe(false);
    expect(isCode("E1")).toBe(false);
    expect(limitationText("zh", "out_of_scope_query")).toEqual({ text: "问题不属于金融范畴", code: "out_of_scope_query" });
    expect(limitationText("zh", "无相关金融证据可用")).toEqual({ text: "无相关金融证据可用" });
  });
});

describe("fallbackReasonText", () => {
  it("renders source attempts and known reasons", () => {
    expect(fallbackReasonText("zh", "eastmoney.quote:error(ProxyError); eastmoney.quote:circuit_open")).toBe(
      "eastmoney.quote 请求失败；eastmoney.quote 熔断中",
    );
    expect(fallbackReasonText("en", "live source returned no data for this record")).toBe("the live source returned no data for it");
  });
});


describe("localizeNames (round 9, E13)", () => {
  const names = new Map([
    ["贵州茅台", "Kweichow Moutai"],
    ["五粮液", "Wuliangye"],
  ]);

  it("writes the names a route reason carries in English", () => {
    const label = humanizeCode("en", "ellipsis:target->五粮液和贵州茅台", "reason");
    expect(localizeNames("en", label, names)).toBe("Kept the security from the last turn: Wuliangye and Kweichow Moutai");
    expect(localizeNames("en", "A、B 和 C", new Map([["A", "a"]]))).toBe("a, B and C");
  });

  it("leaves Chinese and name-free text alone", () => {
    const label = humanizeCode("zh", "ellipsis:target->五粮液和贵州茅台", "reason");
    expect(localizeNames("zh", label, names)).toBe(label);
    expect(localizeNames("en", "Simple single lookup", names)).toBe("Simple single lookup");
    expect(localizeNames("en", "沿用", undefined)).toBe("沿用");
  });
});
