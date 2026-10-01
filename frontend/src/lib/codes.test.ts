import { fallbackReasonText, formatUnknown, humanizeCode, isCode, limitationText, localizeNames } from "./codes";

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

describe("route reasons (round 12, H14)", () => {
  // Every reason format agent/graph.py `guard_in` can emit (router.py, memory.py, frame.py, graph.py).
  const REASONS = [
    "nlu:out_of_scope_query",
    "nlu:missing_entity",
    "nlu:clarification_required",
    "dangling_reference",
    "request_without_object",
    "metric_without_target",
    "ellipsis_without_antecedent",
    "no_target:recommendation",
    "no_target:advice",
    "mode:workflow",
    "mode:agent",
    "multi_entity:2",
    "comparison_targets",
    "question_style:why",
    "question_style:compare",
    "question_style:forecast",
    "intent:macro_policy_impact",
    "intent:market_explanation",
    "intent:peer_compare",
    "multi_intent:3",
    "cross_domain:macro_to_market",
    "lexical:multi_hop_marker",
    "lexical:judgment_or_timing",
    "lexical:forecast",
    "lexical:analysis_request",
    "lexical:why",
    "lexical:follow_up",
    "concept:glossary:北向资金",
    "concept:definition",
    "simple:single_lookup",
    "override:out_of_scope_with_macro_anchor",
    "override:out_of_scope_glossary_concept:换手率",
    "override:out_of_scope_with_finance_anchor",
    "override:out_of_scope_dangling_reference",
    "override:why_style_without_causal_cue",
    "dropped_fuzzy_concept:有色金属",
    "dropped_generic_noun:指数",
    "dropped_advice_phrase:值得买",
    "foreign_listing_lookalike:中国平安",
    "session_memory_over_nlu_context_carry",
    "sector_member:白酒->贵州茅台",
    "override:out_of_scope_sector_of_discussed_target",
    "holding_value",
    "off_topic_request:coding",
    "system_change_request",
    "difference_without_comparison",
    "group_reference_incomplete",
    "input_guard:instruction_like_text_removed",
    "input_guard:prediction_without_target",
    "dropped_unnamed_target_out_of_coverage:比特币ETF",
    "coverage:crypto",
    "coverage:foreign_equity",
    "alias_context:平安->平安银行",
    "alias_default:平安->中国平安|平安银行",
    "session_language:en",
    "model_policy:composition_for_slow_model",
    "session_disambiguation:平安->中国平安",
    "frame:dropped_non_operand:多氟多",
    "frame:style_compare",
    "frame:difference:roe:五粮液|贵州茅台",
    "frame:ratio:pb:中国平安|保险行业平均",
    "frame:relative:pe:中国平安|所属行业平均",
    "frame:which:net_margin:五粮液|贵州茅台|泸州老窖",
    "clarified:贵州茅台",
    "group_reference:前者->五粮液",
    "group_reference:这三家->五粮液和贵州茅台和泸州老窖",
    "group_reference_count_mismatch:三家->五粮液和贵州茅台",
    "group_reference:which->五粮液和贵州茅台",
    "coreference:这两家->宁德时代和比亚迪",
    "coreference:its->BYD",
    "coreference:它->贵州茅台",
    "ellipsis:target->贵州茅台",
    "ellipsis:target->贵州茅台+aspect->营业收入",
    "ellipsis:aspect->市盈率+走势",
    "industry_reference:这个行业->白酒",
    "difference_follow_up:五粮液和贵州茅台",
    "difference_follow_up:五粮液和贵州茅台+aspect->ROE",
    "comparison_follow_up:五粮液和贵州茅台+aspect->市盈率",
    "comparison_anchor:+贵州茅台",
    "session_inherit:target->贵州茅台",
    "session_inherit:macro->CPI、PPI",
    "dangling_why:target->五粮液",
  ];
  const RAW = /->|\||_|^[a-z]+:/;

  it.each(["zh", "en"] as const)("gives every reason a readable %s label", (lang) => {
    for (const reason of REASONS) {
      const label = humanizeCode(lang, reason, "reason");
      expect(label, reason).not.toMatch(RAW);
      // a zh label is written in Chinese (the formatted fallback for unknown codes is English words)
      if (lang === "zh") expect(label, reason).toMatch(/[\u4e00-\u9fff]/);
      // an en label has Chinese only where the code carries a name or the user's words ("Read “它” as …")
      else if (!/[\u4e00-\u9fff]/.test(reason)) expect(label, reason).not.toMatch(/[\u4e00-\u9fff]/);
    }
  });

  it("reads a frame reason as the computed comparison, with the operands in order", () => {
    const names = new Map([
      ["贵州茅台", "Kweichow Moutai"],
      ["五粮液", "Wuliangye"],
    ]);
    expect(humanizeCode("zh", "frame:difference:roe:五粮液|贵州茅台")).toBe("计算ROE差值：五粮液 对比 贵州茅台");
    expect(localizeNames("en", humanizeCode("en", "frame:difference:roe:五粮液|贵州茅台"), names)).toBe(
      "Computed the ROE gap: Wuliangye vs Kweichow Moutai",
    );
    expect(humanizeCode("en", "frame:ratio:pb:中国平安|保险行业平均")).toBe("Computed the P/B ratio: 中国平安 ÷ 保险 industry average");
    expect(humanizeCode("zh", "frame:which:net_margin:A|B|C")).toBe("比较净利率高低：A、B、C");
    expect(humanizeCode("en", "frame:which:net_margin:A|B|C")).toBe("Compared which net margin is higher: A, B and C");
  });

  it("labels the policy, language and alias reasons", () => {
    expect(humanizeCode("zh", "model_policy:composition_for_slow_model")).toBe("慢速推理模型：改用固定流程，由 LLM 撰写回答");
    expect(humanizeCode("en", "session_language:en")).toBe("Kept the session's answer language: English");
    expect(humanizeCode("zh", "alias_default:平安->中国平安|平安银行")).toBe("“平安”默认理解为 中国平安（也可能指 平安银行）");
    expect(humanizeCode("en", "ellipsis:target->贵州茅台+aspect->营收+PE")).toBe(
      "Kept the security (贵州茅台) and the question (营收, PE) from the last turn",
    );
  });

  it("formats an unknown reason readably instead of showing the raw code", () => {
    expect(humanizeCode("en", "frame:brand_new:roe:A|B")).toBe("Frame: brand new · roe · A, B");
    expect(humanizeCode("zh", "new_policy:x->y")).toBe("New policy: x → y");
    expect(formatUnknown("zh", "a_b:c|d")).toBe("A b: c、d");
  });
});
