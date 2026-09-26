import { fallbackReasonText, humanizeCode, isCode, limitationText } from "./codes";

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
