import { localizeDisclaimer, localizeNextQuestion } from "./localize";

// (round 15, review-11 K17) the suggestions and disclaimer of a turn answered in one language follow the UI language
// after a switch: the server's templates are recognised and rebuilt; anything else is left as written.
const names = new Map([
  ["贵州茅台", "Kweichow Moutai"],
  ["600519.SH", "Kweichow Moutai"],
  ["五粮液", "Wuliangye"],
]);

describe("localizeNextQuestion", () => {
  it("rebuilds the agent's zh suggestions in English with the English names", () => {
    expect(localizeNextQuestion("贵州茅台的估值和盈利指标（PE、ROE）如何？", "en", names)).toBe("What are Kweichow Moutai's PE and ROE?");
    expect(localizeNextQuestion("贵州茅台近期的均线和 RSI 指标怎么样？", "en", names)).toBe(
      "What do Kweichow Moutai's moving averages and RSI show?",
    );
    expect(localizeNextQuestion("贵州茅台最近有哪些重要新闻或公告？", "en", names)).toBe(
      "What recent news or announcements affect Kweichow Moutai?",
    );
    expect(localizeNextQuestion("贵州茅台和五粮液在估值与盈利质量上有什么差异？", "en", names)).toBe(
      "How do Kweichow Moutai and Wuliangye differ in valuation and earnings quality?",
    );
    expect(localizeNextQuestion("这些宏观指标变化对哪些行业影响更大？", "en", names)).toBe(
      "Which sectors are most sensitive to these macro indicators?",
    );
    expect(localizeNextQuestion("我可以问哪些金融、市场或经济相关问题？", "en", names)).toBe(
      "What financial, market, or economic questions can I ask?",
    );
  });

  it("rebuilds English suggestions in Chinese, mapping English names back", () => {
    expect(localizeNextQuestion("What is the tone of recent news about Wuliangye?", "zh", names)).toBe("五粮液近期新闻的整体语气如何？");
    expect(localizeNextQuestion("How does Kweichow Moutai compare with its industry peers?", "zh", names)).toBe(
      "贵州茅台与同行业公司相比表现如何？",
    );
  });

  it("keeps a name without a known translation, a suggestion already in the UI language, and unknown text", () => {
    expect(localizeNextQuestion("宁德时代近期新闻的整体语气如何？", "en", names)).toBe("What is the tone of recent news about 宁德时代?");
    expect(localizeNextQuestion("贵州茅台近期新闻的整体语气如何？", "zh", names)).toBe("贵州茅台近期新闻的整体语气如何？");
    expect(localizeNextQuestion("贵州茅台今年分红多少？", "en", names)).toBe("贵州茅台今年分红多少？");
  });
});

describe("localizeDisclaimer", () => {
  const zh = "以上内容仅基于系统检索到的证据生成，不构成投资建议或确定性买卖结论。";
  const en = "This answer is based only on evidence retrieved by the system and is not investment advice or a deterministic buy/sell conclusion.";
  it("swaps the server's default disclaimer for the UI language and leaves other text alone", () => {
    expect(localizeDisclaimer(zh, "en")).toBe(en);
    expect(localizeDisclaimer(en, "zh")).toBe(zh);
    expect(localizeDisclaimer(zh, "zh")).toBe(zh);
    expect(localizeDisclaimer("核查只比对声明中的数字", "en")).toBe("核查只比对声明中的数字");
    expect(localizeDisclaimer(null, "en")).toBeNull();
  });
});
