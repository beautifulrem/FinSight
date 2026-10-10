import type { Lang } from "./i18n";

/**
 * (round 15, review-11 K17) A turn keeps the language it was answered in, but its next-question chips and its
 * disclaimer are UI chrome: after a zh → en switch they should read in English. The server writes them from fixed
 * templates (`query_intelligence/agent/followups.py`, `answer_guards._out_of_scope_next_question_response`,
 * `chat/language.py` DEFAULT_RISK_DISCLAIMER_*), so the client recognises a template in either language and rebuilds
 * it in the UI language, with company names translated through the response's `name_en` map. Text that matches no
 * template is left exactly as written.
 */

type Pair = { zh: string; en: string };

// {a} and {b} are company / fund names. Keep in step with followups.py.
const QUESTIONS: Pair[] = [
  { zh: "{a}和{b}在估值与盈利质量上有什么差异？", en: "How do {a} and {b} differ in valuation and earnings quality?" },
  { zh: "{a}的估值和盈利指标（PE、ROE）如何？", en: "What are {a}'s PE and ROE?" },
  { zh: "{a}近期的均线和 RSI 指标怎么样？", en: "What do {a}'s moving averages and RSI show?" },
  { zh: "{a}最近有哪些重要新闻或公告？", en: "What recent news or announcements affect {a}?" },
  { zh: "{a}近期新闻的整体语气如何？", en: "What is the tone of recent news about {a}?" },
  { zh: "{a}与同行业公司相比表现如何？", en: "How does {a} compare with its industry peers?" },
  { zh: "这些宏观指标变化对哪些行业影响更大？", en: "Which sectors are most sensitive to these macro indicators?" },
  { zh: "还需要哪些证据才能更完整地回答这个问题？", en: "What further evidence would make this answer more complete?" },
  { zh: "我可以问哪些金融、市场或经济相关问题？", en: "What financial, market, or economic questions can I ask?" },
  {
    zh: "如果要分析一个具体问题，需要补充哪些证据或数据？",
    en: "What evidence or data should I provide for a specific analysis question?",
  },
  {
    zh: "如何判断当前证据是否足以支持结论？",
    en: "How can I tell whether the current evidence is enough to support a conclusion?",
  },
];

const DISCLAIMER: Pair = {
  zh: "以上内容仅基于系统检索到的证据生成，不构成投资建议或确定性买卖结论。",
  en: "This answer is based only on evidence retrieved by the system and is not investment advice or a deterministic buy/sell conclusion.",
};

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function matcher(template: string): RegExp {
  const parts = template.split(/(\{[ab]\})/);
  return new RegExp(`^${parts.map((part) => (/^\{[ab]\}$/.test(part) ? "(.+?)" : escape(part))).join("")}$`);
}

const COMPILED = QUESTIONS.map((pair) => ({ pair, zh: matcher(pair.zh), en: matcher(pair.en) }));

function translateName(name: string, to: Lang, names: Map<string, string> | undefined): string {
  if (!names) return name;
  if (to === "en") return names.get(name) ?? name;
  for (const [native, english] of names) {
    if (english === name && !/^\d{6}\.[A-Z]{2}$/.test(native)) return native;
  }
  return name;
}

function fill(template: string, values: string[]): string {
  let index = 0;
  return template.replace(/\{[ab]\}/g, () => values[index++] ?? "");
}

/** A server next-question in the UI language, or the question unchanged when it matches no template. */
export function localizeNextQuestion(question: string, lang: Lang, names?: Map<string, string>): string {
  const from: Lang = lang === "en" ? "zh" : "en";
  const text = question.trim();
  for (const { pair, ...regex } of COMPILED) {
    if (regex[lang].test(text)) return question;
    const found = regex[from].exec(text);
    if (found) return fill(pair[lang], found.slice(1).map((name) => translateName(name, lang, names)));
  }
  return question;
}

/** The server's default risk disclaimer in the UI language; any other disclaimer unchanged. */
export function localizeDisclaimer(text: string | null | undefined, lang: Lang): string | null | undefined {
  if (!text) return text;
  const trimmed = text.trim();
  if (trimmed === DISCLAIMER.zh || trimmed === DISCLAIMER.en) return DISCLAIMER[lang];
  return text;
}
