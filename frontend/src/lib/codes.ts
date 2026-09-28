import type { Lang } from "./i18n";

/**
 * Human-readable labels for the internal codes the backend reports: route reasons, `degraded` flags,
 * compliance notes, NLU risk flags, retrieval warnings, tool error codes and answer sources.
 * The UI shows the label and keeps the raw code in a tooltip (see `CodeLabel`), so users never read
 * `out_of_scope_query` while power users can still copy the exact code.
 */
export type CodeKind =
  | "reason"
  | "degraded"
  | "compliance"
  | "risk"
  | "warning"
  | "toolError"
  | "answerSource"
  | "route"
  | "style"
  | "product"
  | "intent"
  | "limitation"
  | "metric";

type Text = { zh: string; en: string };
type Rule = { test: RegExp; label: (match: RegExpExecArray, lang: Lang) => string };

const pick = (lang: Lang, text: Text) => text[lang];

const ROUTES: Record<string, Text> = {
  workflow: { zh: "固定流程", en: "Workflow" },
  agent: { zh: "LLM 智能体", en: "LLM agent" },
  refuse: { zh: "超出范围（拒答）", en: "Out of scope (refused)" },
  clarify: { zh: "需要澄清", en: "Clarification" },
  classic: { zh: "经典管线", en: "Classic pipeline" },
};

const STYLES: Record<string, Text> = {
  why: { zh: "原因分析", en: "why / cause" },
  compare: { zh: "对比", en: "comparison" },
  forecast: { zh: "走势预测", en: "forecast" },
  advice: { zh: "投资建议类", en: "advice-seeking" },
  fact: { zh: "事实查询", en: "fact lookup" },
  lookup: { zh: "数据查询", en: "lookup" },
};

const PRODUCTS: Record<string, Text> = {
  stock: { zh: "股票", en: "stock" },
  etf: { zh: "ETF", en: "ETF" },
  fund: { zh: "基金", en: "fund" },
  index: { zh: "指数", en: "index" },
  macro: { zh: "宏观", en: "macro" },
  sector: { zh: "行业", en: "sector" },
  bond: { zh: "债券", en: "bond" },
  out_of_scope: { zh: "非金融问题", en: "not financial" },
  unknown: { zh: "未确定", en: "undetermined" },
};

const INTENTS: Record<string, Text> = {
  market_explanation: { zh: "行情归因", en: "market move explanation" },
  macro_policy_impact: { zh: "宏观政策影响", en: "macro/policy impact" },
  peer_compare: { zh: "同业对比", en: "peer comparison" },
  price_query: { zh: "行情查询", en: "price lookup" },
  fundamental_analysis: { zh: "基本面分析", en: "fundamentals" },
  valuation_analysis: { zh: "估值分析", en: "valuation" },
  news_query: { zh: "新闻查询", en: "news" },
  sentiment_analysis: { zh: "情绪分析", en: "sentiment" },
  risk_analysis: { zh: "风险分析", en: "risk" },
};

/** Metrics named by the claim checker (query_intelligence/agent/claim_check.py `_METRICS`). */
const METRICS: Record<string, Text> = {
  close: { zh: "收盘价", en: "Close price" },
  pct_change_1d: { zh: "日涨跌幅", en: "Daily change" },
  pe_ttm: { zh: "市盈率(TTM)", en: "P/E (TTM)" },
  pe: { zh: "市盈率", en: "P/E" },
  pb: { zh: "市净率", en: "P/B" },
  roe: { zh: "净资产收益率(ROE)", en: "Return on equity (ROE)" },
  revenue: { zh: "营业收入", en: "Revenue" },
  net_profit: { zh: "净利润", en: "Net profit" },
  revenue_yoy: { zh: "营收同比增速", en: "Revenue growth (YoY)" },
  netprofit_yoy: { zh: "净利润同比增速", en: "Net profit growth (YoY)" },
  gross_margin: { zh: "毛利率", en: "Gross margin" },
  net_margin: { zh: "净利率", en: "Net margin" },
  dividend_yield: { zh: "股息率", en: "Dividend yield" },
  market_cap: { zh: "总市值", en: "Market cap" },
  eps: { zh: "每股收益", en: "EPS" },
  debt_ratio: { zh: "资产负债率", en: "Debt-to-assets" },
};

const ANSWER_SOURCES: Record<string, Text> = {
  llm_agent: { zh: "LLM 智能体撰写", en: "Written by the LLM agent" },
  llm_compose: { zh: "LLM 基于证据撰写", en: "LLM-written from evidence" },
  template: { zh: "确定性模板", en: "Deterministic template" },
  guardrail: { zh: "范围守卫模板", en: "Scope-guard template" },
  clarification: { zh: "澄清问题", en: "Clarifying question" },
  llm: { zh: "LLM", en: "LLM" },
  ok: { zh: "LLM", en: "LLM" },
  fallback: { zh: "结构化摘要（LLM 不可用）", en: "Structured summary (LLM unavailable)" },
};

const TOOL_ERRORS: Record<string, Text> = {
  unknown_tool: { zh: "未知工具", en: "Unknown tool" },
  invalid_arguments: { zh: "参数无效", en: "Invalid arguments" },
  timeout: { zh: "超时", en: "Timed out" },
  upstream_error: { zh: "数据源出错", en: "Upstream source error" },
  not_found: { zh: "未找到数据", en: "No data found" },
  unavailable: { zh: "暂不可用", en: "Unavailable" },
  internal: { zh: "内部错误", en: "Internal error" },
};

/** Exact codes: route reasons, degraded flags, compliance notes, risk flags, retrieval warnings. */
const EXACT: Record<string, Text> = {
  // Router
  "nlu:out_of_scope_query": { zh: "不属于金融问答范围", en: "Outside financial Q&A" },
  "nlu:missing_entity": { zh: "未识别到具体证券", en: "No specific security named" },
  "nlu:clarification_required": { zh: "问题需要先澄清", en: "Question needs clarifying" },
  dangling_reference: { zh: "指代不明（如“它”）", en: "Unresolved reference (e.g. “it”)" },
  comparison_targets: { zh: "包含比较对象", en: "Has comparison targets" },
  "cross_domain:macro_to_market": { zh: "宏观到市场的跨域问题", en: "Links macro data to markets" },
  "lexical:multi_hop_marker": { zh: "含多步推理用语", en: "Multi-step wording" },
  "lexical:follow_up": { zh: "追问", en: "Follow-up question" },
  "lexical:judgment_or_timing": { zh: "含判断或择时用语", en: "Judgment or timing wording" },
  "lexical:why": { zh: "询问原因", en: "Asks why" },
  metric_without_target: { zh: "只问了指标，没有指明证券", en: "Asked for a metric without naming a security" },
  "input_guard:instruction_like_text_removed": {
    zh: "已移除问题中疑似指令的文本",
    en: "Removed instruction-like text from the question",
  },
  "coverage:crypto": { zh: "加密资产不在数据覆盖范围内", en: "Crypto assets are not covered" },
  "coverage:foreign_equity": { zh: "美股/港股不在数据覆盖范围内", en: "US / Hong Kong stocks are not covered" },
  "override:out_of_scope_dangling_reference": {
    zh: "指代不明的金融追问，改为澄清",
    en: "Dangling finance follow-up: asked to clarify",
  },
  "mode:agent": { zh: "已选择 LLM 智能体模式", en: "LLM agent mode selected" },
  "mode:workflow": { zh: "已选择固定流程模式", en: "Workflow mode selected" },
  "simple:single_lookup": { zh: "简单单点查询", en: "Simple single lookup" },
  "override:out_of_scope_with_macro_anchor": {
    zh: "含宏观关键词，改判为金融问题",
    en: "Macro keyword: treated as a finance question",
  },
  "override:out_of_scope_with_finance_anchor": {
    zh: "含金融关键词，改判为金融问题",
    en: "Finance keyword: treated as a finance question",
  },
  // Degraded
  "verification_failed:repaired": { zh: "核验未通过，已自动修复回答", en: "Verification failed; answer repaired" },
  "no_llm_configured:agent_route_downgraded_to_workflow": {
    zh: "未配置 LLM，已改用固定流程",
    en: "No LLM configured; used the workflow instead",
  },
  instruction_like_text_removed_from_evidence: {
    zh: "已移除证据中疑似指令的文本",
    en: "Removed instruction-like text from evidence",
  },
  instruction_like_text_removed_from_tool_output: {
    zh: "已移除工具输出中疑似指令的文本",
    en: "Removed instruction-like text from tool output",
  },
  language_mismatch_fallback_to_template: {
    zh: "回答语言与提问不一致，已改用模板回答",
    en: "Answer language did not match the question; used the template answer",
  },
  // Cost
  price_table: { zh: "按价格表计算", en: "from the price table" },
  provider_reported: { zh: "服务商计费", en: "provider-reported" },
  // Compliance
  softened_judgment_or_causal_language: { zh: "已弱化判断或因果措辞", en: "Softened judgment or causal wording" },
  removed_trading_instruction: { zh: "已删除交易指令", en: "Removed a trading instruction" },
  conditional_prefix: { zh: "已加上条件性说明", en: "Added a conditional preface" },
  causal_caveat: { zh: "已加上因果关系提示", en: "Added a causality caveat" },
  market_freshness: { zh: "已提示行情数据时效", en: "Added a market-data freshness note" },
  // NLU risk flags and retrieval warnings
  out_of_scope_query: { zh: "问题不属于金融范畴", en: "Question is outside finance" },
  prompt_injection_request: { zh: "请求包含改变系统设定的指令", en: "Request tried to change the system setup" },
  out_of_coverage: {
    zh: "超出数据覆盖范围（仅覆盖 A 股、基金、ETF、指数和中国宏观）",
    en: "Outside data coverage (A-shares, funds, ETFs, indices and China macro only)",
  },
  entity_ambiguous: { zh: "证券名称有歧义", en: "Ambiguous security name" },
  entity_not_found: { zh: "未找到对应证券", en: "Security not found" },
  investment_advice_like: { zh: "类似投资建议的问题", en: "Advice-like question" },
  clarification_required: { zh: "需要澄清", en: "Needs clarification" },
  clarification_required_missing_entity: { zh: "缺少证券名称，需要澄清", en: "Security missing; needs clarification" },
  announcement_not_found_recent_window: { zh: "近期未检索到公告", en: "No recent announcements found" },
};

const BUDGETS: [RegExp, (m: RegExpExecArray) => Text][] = [
  [/step budget of (\d+)/, (m) => ({ zh: `达到推理步数上限（${m[1]} 步），提前作答`, en: `Step budget (${m[1]}) reached; answered early` })],
  [/tool-call budget of (\d+)/, (m) => ({ zh: `达到工具调用上限（${m[1]} 次），提前作答`, en: `Tool-call budget (${m[1]}) reached; answered early` })],
  [/token budget of (\d+)/, (m) => ({ zh: `达到 token 预算（${m[1]}），提前作答`, en: `Token budget (${m[1]}) reached; answered early` })],
  [/run deadline of ([\d.]+)s/, (m) => ({ zh: `达到运行时限（${m[1]} 秒），提前作答`, en: `Run deadline (${m[1]} s) reached; answered early` })],
];

// Aspects are joined with "+" by the server ("市盈率+市净率"; "P/E" keeps its slash).
const aspects = (value: string, separator: string) => value.split("+").filter(Boolean).join(separator);

/**
 * Follow-up rewrites (agent/memory.py, agent/router.py). The suffix after ":" or "->" is user-facing
 * text (a security name, the carried-over question, the pronoun that was resolved).
 */
const REWRITES: [RegExp, (m: RegExpExecArray) => Text][] = [
  [/^ellipsis:target->(.+)$/, (m) => ({ zh: `沿用上一轮的标的 ${m[1]}`, en: `Kept the security from the last turn: ${m[1]}` })],
  [
    /^ellipsis:aspect->(.+)$/,
    (m) => ({ zh: `沿用上一轮的问题：${aspects(m[1]!, "、")}`, en: `Kept the question from the last turn: ${aspects(m[1]!, ", ")}` }),
  ],
  [/^ellipsis(?::.*)?$/, () => ({ zh: "补全了省略的追问", en: "Completed a shortened follow-up" })],
  [/^dangling_why:target->(.+)$/, (m) => ({ zh: `追问原因，沿用上一轮的标的 ${m[1]}`, en: `“Why” follow-up about the last turn's ${m[1]}` })],
  [/^coreference:(.+?)->(.+)$/, (m) => ({ zh: `将“${m[1]}”理解为 ${m[2]}`, en: `Read “${m[1]}” as ${m[2]}` })],
  [/^coreference(?::.*)?$/, () => ({ zh: "根据上文解析了指代", en: "Resolved a reference from earlier turns" })],
  [/^clarified:(.+)$/, (m) => ({ zh: `按澄清回复补全为 ${m[1]}`, en: `Completed with your clarification: ${m[1]}` })],
  [
    /^dropped_fuzzy_concept:(.+)$/,
    (m) => ({ zh: `忽略了模糊匹配到的概念“${m[1]}”`, en: `Ignored the loosely matched concept “${m[1]}”` }),
  ],
];

const RULES: Rule[] = [
  {
    test: /^(?:ellipsis|coreference|clarified|dropped_fuzzy_concept|dangling_why)(?::|$)/,
    label: (m, lang) => {
      for (const [pattern, text] of REWRITES) {
        const hit = pattern.exec(m.input);
        if (hit) return pick(lang, text(hit));
      }
      return prettify(m.input);
    },
  },
  {
    test: /^multi_entity:(\d+)$/,
    label: (m, lang) => (lang === "zh" ? `涉及 ${m[1]} 只证券` : `${m[1]} securities involved`),
  },
  {
    test: /^multi_intent:(\d+)$/,
    label: (m, lang) => (lang === "zh" ? `包含 ${m[1]} 个意图` : `${m[1]} intents`),
  },
  {
    test: /^question_style:(.+)$/,
    label: (m, lang) => (lang === "zh" ? `问题类型：${styleLabel(lang, m[1]!)}` : `Question type: ${styleLabel(lang, m[1]!)}`),
  },
  {
    test: /^intent:(.+)$/,
    label: (m, lang) => (lang === "zh" ? `意图：${intentLabel(lang, m[1]!)}` : `Intent: ${intentLabel(lang, m[1]!)}`),
  },
  {
    test: /^repeated_tool_calls:(\d+)$/,
    label: (m, lang) =>
      lang === "zh" ? `拦截了 ${m[1]} 次重复的工具调用` : `Blocked ${m[1]} repeated tool call${m[1] === "1" ? "" : "s"}`,
  },
  {
    test: /^budget:(.*)$/,
    label: (m, lang) => {
      for (const [pattern, text] of BUDGETS) {
        const hit = pattern.exec(m[1] ?? "");
        if (hit) return pick(lang, text(hit));
      }
      return lang === "zh" ? "达到运行预算，提前作答" : "Run budget reached; answered early";
    },
  },
  {
    test: /^verification_failed(?::.*)?$/,
    label: (_m, lang) => (lang === "zh" ? "核验未通过" : "Verification failed"),
  },
  {
    test: /^no_llm_configured(?::.*)?$/,
    label: (_m, lang) => (lang === "zh" ? "未配置 LLM，使用确定性路径" : "No LLM configured; deterministic path used"),
  },
  {
    test: /^instruction_like_text_removed/,
    label: (_m, lang) => (lang === "zh" ? "已移除疑似注入的指令文本" : "Removed instruction-like (injection) text"),
  },
  {
    test: /^llm_error(?::.*)?$/,
    label: (_m, lang) => (lang === "zh" ? "LLM 调用失败，已降级处理" : "LLM call failed; fell back"),
  },
  {
    test: /^llm_compose_failed(?::.*)?$/,
    label: (_m, lang) => (lang === "zh" ? "LLM 撰写失败，改用模板回答" : "LLM answer failed; used the template"),
  },
  {
    test: /^llm_revision_failed(?::.*)?$/,
    label: (_m, lang) => (lang === "zh" ? "LLM 修订失败，保留原回答" : "LLM revision failed; kept the draft"),
  },
  {
    test: /^language_mismatch/,
    label: (_m, lang) => pick(lang, EXACT.language_mismatch_fallback_to_template!),
  },
  {
    test: /^market_served_last_known_good:([^:]+)/,
    label: (m, lang) =>
      lang === "zh" ? `${m[1]} 行情沿用最近一次成功获取的数据` : `${m[1]} prices: served the last known good data`,
  },
  {
    test: /^market_provider_empty_rows:([^:]+):([^:]+)/,
    label: (m, lang) => (lang === "zh" ? `${m[1]} 未返回 ${m[2]} 的行情` : `${m[1]} returned no prices for ${m[2]}`),
  },
  {
    test: /^([\w.]+?)_retry_succeeded(?::.*)?$/,
    label: (m, lang) => (lang === "zh" ? `${m[1]} 重试后成功` : `${m[1]} succeeded after a retry`),
  },
  {
    test: /^([\w.]+?)_skipped:circuit_open/,
    label: (m, lang) => (lang === "zh" ? `${m[1]} 熔断中，已跳过` : `${m[1]} skipped (circuit open)`),
  },
  {
    test: /^([\w.]+?)_failed(?::.*)?$/,
    label: (m, lang) => (lang === "zh" ? `${m[1]} 请求失败` : `${m[1]} request failed`),
  },
  {
    test: /^([\w.]+?)_empty(?::(.*))?$/,
    label: (m, lang) => (lang === "zh" ? `${m[1]} 无数据${m[2] ? `（${m[2]}）` : ""}` : `${m[1]} returned no data${m[2] ? ` (${m[2]})` : ""}`),
  },
];

function styleLabel(lang: Lang, style: string): string {
  return STYLES[style]?.[lang] ?? prettify(style);
}

function intentLabel(lang: Lang, intent: string): string {
  return INTENTS[intent]?.[lang] ?? prettify(intent);
}

/** `some_code_name` → "Some code name" (last resort for codes the UI has no label for yet). */
export function prettify(code: string): string {
  const text = code.replace(/[_:]+/g, " ").replace(/\s+/g, " ").trim();
  return text ? text[0]!.toUpperCase() + text.slice(1) : code;
}

/**
 * True for machine codes (`out_of_scope_query`, `budget:step budget of 6 reached`, `llm_error:…`)
 * and false for prose limitations written for people ("检索置信度较低", "Retrieval confidence is low").
 */
export function isCode(text: string): boolean {
  const value = text.trim();
  if (!value || /[㐀-鿿]/.test(value.split(":")[0] ?? "")) return false;
  if (/^[a-z][a-z0-9_.-]*:/.test(value)) return true;
  return /^[a-z][a-z0-9.]*(?:_[a-z0-9.]+)+$/.test(value);
}

/** The display label for a code. Unknown codes are prettified rather than shown raw. */
export function humanizeCode(lang: Lang, code: string, kind?: CodeKind): string {
  const value = code.trim();
  switch (kind) {
    case "route":
      return ROUTES[value]?.[lang] ?? prettify(value);
    case "style":
      return styleLabel(lang, value);
    case "product":
      return PRODUCTS[value]?.[lang] ?? prettify(value);
    case "intent":
      return intentLabel(lang, value);
    case "answerSource":
      return ANSWER_SOURCES[value]?.[lang] ?? prettify(value);
    case "toolError":
      return TOOL_ERRORS[value]?.[lang] ?? prettify(value);
    case "metric":
      return METRICS[value]?.[lang] ?? prettify(value);
    default:
      break;
  }
  const exact = EXACT[value];
  if (exact) return exact[lang];
  for (const rule of RULES) {
    const match = rule.test.exec(value);
    if (match) return rule.label(match, lang);
  }
  return prettify(value);
}

/** Limitations mix prose and codes: humanise only the codes. */
export function limitationText(lang: Lang, text: string): { text: string; code?: string } {
  return isCode(text) ? { text: humanizeCode(lang, text, "limitation"), code: text } : { text };
}

// ---- data-source fallback reasons (mirrors integrations/sources/provenance.py `reason_zh`) ----

const REASONS: [string, Text][] = [
  ["live market data disabled", { zh: "未开启实时行情", en: "live market data is turned off" }],
  ["live macro data disabled", { zh: "未开启实时宏观数据", en: "live macro data is turned off" }],
  ["live market fetch failed", { zh: "实时行情获取失败", en: "the live market fetch failed" }],
  ["live market sources returned no rows", { zh: "实时行情源均无数据", en: "no live market source returned rows" }],
  ["live macro unavailable", { zh: "实时宏观数据不可用", en: "live macro data was unavailable" }],
  ["live source returned no data", { zh: "实时源未返回该数据", en: "the live source returned no data for it" }],
  ["indicator not fetched from live sources", { zh: "该指标未从实时源获取", en: "this indicator is not fetched live" }],
  ["industry board history unavailable", { zh: "行业指数行情不可用", en: "industry board history was unavailable" }],
  ["no live source returned data", { zh: "所有实时源均无数据", en: "no live source returned data" }],
];

const OUTCOMES: Record<string, Text> = {
  circuit_open: { zh: "熔断中", en: "circuit open" },
  timeout: { zh: "超时", en: "timed out" },
  empty: { zh: "无数据", en: "returned nothing" },
  ok: { zh: "成功", en: "ok" },
};

/** "eastmoney.quote:error(ProxyError); eastmoney.quote:circuit_open" → "eastmoney.quote 请求失败；eastmoney.quote 熔断中". */
export function fallbackReasonText(lang: Lang, reason: string): string {
  for (const [prefix, text] of REASONS) if (reason.startsWith(prefix)) return text[lang];
  const parts = reason.split(/;\s*/).map((item) => {
    const cut = item.lastIndexOf(":");
    if (cut <= 0) return item;
    const source = item.slice(0, cut);
    const outcome = item.slice(cut + 1);
    const known = OUTCOMES[outcome];
    const text = known ? known[lang] : outcome.startsWith("error") ? (lang === "zh" ? "请求失败" : "request failed") : outcome;
    return `${source} ${text}`;
  });
  return [...new Set(parts)].join(lang === "zh" ? "；" : "; ");
}
