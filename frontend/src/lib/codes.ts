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
  amount: { zh: "成交额", en: "Turnover" },
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
  cpi_yoy: { zh: "CPI 同比", en: "CPI (YoY)" },
  ppi_yoy: { zh: "PPI 同比", en: "PPI (YoY)" },
  pmi: { zh: "制造业 PMI", en: "Manufacturing PMI" },
  m2_yoy: { zh: "M2 同比", en: "M2 (YoY)" },
  cn10y: { zh: "10 年期国债收益率", en: "10Y government bond yield" },
  lpr_1y: { zh: "1 年期 LPR", en: "1-year LPR" },
  lpr_5y: { zh: "5 年期以上 LPR", en: "5-year LPR" },
  gdp_yoy: { zh: "GDP 同比", en: "GDP (YoY)" },
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
  // Router and graph route reasons (round 12, H14): every exact code `guard_in` can emit has a label.
  request_without_object: { zh: "只有请求，没有说明对象", en: "A request without an object" },
  "no_target:recommendation": { zh: "要求推荐，但没有指明范围", en: "Asked for picks without naming a target" },
  "no_target:advice": { zh: "判断或择时问题，但没有指明证券", en: "Judgment or timing question without a security" },
  "lexical:forecast": { zh: "含预测用语", en: "Forecast wording" },
  "lexical:analysis_request": { zh: "请求分析", en: "Asks for an analysis" },
  "concept:definition": { zh: "概念或公式问题", en: "Concept or formula question" },
  "override:why_style_without_causal_cue": {
    zh: "没有因果用语，按事实查询处理",
    en: "No causal wording: treated as a fact lookup",
  },
  "override:out_of_scope_sector_of_discussed_target": {
    zh: "问的是正在讨论的标的所属行业，按金融问题处理",
    en: "Sector of the security under discussion: treated as a finance question",
  },
  session_memory_over_nlu_context_carry: { zh: "按会话记忆确定标的", en: "Target taken from the session's memory" },
  holding_value: { zh: "计算持仓市值", en: "Holding value calculation" },
  system_change_request: { zh: "要求更改系统设定（已拒绝）", en: "Asked to change the system setup (refused)" },
  ellipsis_without_antecedent: {
    zh: "省略式提问，但没有上文可以补全",
    en: "Shortened question with no earlier turn to complete it",
  },
  difference_without_comparison: {
    zh: "问差多少，但没有可比较的对象或指标",
    en: "Asked for a gap with no comparison to compute it from",
  },
  group_reference_incomplete: {
    zh: "指代的对象比上文讨论的多，需要澄清",
    en: "Referred to more targets than were discussed; asked to clarify",
  },
  "input_guard:prediction_without_target": {
    zh: "移除注入指令后，只剩没有标的的预测请求",
    en: "Without the injected text, only a prediction request with no target was left",
  },
  "model_policy:composition_for_slow_model": {
    zh: "慢速推理模型：改用固定流程，由 LLM 撰写回答",
    en: "Slow reasoning model: used the workflow with an LLM-written answer",
  },
  "frame:style_compare": { zh: "按对比计算处理", en: "Treated as a computed comparison" },
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
  removed_prohibited_promotion: {
    zh: "已删除收益保证、荐股或联系方式内容",
    en: "Removed return guarantees, stock tips or contact details",
  },
  omitted_document_promotion: {
    zh: "已省略文档中的推广或联系方式内容",
    en: "Omitted promotional content or contact details from a document",
  },
  omitted_document_trading_call: { zh: "已省略文档中的买卖建议", en: "Omitted a buy/sell call from a document" },
  attributed_document_claim: {
    zh: "已将仅有文档来源的说法标注为未经证实",
    en: "Marked a document-only claim as unconfirmed",
  },
  omitted_conflicting_document_figure: {
    zh: "已省略与结构化数据不一致的文档数值",
    en: "Omitted a document figure that contradicts the structured data",
  },
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
  [
    // "2024年的呢" after "茅台的营收": the target and the aspect both come from the last turn (memory.py)
    /^ellipsis:target->(.+?)\+aspect->(.+)$/,
    (m) => ({
      zh: `沿用上一轮的标的 ${m[1]} 和问题：${aspects(m[2]!, "、")}`,
      en: `Kept the security (${m[1]}) and the question (${aspects(m[2]!, ", ")}) from the last turn`,
    }),
  ],
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

/** Joins names: "A、B" in Chinese, "A and B" / "A, B and C" in English. */
function joinNames(lang: Lang, items: string[]): string {
  if (lang === "zh") return items.join("、");
  if (items.length <= 2) return items.join(" and ");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** A frame operand: a security name, "保险行业平均" (an industry average) or "所属行业平均" (its own industry's). */
function operandName(lang: Lang, operand: string): string {
  if (lang === "zh") return operand;
  if (operand === "所属行业平均") return "its industry average";
  const industry = /^(.+)行业平均$/.exec(operand);
  return industry ? `${industry[1]} industry average` : operand;
}

/** The comparison frame's metric keys (agent/frame.py `FRAME_METRICS`), short enough for a chip. */
const FRAME_METRICS: Record<string, Text> = {
  net_margin: { zh: "净利率", en: "net margin" },
  gross_margin: { zh: "毛利率", en: "gross margin" },
  roe: { zh: "ROE", en: "ROE" },
  pe: { zh: "市盈率", en: "P/E" },
  pb: { zh: "市净率", en: "P/B" },
  eps: { zh: "每股收益", en: "EPS" },
  dividend_yield: { zh: "股息率", en: "dividend yield" },
  market_cap: { zh: "总市值", en: "market cap" },
  revenue: { zh: "营业收入", en: "revenue" },
  net_profit: { zh: "净利润", en: "net profit" },
  pct_change: { zh: "涨跌幅", en: "price change" },
  amount: { zh: "成交额", en: "turnover" },
  volume: { zh: "成交量", en: "volume" },
  close: { zh: "收盘价", en: "close" },
};

function frameMetric(lang: Lang, key: string): string {
  return FRAME_METRICS[key]?.[lang] ?? METRICS[key]?.[lang] ?? prettify(key);
}

/** `frame:{operation}:{metric}:{A|B…}` (agent/frame.py `resolve_frame_question`): the computed comparison. */
function frameLabel(lang: Lang, operation: string, metric: string, operands: string): string | null {
  const names = operands.split("|").filter(Boolean).map((name) => operandName(lang, name));
  const what = frameMetric(lang, metric);
  const [a = "", b = ""] = names;
  switch (operation) {
    case "difference":
      return lang === "zh" ? `计算${what}差值：${a} 对比 ${b}` : `Computed the ${what} gap: ${a} vs ${b}`;
    case "ratio":
      return lang === "zh" ? `计算${what}比值：${a} ÷ ${b}` : `Computed the ${what} ratio: ${a} ÷ ${b}`;
    case "relative":
      return lang === "zh" ? `计算${what}相对差：${a} 相对 ${b}` : `Computed the relative ${what} gap: ${a} vs ${b}`;
    case "which":
      return lang === "zh"
        ? `比较${what}高低：${joinNames(lang, names)}`
        : `Compared which ${what} is higher: ${joinNames(lang, names)}`;
    default:
      return null;
  }
}

const OFF_TOPIC: Record<string, Text> = {
  coding: { zh: "编程", en: "coding" },
  translation: { zh: "翻译", en: "translation" },
  travel: { zh: "旅行", en: "travel" },
  weather: { zh: "天气", en: "weather" },
  writing: { zh: "写作", en: "writing" },
  entertainment: { zh: "娱乐", en: "entertainment" },
};

const LANGUAGES: Record<string, Text> = { zh: { zh: "中文", en: "Chinese" }, en: { zh: "英文", en: "English" } };

/** Session references and frame / alias reasons (agent/memory.py, agent/graph.py, agent/router.py). */
const SESSION: [RegExp, (m: RegExpExecArray, lang: Lang) => string][] = [
  [
    /^frame:dropped_non_operand:(.+)$/,
    (m, lang) => (lang === "zh" ? `忽略了不在比较对象中的 ${m[1]}` : `Dropped ${m[1]}: not one of the compared targets`),
  ],
  [
    /^frame:([a-z_]+):([a-z_0-9]+):(.+)$/,
    (m, lang) => frameLabel(lang, m[1]!, m[2]!, m[3]!) ?? formatUnknown(lang, m.input),
  ],
  [
    /^group_reference_count_mismatch:(.+?)->(.+)$/,
    (m, lang) =>
      lang === "zh" ? `“${m[1]}”指三个对象，但上文只讨论了 ${m[2]}` : `“${m[1]}” means three, but only ${m[2]} were discussed`,
  ],
  [
    /^group_reference:which->(.+)$/,
    (m, lang) => (lang === "zh" ? `“哪家”指上文的 ${m[1]}` : `“Which” refers to ${m[1]} from earlier turns`),
  ],
  [/^group_reference:(.+?)->(.+)$/, (m, lang) => (lang === "zh" ? `将“${m[1]}”理解为 ${m[2]}` : `Read “${m[1]}” as ${m[2]}`)],
  [
    /^industry_reference:(.+?)->(.+)$/,
    (m, lang) => (lang === "zh" ? `将“${m[1]}”理解为${m[2]}行业` : `Read “${m[1]}” as the ${m[2]} industry`),
  ],
  [
    /^(difference|comparison)_follow_up:(.+?)(?:\+aspect->(.+))?$/,
    (m, lang) => {
      const gap = m[1] === "difference";
      const aspect = m[3] ? aspects(m[3], lang === "zh" ? "、" : ", ") : "";
      if (lang === "zh") return `${gap ? "接上一轮的比较计算差值" : "接上一轮的比较"}：${m[2]}${aspect ? `（${aspect}）` : ""}`;
      return `${gap ? "Gap follow-up on" : "Follow-up on"} the last comparison: ${m[2]}${aspect ? ` (${aspect})` : ""}`;
    },
  ],
  [
    /^comparison_anchor:\+(.+)$/,
    (m, lang) => (lang === "zh" ? `比较时加入上文的 ${m[1]}` : `Added ${m[1]} from earlier turns to the comparison`),
  ],
  [
    /^session_inherit:target->(.+)$/,
    (m, lang) => (lang === "zh" ? `沿用会话中的标的 ${m[1]}` : `Kept the conversation's security: ${m[1]}`),
  ],
  [
    /^session_inherit:macro->(.+)$/,
    (m, lang) => (lang === "zh" ? `沿用会话中的宏观主题 ${m[1]}` : `Kept the conversation's macro topic: ${m[1]}`),
  ],
  [
    /^session_disambiguation:(.+?)->(.+)$/,
    (m, lang) =>
      lang === "zh" ? `按正在讨论的标的，将“${m[1]}”理解为 ${m[2]}` : `Read “${m[1]}” as ${m[2]}, the security under discussion`,
  ],
  [
    /^session_language:(zh|en)$/,
    (m, lang) =>
      lang === "zh" ? `沿用会话指定的回答语言：${LANGUAGES[m[1]!]!.zh}` : `Kept the session's answer language: ${LANGUAGES[m[1]!]!.en}`,
  ],
  [
    /^alias_context:(.+?)->(.+)$/,
    (m, lang) => (lang === "zh" ? `按问题中的行业用语，将“${m[1]}”理解为 ${m[2]}` : `Read “${m[1]}” as ${m[2]} from the industry wording`),
  ],
  [
    /^alias_default:(.+?)->([^|]+)(?:\|(.+))?$/,
    (m, lang) => {
      const others = (m[3] ?? "").split("/").filter(Boolean);
      if (lang === "zh") return `“${m[1]}”默认理解为 ${m[2]}${others.length ? `（也可能指 ${others.join("、")}）` : ""}`;
      return `Read “${m[1]}” as ${m[2]} by default${others.length ? ` (could also be ${joinNames(lang, others)})` : ""}`;
    },
  ],
  [
    /^sector_member:(.+?)->(.+)$/,
    (m, lang) => (lang === "zh" ? `${m[1]}行业的问题，保留正在讨论的 ${m[2]}` : `Sector question (${m[1]}): kept ${m[2]} in scope`),
  ],
  [
    /^off_topic_request:(.+)$/,
    (m, lang) => {
      const kind = OFF_TOPIC[m[1]!]?.[lang] ?? prettify(m[1]!).toLowerCase();
      return lang === "zh" ? `不是投研问题（${kind}），已拒答` : `Not a research question (${kind}); refused`;
    },
  ],
  [/^concept:glossary:(.+)$/, (m, lang) => (lang === "zh" ? `术语解释：${m[1]}` : `Glossary term: ${m[1]}`)],
  [
    /^override:out_of_scope_glossary_concept:(.+)$/,
    (m, lang) => (lang === "zh" ? `金融术语“${m[1]}”，按金融问题处理` : `Finance term “${m[1]}”: treated as a finance question`),
  ],
  [/^dropped_generic_noun:(.+)$/, (m, lang) => (lang === "zh" ? `忽略了泛称“${m[1]}”` : `Ignored the generic word “${m[1]}”`)],
  [
    /^dropped_advice_phrase:(.+)$/,
    (m, lang) => (lang === "zh" ? `“${m[1]}”是建议用语，不是公司名` : `“${m[1]}” is advice wording, not a company name`),
  ],
  [
    /^dropped_unnamed_target_out_of_coverage:(.+)$/,
    (m, lang) =>
      lang === "zh"
        ? `忽略了猜测的标的 ${m[1]}（问题涉及覆盖范围外的资产）`
        : `Dropped the guessed target ${m[1]} (the question is about an asset outside coverage)`,
  ],
  [
    /^foreign_listing_lookalike:(.+)$/,
    (m, lang) =>
      lang === "zh" ? `“${m[1]}”只出现在港股或美股公司名中，已忽略` : `${m[1]} only appeared inside a Hong Kong or US company name; ignored`,
  ],
];

/**
 * Last resort for a code the UI has no label for yet: readable, never the raw code. "frame:new_op:roe:A|B" →
 * "Frame: new op · roe · A、B"; "x:a->b" → "X: a → b".
 */
export function formatUnknown(lang: Lang, code: string): string {
  const list = lang === "zh" ? "、" : ", ";
  const parts = code
    .split(":")
    .map((part) =>
      part
        .replace(/->/g, " → ")
        .replace(/\|/g, list)
        .replace(/_+/g, " ")
        .replace(/\s+/g, " ")
        .trim(),
    )
    .filter(Boolean);
  if (!parts.length) return code;
  const [head, ...rest] = parts;
  const title = head![0]!.toUpperCase() + head!.slice(1);
  return rest.length ? `${title}: ${rest.join(" · ")}` : title;
}

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

/**
 * The English UI shows the security names a code carries ("ellipsis:target->五粮液和贵州茅台") in English, from the
 * names the server sent for the turn (`nlu_summary.entities[].name_en`); Chinese joiners become "and" / "," (round 9).
 */
export function localizeNames(lang: Lang, text: string, names?: ReadonlyMap<string, string>): string {
  if (lang !== "en" || !names?.size || !/[\u4e00-\u9fff]/.test(text)) return text;
  let out = text;
  for (const [zh, en] of [...names].sort((a, b) => b[0].length - a[0].length)) out = out.split(zh).join(en);
  return out.replace(/\s*和\s*/g, " and ").replace(/\s*、\s*/g, ", ");
}

/** Chinese name -> English name from a turn's entities (`nlu_summary.entities`), for `localizeNames`. */
export function entityNames(
  entities: readonly { name?: string | null; name_en?: string | null }[] | null | undefined,
): Map<string, string> {
  return new Map((entities ?? []).flatMap((entity) => (entity.name && entity.name_en ? [[entity.name, entity.name_en] as [string, string]] : [])));
}

/** The display label for a code. Unknown codes are formatted readably rather than shown raw. */
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
  for (const [pattern, label] of SESSION) {
    const match = pattern.exec(value);
    if (match) return label(match, lang);
  }
  for (const rule of RULES) {
    const match = rule.test.exec(value);
    if (match) return rule.label(match, lang);
  }
  return value.includes(":") ? formatUnknown(lang, value) : prettify(value);
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
