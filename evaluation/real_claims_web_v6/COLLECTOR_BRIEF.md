# Collector brief (identical for every collector; only the slice differs)

You collect REAL published claims (sentences from web pages) for a blind evaluation set. Today is 2026-10-11.

Allowed reads: /Volumes/Remi/finsight-blind/v6/SCOPE_FOR_COLLECTORS.md, /Volumes/Remi/finsight-blind/v6/COLLECTOR_BRIEF.md,
/Volumes/Remi/finsight-blind/v6/do_not_use_urls.txt. Do NOT open, list or search anything under /Volumes/Remi/FinSight*
or /Volumes/Remi/finsight-review, nor any other file in /Volumes/Remi/finsight-blind/v6/. You are not told any reference
values and must not try to judge whether a claim is correct: never keep or drop a candidate because its number looks
right or wrong.

Tools: load Tavily via ToolSearch (query "select:mcp__tavily__tavily_search,mcp__tavily__tavily_extract"). Use
tavily_search to find pages, then tavily_extract (with a query; retry with extract_depth "advanced" if no article
text) to confirm the exact sentence is on the page. Brave search (mcp__brave-search__brave_web_search) may be used as
a fallback search engine.

What counts: one sentence or clause, as published, that states a checkable fact in the scope (a number, a direction,
a comparison or a definition with a number). Quote whole clauses only; you may cut a clause from a longer sentence
(say so in `note`). Do not paraphrase or translate. Skip URLs listed in do_not_use_urls.txt. At most 2 candidates per
URL. Skip claims whose only period is a quarter, half-year, forecast or target, and price claims whose date cannot be
established from the sentence or its immediate context.

Write one JSON object per line to your output file (UTF-8, ensure_ascii false), fields:
- candidate_id: your slice letter + two digits (e.g. "B07")
- published_text: the exact published sentence/clause
- claim_text: published_text with an implicit subject, period or metric (that the surrounding text names) inserted in
  square brackets, e.g. "[2025年]营收同比增长12%"; identical to published_text if nothing is inserted
- subject_inserted: true/false
- lang: "zh" or "en"; script: "simplified", "traditional" or "latin"
- entities: list of company/index/ETF names the claim is about, first-named first
- period: "FY2025", "2026-09-30" or "2026-04-22"
- period_basis: "sentence" (period in the sentence), "context" (immediately surrounding text), "dateline" (only the
  article date), or "none"
- multi_session: true if the claim is about a multi-day range
- features: list from this fixed vocabulary (all that apply): "definition" (a metric definition stated with a number),
  "weak_direction" (微降/小幅/略/基本持平/大增/大幅/腰斩/edged/slipped/flat/surged etc.), "direction" (增长/下降/上涨/rose/fell
  without a number or with one), "same_company_comparison" (two metrics of one company compared), "cross_company_comparison",
  "kou_fei" (扣非 figure or 扣非 vs 归母), "growth_multiple" (翻倍/N倍/doubled), "short_name" (a short or nick name such as
  宁王, 茅台, 招行, 工行, 紫金, CATL), "approx" (约/近/超/逾/about/over/nearly), "valuation" (市值/PE/PB/破净), "bar" (close,
  change, 成交额, volume), "multi_part" (two or more checkable facts in one claim)
- source_name (outlet or account), source_type: "media", "social", "broker" or "company"; social_platform (slug like
  "xueqiu", "eastmoney_caifuhao", "weibo", "zhihu", "sohu", "163", "x", or null)
- url, published (YYYY-MM-DD or null if not shown), retrieved: "2026-10-11"
- verified_by: "tavily_extract 2026-10-11 (sentence found in extracted page text)" or "tavily_search snippet only"
- is_claim: true
- note: anything a reader needs (where the clause was cut, where the period comes from, if the page is a repost)

Variety wanted across the whole set: definitions stated with numbers, weak direction words, same-company comparisons
(营收增速快于净利润增速), 扣非 vs 归母, growth multiples, direction and comparison clauses, short names, traditional
characters (HK/TW media such as 經濟日報, 信報, 明報, 東網, 香港01, 鉅亨網, 中央社, 雅虎財經 HK), English and Chinese, and
media / social (雪球, 东方财富财富号, 微博, 知乎, 搜狐号, 网易号, 新浪看点) / broker (research-note summaries, 券商公众号) /
company (annual-report announcements, IR pages, press releases) sources. Several claims per page are fine (max 2 per URL).

Final reply: the number of candidates written, counts by source_type, lang and script, and anything you could not verify.
