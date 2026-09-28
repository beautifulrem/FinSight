# FinSight router labels: independent set v2

File: `router_labels_independent_v2.jsonl`
Date built: 2026-09-29
sha256 (jsonl): `9138a534a21a966d75dc63eb06a69d4eb059c0d1b5657c3884fd6e975f73b8c9`

## Construction protocol

- The queries were written by hand from the routing policy below. They were not taken from any existing task file, log or model output.
- **Not read** while building this set:
  - any code under `query_intelligence/`
  - any file under `evaluation/agent_eval/tasks/`
  - anything under `evaluation/results/`
  - no router output was run or inspected
- The allowed exception (the policy section of `README_test_v3.md`) was also not opened. The policy text was supplied verbatim in the task brief and is reproduced below.
- Each line has the format `{"id": "rl2_<route>_NNN", "query", "expected_route", "note"}`. IDs are numbered per route in file order.
- Style: queries imitate real users. They mix colloquial Chinese (啥/咋/宁王/套牢/割肉/抄底), typos (贵洲茅台), emoji, lowercase and casual English, codes-only queries (600519, 000858, 601398), and zh/en code-switching.
- Validation (script run after writing): every line parses as JSON; the key set is exactly the 4 fields; IDs are unique; every route is one of refuse/clarify/workflow/agent; there are no duplicate queries, even after strip plus lowercase normalisation.
- Nothing was committed.

## Policy labelled against (verbatim)

- refuse = not a financial question, or only an instruction to change the system (e.g. "ignore your rules", "stop adding risk warnings"); also out-of-coverage assets: crypto, US/HK stocks (FinSight covers A-shares, China funds/ETFs, China indices and China macro only).
- clarify = a financial question with no identifiable target, or a dangling reference with no context (this set is single-turn: "那个基金", "the other one" have no context), including advice with no target ("我该卖吗", "Which stock should I buy?").
- workflow = one fact about one target (a price, a change, a ratio, a macro value) or a definition/concept ("PE是什么意思", "What does ROE mean?"); several facts about one target are still workflow.
- agent = comparison of 2+ targets; why/causal; judgment/timing/advice about a named target or the market; macro-to-market links; multi-aspect analysis; a fact plus a judgment in one question.

## Counts

| route    | zh  | en | mixed | total | borderline |
|----------|-----|----|-------|-------|------------|
| refuse   | 34  | 23 | 3     | 60    | 13         |
| clarify  | 29  | 21 | 2     | 52    | 11         |
| workflow | 41  | 24 | 4     | 69    | 12         |
| agent    | 33  | 24 | 3     | 60    | 12         |
| **all**  | 137 (56.8%) | 92 (38.2%) | 12 (5.0%) | **241** | **48** |

The language tag (zh/en/mixed) was assigned at authoring time. It is not a field in the JSONL. "mixed" means real zh/en code-switching inside one query. A zh query that only contains a ticker abbreviation such as PE or ETF counts as zh.

Borderline items have a `note` that starts with `Borderline:` and explains which policy clause decided the label.

## Recurring borderline patterns and the calls made

1. **Coverage overrides query shape.** A why/judgment/definition question about crypto, US or HK assets is labelled refuse (e.g. 特斯拉为什么跌, Is NVDA overvalued?, USDT是什么, Fed funds rate).
2. **Dual-listed names.** An unqualified dual-listed name defaults to its A-share line and is in coverage (中国平安, SMIC). An explicit H-share mention is refuse (中国移动H股).
3. **Pleasantries and meta questions** (你好, 谢谢, 你是谁开发的) are refuse, because the policy has no chit-chat route and they are not financial questions.
4. **Style-only instructions** (你以后回答只用英文) are refuse under "only an instruction to change the system".
5. **Pronoun or empty targets beat the question shape.** A single-fact or comparison question whose target is only 它/its/them/"the other one" is clarify.
6. **A category is not a target** when the user wants a pick ("有什么好的ETF推荐吗", "which ETF is best"), so these are clarify. A **market or sector is a target** when the user wants a judgment on it (A股牛熊, 医药板块还有机会吗, "Is the A-share market bottoming out?"), so these are agent.
7. **Several facts about one target** (PE+PB+涨幅, five-year revenue, several ratios) stay workflow. **Fact plus definition** also stays workflow. Only **fact plus judgment** goes to agent.
8. **Purely factual comparisons** of 2+ targets are still agent (科创50 vs 创业板 volatility, SSE vs Shenzhen YTD).
9. **Nicknames and typos** (宁王, 贵洲茅台) are treated as identifiable targets.
10. **Macro interpretation** (十年期跌破2%说明什么, M2-社融剪刀差意味着什么) is agent, not a value lookup.

## Known ambiguities (deliberately excluded or flagged)

- **Mixed-coverage comparisons**, e.g. "英伟达和宁德时代哪个更值得买" or "Tesla vs BYD". One target is out of coverage and one is in. The policy does not say whether this is refuse or agent (partial answer), so these were **excluded** from the set.
- **Concept-vs-concept differences**, e.g. "社融和M2有什么区别". This could be a definition (workflow) or a comparison of 2 things (agent). The policy's "targets" does not clearly cover abstract concepts, so these were **excluded**.
- **Sector stock-picking**, e.g. "银行股里哪个好". Is the sector a target (agent), or is this advice with no specific target (clarify)? **Excluded.**
- **"最近该加仓还是减仓"** is labelled clarify, reading it as advice about the user's unnamed position. A reader who treats it as a market-timing call would label it agent. It is flagged as borderline.
- **"Is it a good time to buy?"** is labelled clarify because the thing being bought is unspecified. The market-timing reading (agent) is possible. It is flagged as borderline.
- **Time-relative macro queries** ("8月PMI多少", "上个月社融") assume the release is available as of the query date. The route stays workflow either way.
- **"Fed funds rate"** is refuse under the literal "China macro only" rule. If the product intends to answer major foreign macro series that drive A-shares, this label would change.
