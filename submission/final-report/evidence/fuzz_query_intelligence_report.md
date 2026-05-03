# Query Intelligence Fuzz Test Report

## Summary

- Generated at: `2026-04-22T18:33:05.340401+00:00`
- Total cases: `32`
- Passed cases: `32`
- Failed cases: `0`
- Overall score: `10.0/10`
- NLU score: `10.0/10`
- Retrieval score: `10.0/10`
- Schema score: `10.0/10`

## Defect Summary

## Recommendations

- 当前 fuzz 矩阵未暴露明显缺陷，可以继续扩口语、错别字和多轮对话样本。

## Case Results

### canonical_stock_advice [passed]
- Query: `茅台今天为什么跌？后面还值得拿吗？`
- Category: `baseline_stock`
- Passed checks: `16/16`

### colloquial_stock_followup [passed]
- Query: `茅台今天为啥跌, 后面还能拿吗`
- Category: `colloquial_paraphrase_gap`
- Passed checks: `11/11`

### recent_week_stock_why [passed]
- Query: `贵州茅台最近一周为啥跌`
- Category: `time_scope_why_gap`
- Passed checks: `11/11`

### generic_bucket_index [passed]
- Query: `宽基指数最近能买吗`
- Category: `generic_bucket`
- Passed checks: `11/11`

### generic_growth_index [passed]
- Query: `成长指数后面还值得拿吗`
- Category: `generic_bucket`
- Passed checks: `10/10`

### sector_bucket_query [passed]
- Query: `白酒板块最近还能买吗`
- Category: `generic_bucket`
- Passed checks: `11/11`

### etf_alias_fee_query [passed]
- Query: `300ETF费率高不高`
- Category: `alias_normalization`
- Passed checks: `11/11`

### numeric_symbol_comparison_clarification [passed]
- Query: `510300和159915哪个好`
- Category: `clarification_required`
- Passed checks: `10/10`

### resolved_stock_comparison [passed]
- Query: `贵州茅台和五粮液哪个好`
- Category: `comparison`
- Passed checks: `12/12`

### self_comparison_alias [passed]
- Query: `沪深300etf和300ETF有什么不同`
- Category: `same_entity_compare`
- Passed checks: `10/10`

### typo_stock_name [passed]
- Query: `茅苔今天为什么跌`
- Category: `typo_tolerance`
- Passed checks: `10/10`

### followup_with_dialog_context [passed]
- Query: `后面还值得拿吗`
- Category: `context_carryover`
- Passed checks: `12/12`

### followup_with_user_profile [passed]
- Query: `后面还值得拿吗`
- Category: `context_carryover`
- Passed checks: `12/12`

### colloquial_stock_advice_variant [passed]
- Query: `茅台咋跌了，后面还能不能拿`
- Category: `colloquial_paraphrase_gap`
- Passed checks: `13/13`

### monthly_stock_why [passed]
- Query: `贵州茅台近一个月为什么跌`
- Category: `time_scope_why_gap`
- Passed checks: `13/13`

### stock_buy_timing_colloquial [passed]
- Query: `五粮液现在这个位置能上车吗`
- Category: `buy_sell_timing`
- Passed checks: `12/12`

### stock_risk_colloquial [passed]
- Query: `中国平安风险大不大`
- Category: `risk_analysis`
- Passed checks: `11/11`

### stock_fundamental_colloquial [passed]
- Query: `中国平安基本面咋样`
- Category: `fundamental_analysis`
- Passed checks: `11/11`

### lowercase_etf_comparison [passed]
- Query: `300etf和创业板etf哪个好`
- Category: `comparison`
- Passed checks: `12/12`

### generic_etf_lof_compare [passed]
- Query: `ETF和LOF有什么区别`
- Category: `generic_product_compare`
- Passed checks: `11/11`

### fund_etf_price_difference [passed]
- Query: `基金净值和ETF价格有啥区别`
- Category: `generic_product_compare`
- Passed checks: `8/8`

### macro_policy_industry [passed]
- Query: `宏观政策会影响白酒吗`
- Category: `macro_policy`
- Passed checks: `11/11`

### stock_news_query [passed]
- Query: `最近有哪些茅台新闻`
- Category: `event_news`
- Passed checks: `10/10`

### stock_announcement_query [passed]
- Query: `贵州茅台最近有什么公告`
- Category: `event_news`
- Passed checks: `11/11`

### generic_buy_clarification [passed]
- Query: `这个能买吗`
- Category: `clarification_required`
- Passed checks: `10/10`

### generic_hold_clarification [passed]
- Query: `后面还值得拿吗`
- Category: `clarification_required`
- Passed checks: `10/10`

### english_stock_advice [passed]
- Query: `Why did Moutai fall today? Is it still worth holding?`
- Category: `english`
- Passed checks: `14/14`

### english_announcement_query [passed]
- Query: `Any recent announcements from Kweichow Moutai?`
- Category: `english`
- Passed checks: `11/11`

### english_etf_comparison [passed]
- Query: `Which is better, CSI 300 ETF or ChiNext ETF?`
- Category: `english`
- Passed checks: `12/12`

### english_generic_product_compare [passed]
- Query: `What is the difference between ETF and LOF?`
- Category: `english`
- Passed checks: `11/11`

### english_macro_policy_industry [passed]
- Query: `Does macro policy affect liquor?`
- Category: `english`
- Passed checks: `11/11`

### english_bucket_sector_advice [passed]
- Query: `Is the liquor sector still buyable?`
- Category: `english`
- Passed checks: `12/12`
