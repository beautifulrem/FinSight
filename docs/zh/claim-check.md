# 声明核查（Claim Check）

语言：[English](../claim-check.md) | 中文

`POST /agent/claim-check` 对从研报、新闻或社交媒体粘贴的市场说法（例如「贵州茅台ROE超过30%，市盈率不是15倍」「Moutai's ROE is above 30%」）进行逐项数值与关系核查，直接对照行情、基本面与宏观工具数据判定，**完全不调用 LLM**。

核心实现：`query_intelligence/agent/claim_check.py`、`query_intelligence/agent/claim_definitions.py`、`query_intelligence/agent/hearsay.py` 与前端组件 `frontend/src/components/ClaimCheck.tsx`。

```bash
curl -s -X POST http://127.0.0.1:8000/agent/claim-check \
  -H 'Content-Type: application/json' \
  -d '{"claim": "贵州茅台ROE超过30%，市盈率不是15倍"}'
```

## 返回结构

每条核查项（`checks[]`）包含：
- `target`、`metric`、`claimed`、`claimed_high`（区间上限）、`claimed_unit`、`comparator`、`negated`、`direction`（`up|down`）
- 对比/倍数关系：`reference`、`reference_value`、`reference_metric`、`reference_evidence_id`、`ratio`
- 判定与溯源：`actual`、`status`（`supported|contradicted|unverifiable`）、`reason`（无法核查原因）、`evidence_id`、`source`、`as_of`、`as_of_basis`、`note` 及可选 `degraded[]`
- 报告级字段：`verdict`（`supported|contradicted|partially_supported|unverifiable`）、`coverage`（`full|partial|none`）、`unchecked[]`（不含数字/比较的未核查子句）、`excluded[]`（群组排除项）、`limitations[]`、`degraded[]`、`disclaimer`

没有任何子句会被静默丢弃：无法抽取数值或比较的子句都会显式进入 `unchecked`（`reason: no_claim`）。

## 解析与判定规则

### 数值、繁简/地区术语规范化与标的绑定
- 自动规范化全角数字/字母、繁体字及港澳台财经术语（`純利` → `净利润`、`本益比` → `市盈率`、`市賬率`/`股價淨值比` → `市净率`、`股東應佔溢利` → `归母净利润`、`按年` → `同比` 等，定义于 `query_intelligence/traditional.py`）。
- 解析阿拉伯数字、中文数字与财务大写（`二十四点六倍`、`三成`、`百分之三十`、`一千七百亿`）、阈值动词后的裸「万亿」（`突破万亿`）、区间（`20到30倍`、`between 20 and 30`）及英文分数（`half a percent`、`twice`、`a quarter of`）。
- 自动排除日期、窗口期（`近20日`）、指标参数（`RSI(14)`）、证券代码与指数名，不误判为声称数值。
- 就近绑定前置标的，支持 `分别` / `respectively` 顺序配对、常用简称与群组排除句式（`除平安银行外…均同比增长`）。

### 指标、会计口径与宏观指标
- 按单位兼容性绑定最近指标（`倍`/`x` → PE/PB；`%` → ROE、利润率、涨跌幅或同比增速；`元` → 价格；`亿`/`万`/`billion` → 营收/利润/成交额/市值；`点` → 指数点位）；单位不兼容返回 `unit_mismatch`，无单位金额返回 `no_unit`。
- 严格区分 `扣非净利润` 与 `归母净利润`、`流通市值` 与总市值、`营业收入` 与 `营业总收入`、加权与期末 ROE、基本与稀释 EPS、TTM 与动态 PE；口径不一致显式标记 `definition_mismatch`。
- 内联核验指标公式定义（`kind: "definition"`，`evidence_id: glossary_<term>`），包括「破净 / 市净率小于 1」。
- 自动识别 `CPI`、`PPI`、`PMI`、`M2`、10 年期国债收益率、1Y/5Y LPR 与 `GDP`，对照 `get_macro_indicators` 核查（`as_of_basis: indicator_date`）。

### 比较词与否定

| `comparator` | 示例（中 / 英） | 判定为 `supported` 的条件 |
|---|---|---|
| `eq` | `为`、`只有`、`is` | 末位数字半步长或 2% 容差内 |
| `approx` | `约`、`左右`、`about`、`around` | 5% 或末位有效数字半步长内（取较大者） |
| `approx`（下方接近） | `接近`、`将近`、`nearly`、`almost` | `0.9N ≤ actual ≤ N`（加末位舍入半步长） |
| `gt` / `ge` | `超过`、`高于`、`逾`、`站上`、`至少`、`以上` / `above`、`at least` | `actual > claimed` / `actual ≥ claimed`（`30多倍`、`出头` 附带上界步长） |
| `lt` / `le` | `低于`、`不到`、`不足`、`跌破`、`不超过`、`以下` / `below`、`at most` | `actual < claimed` / `actual ≤ claimed` |
| `range` | `X到Y之间`、`X-Y倍`、`between X and Y` | `X ≤ actual ≤ Y` |
| `ne` | 否定 `eq`（`不是`、`并非`、`not`） | 超出 `eq` 容差范围 |

否定词（`不是`、`没有`、`未`、`不`、`not`、`never`）作用于前一数字或子句起至当前数字，翻转比较方向（`eq↔ne`、`gt↔le`、`ge↔lt`）。

### 涨跌幅与弱方向词区间约定

- **单日涨跌幅**：涨跌词后的边界指**该方向上的变动幅度**（`跌超1%` 要求 `change ≤ -1%`；`跌不到1%` 要求 `-1% < change ≤ 0%`；方向相反直接判定 `contradicted`）。定性单日涨跌阈值：`大跌`/`暴跌`/`plunged`、`大涨`/`surged` 为 `|move| ≥ 3%`；`微跌`/`小跌`/`dipped`、`微涨`/`edged up` 为 `0 < |move| < 1%`。
- **同比增速弱方向词区间**（无数字时生效，`g` 为基本面同比增速 `%`；否定翻转区间）：

| 档位 | 词汇（中；英） | 成立条件 |
|---|---|---|
| `flat` | 基本持平、大致持平、持平、变化不大、几乎没有变化；flat, roughly/broadly/essentially/largely flat, little changed, unchanged | `−2 ≤ g ≤ 2` |
| `slight` | 微增、微升、微降、略增、略降、略有/稍有 增长/下降；edged/inched/ticked up/down, rose/fell slightly, marginally up/down | 同向且 `0 < \|g\| ≤ 5` |
| `modest` | 小幅 增长/增加/上升/下降/下滑/减少；rose/grew/fell/declined modestly, modestly higher/lower | 同向且 `0 < \|g\| ≤ 10` |
| `sharp` | 大增、暴增、猛增、激增、大幅 增长/下降、骤降、锐减；surged, soared, plunged, slumped, rose/fell sharply | 同向且 `\|g\| ≥ 20` |

### 关系比较、多公司求和与极值
支持双标的比较（`比…高/低`）、倍数（`是…的1.5倍`、`X倍半`）、显式差值（`高出N个百分点`）、同公司双增速比较（`营收增速快于净利润增速`、`降幅较营收收窄`）、多公司求和（`茅台和五粮液成交额加起来不到50亿`、`A与B净利润合计不足C`）、52 周高低点参照、行业均值比较与多标的极值判定。

### 截止日期、缺失更正层降级与不可核查原因
- `as_of_basis` 明确标注 `trade_date`（行情）、`valuation_date`（估值日）、`report_date`（报告期）或 `indicator_date`（宏观周期）。
- 若启用 `QI_OFFLINE_SNAPSHOT_FIX` 但缺失 `data/snapshot/structured_data_fix.json`，依赖未更正 v1 种子的核查项会标注 `degraded: ["data_layer_missing:corrections"]`、报告 `coverage: "partial"` 并在 `limitations` 与 `disclaimer` 中显式提示。
- `reason` 枚举：`no_target`、`no_metric`、`no_data`、`growth_unavailable`、`unit_mismatch`、`no_unit`、`forecast`、`period_mismatch`、`multi_day`、`no_reference`、`definition_mismatch`、`currency`。

### 对话内联传闻核查
当对话提问包含核实/传闻线索（`听说…是真的吗`、`网传…`、`I read that … True?`、`核实`）时，`agent/hearsay.py` 自动调用 `check_claim`，以确定性核查结论开头并随响应附带 `fact_check` 对象。

## 基准与验证

基准集位于 `evaluation/claim_bench/`（开发集 `claims_v1.jsonl`、保留集 `claims_v1_holdout.jsonl`）及盲测真实公开说法集 `evaluation/real_claims_web*/`。

```bash
python -m evaluation.claim_bench.run --set dev
python -m evaluation.claim_bench.run --set holdout
python -m pytest -q tests/test_claim_bench.py tests/test_agent_round17_claims.py tests/test_agent_round18_claims.py
```

当前离线得分：开发集（311 条）**1.000**、保留集（47 条 / 54 项）**1.000**；盲收集真实说法集首次未暴露得分：v2 **0.500**、v3 **0.540**、v4 **0.740**、v5 **0.680**、v6 **0.650**（详见 [evaluation.md](evaluation.md) 与 [evaluation-history.md](evaluation-history.md)）。

## 已知局限

- 规则抽取无法覆盖词类之外的新句式（如英文排除句 `"all but X"`、跨句排除或非公式化定义）。
- 离线快照缺少历史比率同比（如毛利率同比变化）、多日累计区间与扩展行业均值卡片，相关核查会如实返回 `unverifiable`。
- 涉及外币或境外上市标的（港美股/GDR/ADR）的声明返回 `unverifiable`（`currency` / `no_data`）。
