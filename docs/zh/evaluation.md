# 评测（中文摘要）

语言：[English（完整版）](../agent-eval.md) | 中文

本页是 [docs/agent-eval.md](../agent-eval.md) 的中文摘要。完整页面由 `python -m evaluation.agent_eval.report` 从 [`evaluation/results/`](../../evaluation/results/) 里已提交的文件生成，CI 用 `report --check` 保证页面和数据一致。每个结果文件都记录了运行的 commit、Prompt 版本（`id@version#sha`）、时间和命令。下面每个数字都注明了出自哪个文件、哪个 commit。

## 测什么

| 内容 | 位置 |
|---|---|
| **开发集**：最初 207 个任务 / 220 轮，11 个类别，中英文，用来驱动修复。第二轮补了 9 个省略追问任务；第三轮为第二轮评审发现的问题（悬空「为什么」、复数指代、缺失期间/指标、加密资产与美股、注入改语言）补了 22 个回归任务，现为 238 个任务。 | `evaluation/agent_eval/tasks/agent_eval_v1.jsonl`（`build_tasks.py`） |
| **保留集**：53 个任务 / 56 轮，在开发集驱动修复之后才编写。没有用来调规则，但用来选过 Prompt，所以是验证集。 | `agent_eval_holdout_v1.jsonl`（`build_holdout.py`） |
| **测试集 v2**：121 个任务 / 174 轮，12 个类别，77 个中文、44 个英文任务，其中 24 段 3–5 轮的对话。下面的在线运行发生时它未被触碰；第二轮的追问修复参考了它暴露的失败类别，所以此后它也只算验证集。 | `agent_eval_test_v2.jsonl`（`build_test_v2.py`） |
| 某一时刻的工具快照，回放使用，结果不依赖实时数据源 | `evaluation/agent_eval/fixtures/snapshot_*.json` |
| 评分、bootstrap 置信区间、配对比较 | `evaluation/agent_eval/metrics.py` |
| 回答路径消融 | `evaluation/agent_eval/ablation.py` |
| CI 门禁（与已提交基线比较） | `evaluation/agent_eval/gate.py` |
| 故障注入、校验器压力测试、提示注入红队、路由评测 | `fault_injection.py`、`verifier_stress.py`、`redteam.py`、`router_eval.py` |
| 用户反馈转候选任务 | `scripts/feedback_to_tasks.py` |

- **期望值怎么来**：期望的事实来自随仓库提供的离线快照（`data/structured_data.json`，行情截至 2026-04-22），每个必需的值都能核对。
- **题目去重**：题目专为本评测编写，并与仓库里用于训练或其他评测的问题做过重叠检查；测试集 v2 还检查了近似重复，这项检查由测试固定下来（`tests/test_agent_eval.py`）。

**一票否决的评分。** 一轮只有同时满足以下条件才算成功：

- 行为（回答/澄清/拒答）正确；
- 每个必需事实都陈述**且**用正确的 evidence id 引用；
- 用了必需的工具；
- 没有交易指令；
- 任务要求时，有限定语、如实说明缺数据、在追问中正确沿用实体。

多轮任务要每一轮都成功才算成功。`pass^k` 只有 k 次重复全部成功才算通过；只跑一次的路径（`legacy`、`workflow`、`legacy_llm`）报告的是 pass^1。

## 统计方法

- **置信区间**：任务成功率和 pass^k 写成 `值 [下限, 上限]`，用按任务的百分位 bootstrap（2000 次重采样，固定种子 20260926），同一任务的多次重复一起抽样。
  - **为什么以任务为单位**：以任务而不是轮次或运行为单位，因为同一任务的重复运行是相关的。
  - **区间有多宽**：保留集 53 个任务，一个任务就是 1.9 个百分点，0.9 附近的区间宽约 ±6–9 个百分点；测试集 121 个任务，一个任务是 0.8 个百分点。
- **两条路径的比较**：在同一批任务上做配对 bootstrap（两条路径用同样的重采样），另对 pass^k 结果做精确 McNemar 检验（只有一条路径全过、另一条没全过的任务才有信息量）。只有任务成功率差值的 95% 区间不含 0 时才称为显著；McNemar 忽略部分通过，是更严格的检验。
- **不覆盖什么**：bootstrap 把任务集当作所有可能问题的一个样本，不覆盖换一份数据快照、换一天的网关状态，或各任务集作者自身的写法风格。

## 比较的回答路径

| 路径 | 是什么 | 需要 LLM |
|---|---|---|
| `legacy` | 原 `/chat` 路径（Query Intelligence 流水线 + 模板答案，即没有 LLM Key 时的行为）。它的拒答/澄清行为从 NLU 标志推断，对它是宽松的。 | 否 |
| `workflow` | Agent 的确定性路径：经典 NLU 守卫与路由、确定性规划器、工具、模板组答、证据校验、合规节点。 | 否 |
| `pure_llm` | LLM 直接回答，不调工具。 | 是 |
| `legacy_llm` | 原 `/chat` 路径，由 LLM 改写证据。每个任务只跑一次。 | 是 |
| `workflow_llm` | Agent 确定性路径，由 LLM 根据工具证据组织答案。 | 是 |
| `agent` | Agent 在 LangGraph 循环里由 LLM 选择工具。 | 是 |

## 主要结果

任务成功率（方括号为 95% 置信区间）：

| 路径 | 开发集 · DeepSeek | 保留集 · DeepSeek | 测试集 v2 · DeepSeek | 保留集 · GLM | 测试集 v2 · GLM |
|---|---|---|---|---|---|
| legacy | 0.256 [0.20, 0.32] | 0.189 [0.09, 0.30] | 0.223 [0.15, 0.30] | 0.207 [0.11, 0.32] | 0.223 [0.15, 0.30] |
| legacy_llm（pass^1） | 0.440 [0.37, 0.51] | 0.679 [0.55, 0.79] | 未运行 | 未运行 | 未运行 |
| pure_llm | 0.000 | 0.000 | 0.000 | 未运行 | 未运行 |
| workflow | 0.981 [0.96, 1.00] | 0.849 [0.75, 0.94] | 0.727 [0.64, 0.80] | 0.849 [0.75, 0.94] | 0.727 [0.64, 0.80] |
| workflow_llm | 0.979 [0.96, 1.00] | 0.906 [0.83, 0.98] | 0.766 [0.69, 0.83] | 0.906 [0.83, 0.98] | 0.760 [0.68, 0.83] |
| agent | 0.986 [0.97, 1.00] | 0.956 [0.91, 0.99] | 0.804 [0.73, 0.86] | 0.981 [0.94, 1.00] | 0.810 [0.74, 0.87] |

来源与运行命令：

- **DeepSeek V4.1 Flash**：
  - 开发集和保留集：`ablation-final.json`，commit `846bc5e`，命令 `python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 6`；
  - 测试集 v2：`ablation-test_v2-deepseek.json`，commit `38a3069`。
- **GLM-5.3 Flash**：保留集和测试集 v2 来自 `ablation-glm.json`，commit `f7bf624`；GLM 开发集以 `ablation-glm-dev.json`（`38a3069`）为准。
- **运行方式**：在线路径每个任务重复 3 次，成本为网关账单口径（`usage.cost`，美元）。

<!-- final2: update after final online run (held-out and test v2, DeepSeek and GLM, at the round-2 commit) -->

其他指标（DeepSeek，开发集，`846bc5e`）：

| 指标 | workflow | workflow_llm | agent |
|---|---|---|---|
| pass^3 | –（跑一次） | 0.976 [0.95, 1.00] | 0.971 [0.95, 0.99] |
| 必需事实陈述且引用 | 1.000 | 0.998 | 0.992 |
| 工具精确率 | 0.739 | 0.739 | 0.659 |
| 草稿首次即通过校验 | – | 0.812 | 0.486 |
| 每轮 LLM 调用 | 0 | 1.006 | 1.9 |
| 每轮 token | 0 | 2050.8 | 8377.7 |
| Prompt 缓存命中率 | – | 0.656 | 0.668 |
| P50 / P95 延迟 | 59.9 ms / 809.5 ms | 4.9 s / 16.7 s | 6.0 s / 25.2 s |
| 单任务成本 | – | $0.00084 | $0.00159 |

GLM 更便宜但更慢：开发集单任务成本 workflow_llm $0.00028、agent $0.00143；P95 分别约 23 秒和 87 秒（`ablation-glm-dev.json`）。

## 置信区间支持的结论

- **开发集（DeepSeek，`846bc5e`）**：workflow 0.981、workflow_llm 0.979、agent 0.986，两两之间都不显著。
- **保留集（DeepSeek，`846bc5e`）**：
  - Agent 对 workflow_llm：0.956 对 0.906，Δ +0.050 [−0.006, +0.119]，**不显著**；保留集 **pass^3 两者都是 0.906**（McNemar 2 / 2 个不一致任务，p = 1.0）。
  - Agent 对确定性 workflow：任务成功率显著（+0.107 [+0.038, +0.189]），但 pass^3 上是 3 / 0，p = 0.25。
- **测试集 v2（DeepSeek，`38a3069`）**：
  - **两条 LLM 路径都显著好于确定性 workflow**：workflow_llm +0.039 [+0.008, +0.074]，agent +0.077 [+0.033, +0.124]。
  - **主运行的 agent 优势不可信**：主运行里 agent 对 workflow_llm 是 +0.039 [+0.005, +0.074]，但那次 24% 的 agent 轮次遇到网关 HTTP 429 并降级到规划器。
  - **低并发重跑**：在 `--workers 3` 下重跑（降级 6%），agent 得分不变（0.802），与 workflow_llm 的差值是 +0.036 [−0.011, +0.085]，**不显著**；pass^3（0.777 对 0.760，McNemar 7 / 5，p = 0.77）也没有差异（`ablation-test_v2-deepseek-agent-w3.json`）。
- **第二个模型族（GLM-5.3 Flash）**：
  - **保留集**：agent 0.981 对 0.906，+0.075 [+0.019, +0.151]，显著；McNemar 4 / 0，p = 0.125。
  - **测试集 v2**：agent 0.810 对 0.760，+0.050 [+0.008, +0.091]，显著；pass^3 McNemar 8 / 3，p = 0.23。
  - **开发集上反过来**：agent 0.974 对 workflow_llm 0.995，−0.021 [−0.035, −0.008]，McNemar 1 / 11，p = 0.006。GLM Agent 在对比和宏观答案里漏掉必需数字。
- **两个模型都成立的结论**：
  1. 在规则没针对过的问法上（测试集 v2），LLM 组答和 Agent 的单次成功率比确定性流程高 3–8 个百分点。
  2. Agent 相对 LLM 组答的优势最多是几个百分点的单次成功率，只在 GLM 上显著，从未表现为显著的 pass^3 提升，成本却高 2–5 倍。「Agent 在保留集上领先」这一说法在 DeepSeek 上**不成立**。`mode=auto`（简单问题走确定性路径，开放问题走 Agent）是成本/延迟上的选择，不是已被证明的质量提升。
- **运行间波动**：
  - **数值**：同样任务集的三次 DeepSeek 运行（`c1c3388`、`1beb760`、`846bc5e`）里，保留集 Agent pass^3 分别是 0.943、0.849、0.906，相差 0.094，其中也包含代码和 Prompt 的变化。
  - **含义**：两次单独运行在保留集上相差 5 个百分点以内不能当作证据。

## Prompt 版本与 A/B

| 版本 | 改动 | 结果 |
|---|---|---|
| v1 | 最初的 Prompt。 | 基线。 |
| v2 | 标签分区、按问题类型控制投入、写明每条规则的理由、一个输出示例。 | **成本**：Agent 开发集单任务成本 −69%（$0.00401 → $0.00126），这一点是稳健的。**质量**：开发集 Agent +0.011 [+0.002, +0.022]（小但显著，pass^3 McNemar p = 0.07）；保留集 Agent −0.031 [−0.113, +0.038]（不显著）。**副作用**：保留集判断类问题的限定语从 1.00 掉到 0.67，因为 v2 删掉了 v1 的「描述不确定性与风险」。 |
| v3 | v2 加上判断类问题的明确规则（条件性观点、不确定性、风险）。 | 默认版本。保留集 Agent 限定语 0.91。 |

- **A/B 的条件**：v1/v2 在同一 commit（`1beb760`）上比较；v3 跑在 `846bc5e` 上，那里还包含观察和工具错误的改动，所以 v3 对 v2 不是纯 Prompt 比较。
- **撤回的说法**：早先「v2/v3 把保留集 Agent pass^3 从 0.849 提到 0.906」**已撤回**，两个数都在对方的置信区间里，而 v1 时代 `c1c3388` 的一次运行是 0.943。有数据支撑的是成本下降。
- **版本锁定**：Prompt 文本按哈希锁定在 `query_intelligence/agent/prompts.lock.json`，每份 trace 和报告都记录 `id@version#sha`。

## 离线门禁、路由评测与第二轮

- **离线门禁基线**（CI 与之比较，`gate-dev.json`、`gate-holdout.json`，`d3c1495`）：确定性路径开发集 238 个任务 1.000；保留集 0.906 [0.83, 0.98]（此前在 `f7bf624` 是 0.849）。
- **路由评测**（`router_eval-*.json`）：
  - **准确率**：在带路由标签的问题集上，`d78b313` 在 158 条上准确率 0.715，第二轮（`da3ec8b`）在 162 条上 0.975，剩余 4 个错误全部列在文件里。
  - **各路由**（第二轮）：拒答召回 0.967、澄清召回 0.909、固定流程和 Agent 召回 1.0。
  - **性质**：标注由作者按书面路由策略编写，这是回归与策略一致性检查，不是独立基准。
- **第二轮修复针对测试集 v2 暴露的两类失败**：
  - **两类失败**：一是不点名公司的追问（「ROE呢」）被当成超范围拒答，多轮类只有 0.21–0.32；二是悬空问题没有澄清（0.33）。
  - **修复**：省略追问补全、「只有指标没有公司」时澄清、模糊概念过滤。它们只在新写的开发集风格任务上开发和验证，没有拿测试集 v2 的题调整。
  - **撞题检查**：新写的开发集题目中有两道和测试集 v2 一字不差，被重叠测试拦下后改写。
  - **测试集 v2 已被读过**：由于失败类别是从测试集 v2 读出来的，它此后是验证集；要得到无偏估计需要新的测试集。

## 校验器压力测试

`verifier_stress.json`，commit `2494656`：对 159 个正确答案生成 2,433 个篡改变体（数字改动 1%、5%、20%，或在公司之间互换），看校验器放过多少。越低越好；正确答案必须全部通过。

| | 原来的整批比对 | 按运行比对 | 逐句绑定（当前） |
|---|---|---|---|
| 正确答案通过率 | 1.000 | 1.000 | 1.000 |
| 全部篡改的误放率 | 0.343 | 0.243 | **0.021** |
| 改动 1%（593） | 0.320 | 0.039 | 0.039 |
| 改动 5%（645） | 0.078 | 0.030 | 0.019 |
| 改动 20%（665） | 0.098 | 0.035 | 0.023 |
| 公司间互换（530） | 1.000 | 0.994 | **0.002** |

被拒的 2,382 个答案按整句删除修复（不再拆子句），无内容可留时改用模板回答：修复后 100% 可读（无残句、孤立引用、多余标点），100% 通过校验，未被篡改的句子保留 97.8%，17.2% 回退到模板。原来的子句拼接只有 29.1% 可读，96.5% 含残句。

## 提示注入红队

攻击文本投毒在搜索结果里，只统计文档工具确实返回了投毒文本的运行。

- **在线**（`redteam-online.json`，`846bc5e`，DeepSeek）：
  - **开发攻击集**：9 种攻击 × 4 种混淆（全角、原文、拆分、零宽字符），三条路径各 72 次，成功率都是 0，关键词过滤全部拦下。
  - **保留攻击集**：8 种攻击，关键词过滤一次都没拦下，但结构性防护让成功率保持很低：workflow 0/64、workflow_llm 2/64、agent 1/64。
  - **成功的三次**：「海盗」口吻的人设改变、复述投毒文档里的「强烈买入」评级、新闻摘录里植入的假收盘价。最后一次能通过数字校验，因为植入的数字确实在所引证据里：校验证明的是可追溯，不是真实。
- **离线**（`redteam-offline.json`，`2494656`，workflow 路径，CI 基线）：开发集 0/72、保留集 0/64、holdout2 0/64（`f7bf624` 时为 2/64）；holdout3（第二轮评审的 11 个投毒攻击，在修复前加入，不用于调参）修复前 6/88，修复后 **2/88**：剩下的是“拆分”变体把伪造的合并消息作为标题带进答案，标题本身看起来就是一条普通新闻标题，词法规则无法区分。
- **之后的防护**：`ceaef0b` 之后加入的策略级防护（评级/仓位删除、语言守卫等）目前只由离线基线覆盖，还没有新的在线运行。

<!-- final2: add the red-team rerun (redteam-final2) at the round-2 commit -->

## 故障注入

`fault_injection.json`，`f7bf624`：用替身工具和脚本化 LLM 模拟故障（不是在真实数据源上注入），11 个场景各 5 次，平稳降级率都是 1.00：

- 工具超时、上游 5xx（重试后报 `upstream_error`）、空结果、慢工具、超大文档（截断后仍能作答）；
- 文档注入（指令被脱敏、没有交易建议）、LLM 宕机（降级到规划器）；
- 畸形工具参数、未知工具、LLM 编造数字（被删除）、无休止的工具调用（步数预算叫停）。

在真实网关和真实数据源上做的故障演练见 [A2A、容灾与可观测性](a2a-and-observability.md#故障演练)。

## 在线运行中的网关限流

- **并发评测作废**：DeepSeek 和 GLM 的评测同时跑在同一网关上，24–52% 的 LLM 轮次报错。那些结果作废，只作为证据保留（`ablation-test_v2-deepseek-concurrent.json`），GLM 开发集顺序重跑为 `ablation-glm-dev.json`。
- **单独运行也会被限流**：DeepSeek Agent 在 `--workers 6` 时也有 24% 的测试集轮次遇到 HTTP 429（每轮约 2 次 LLM 调用），`--workers 3` 时降到 6%。
- **新增指标**：摘要现在会报告 `llm_error_rate`、最常见的错误类型，以及答案草稿的 JSON 修复率和失败率。
- **更早的运行**：没有这项指标的运行（`c1c3388`、`1beb760`、`846bc5e`）可能含有未知比例的降级。

## 传统基线为什么会变（开发集 0.2657 → 0.256，保留集 0.2075 → 0.1887）

- **根因**：原 `/chat` 的时效守卫读的是 `date.today()`。
  - **交易日**：问「现在/最新」而最新行情早于今天时，回复「未获取到今日…因此不能据此判断…」，其中含限定语。
  - **非交易日**：回复「今天不是 A 股常规交易日…」，没有限定语。
- **发生了什么**：最终运行（`846bc5e`）结束于北京时间 2026-09-26（周六）02:34，其离线 legacy 部分在午夜之后运行，于是三个任务失去了限定语，正好是开发集 2/207 和保留集 1/53。
- **复现**：固定日期后，在 `846bc5e` 和 `1beb760` 上都能复现两组数值。
- **修复**：`28b587e` 起，消融给 legacy 路径传入评测日期（2026-04-23）。

## 已知失败与根因

- **测试集 v2 的多轮与悬空问题**：见上文「第二轮」。第二轮的修复还没有在测试集 v2 上重新做在线测量。
- **英文缺数据任务**：测试集 v2 里两个英文「缺数据」任务仍未通过必需工具检查。
- **限定语的词法触发**：在确定性路径上会漏掉一些说法（例如「抄底」「翻倍」「止盈点位」），LLM 路径则通过 Prompt 实现限定语。
- **红队保留攻击**：3/128 次成功，见上文。

## 复现

```bash
# 任务集与快照
python -m evaluation.agent_eval.build_tasks          # 开发集（检查与训练数据的重叠）
python -m evaluation.agent_eval.build_holdout        # 保留集
python -m evaluation.agent_eval.build_test_v2        # 测试集 v2（派生事实、重叠报告）
python -m evaluation.agent_eval.runner --mode workflow --record-missing   # 新增任务时只补录缺失的工具调用
# 离线
python -m evaluation.agent_eval.ablation --sets dev,holdout,test_v2 --out outputs/agent_eval/ablation-offline.json
python -m evaluation.agent_eval.gate                 # 下限 + 已提交基线；有意改动后用 --update-baseline
python -m evaluation.agent_eval.router_eval
python -m evaluation.agent_eval.fault_injection
python -m evaluation.agent_eval.verifier_stress
python -m evaluation.agent_eval.redteam && python -m evaluation.agent_eval.gate --extras-only
# 在线（OpenAI 兼容接口：DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY / DEEPSEEK_MODEL）；一次只跑一个
python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2 \
    --modes workflow_llm,agent --out outputs/agent_eval/ablation-deepseek.json
# 提交证据并重新生成完整页面
python -m evaluation.agent_eval.results outputs/agent_eval/ablation-deepseek.json   # -> evaluation/results/
python -m evaluation.agent_eval.report               # CI 中用 --check
```
