# 评测历史与方法总览

语言：[English](../evaluation-history.md) | 中文 · [评测摘要（中文）](evaluation.md) · [自动生成完整版](../agent-eval.md) · [已知问题](known-issues.md)

本文汇总 FinSight 的评测口径、独立任务集（首次运行 vs 暴露后）、校验器与红队结果、完整局限以及 CI / 人工评测流程。文中引用的所有结果文件均提交于 [`evaluation/results/`](../../evaluation/results/)，由 `python -m evaluation.agent_eval.report --check` 自动校验其存在性与一致性。

## 评测口径与任务集

Agent 评测采用严格**一票否决（dealbreaker）**口径：行为（回答/澄清/拒答）、全部必需事实陈述且引用、必需工具调用、判断类问题限定语、全文无交易指令五项全部通过才算成功。在线评测每题重复 3 次，报告 95% bootstrap 置信区间与配对 bootstrap / 精确 McNemar 检验。

| 任务集 | 规模 | 作者与暴露状态 |
|---|---|---|
| **开发集（`dev`）** | 558 题 | 项目作者编写，用于驱动修复（`gate-dev.json`：**1.000**） |
| **保留验证集（`holdout`）** | 53 题 | 项目作者编写，曾用于 Prompt 选型（`gate-holdout.json`：**0.943**） |
| **扩展快照集（`ext`）** | 123 题 | 开启离线扩展层直连工具，无回放（`gate-ext.json`：**1.000**） |
| **发布默认集（`shipped`）** | 21 题 | 扩展层 + 一手年报更正层全开（`gate-shipped.json`：**1.000**） |
| **测试集 v3** | 130 题 / 155 轮 | 独立作者编写；首次 LLM 运行（`9536abf`）完全未接触，后续仅用于 `v4` Prompt A/B 决策一次 |
| **测试集 v2 / 多轮 v1** | 121 题 / 49 组对话 | 保留首次未暴露结果；修复缺陷类别后标记为**暴露后** |
| **外部评审切片（r4–r14）** | 每轮 38–60 对话 + 22–67 声明 | 独立评审在修复前编写；仅对当轮修复为样本外，并附文本/内容/特征三档去重计数（`slice_overlap`） |
| **盲收集真实说法（v2–v6）** | 每套 50–60 条 | 未读核查代码前从公开新闻/研报/社区盲收集 |
| **独立路由标注（v1 / v2）** | 154 / 241 条 | 独立作者依据书面路由规则标注 |

## 最终在线消融（`9536abf`，3 次重复）

数据来源：[`ablation-final4-deepseek-testv3-holdout.json`](../agent-eval.md#test-set-v3-independent-author-untouched-first-runs)、`ablation-final4-deepseek-testv2-multiturn.json`、`ablation-final4-glm-testv3.json`、`ablation-test_v3-purellm-deepseek.json`。

| 回答路径 | 测试集 v3（DeepSeek 首次） | 测试集 v3（GLM） | 保留集（DeepSeek） | 测试集 v2（暴露后） | 多轮 v1（暴露后） |
|---|---|---|---|---|---|
| 经典 `/chat`（无 LLM） | 0.008 | 0.008 | 0.208 | 0.223 | 0.000 |
| 纯 LLM（无工具） | **0.000**（0/130） | — | — | — | — |
| 确定性工作流（无 LLM） | 0.769 [0.69, 0.84] | 0.769 | 0.943 | 0.901 | 1.000 |
| 工作流 + LLM 组织答案 | 0.831 [0.76, 0.89] | 0.818（P95 27 s） | 0.981 | 0.931 | 0.980 |
| **LLM Agent（规划预取）** | **0.869 [0.81, 0.92]**（pass^3 0.854） | 0.815（P95 82 s） | **1.000** | **0.959** | 0.959 |

- **DeepSeek 测试集 v3 配对检验**：Agent 相对确定性流程 **+0.100** `[+0.049, +0.156]`（McNemar `p = 0.007`）；Agent 相对 LLM 组织答案单次 `+0.038`，`pass^3` 差异不显著（`p = 0.29`）。
- **慢速模型策略**：GLM 上 Agent 与组织答案持平（`-0.003`）但 P95 高达 `82 s`，因此 `mode=auto` 默认将慢速推理模型路由至工作流 + LLM 组织答案（`QI_AGENT_SLOW_MODEL_POLICY=on`）。
- **Prompt `v4` 默认版本**：在测试集 v3 A/B 对比（`ablation-ab-prompt-v3-testv3.json` vs `ablation-ab-prompt-v4-testv3.json`）后启用，新增文档防注入规则且未引入显著通过率损失。

## 独立任务集：首次运行与暴露后

所有独立评测集均永久保留未暴露首次运行成绩，不以修复后的结果覆盖：

| 独立集合 | 首次运行（无偏估计） | 修复后 / 暴露后 |
|---|---|---|
| 测试集 v3（DeepSeek，130 题） | **0.869** `[0.81, 0.92]`（`ablation-final4-deepseek-testv3-holdout.json`） | 未针对调优 |
| 多轮 v1（无 LLM / DeepSeek） | 0.224 / 0.361（`multiturn_v1-auto-nollm-first-run.json`、`ablation-multiturn_v1-deepseek-first-run.json`） | 1.000 / 0.959（暴露后） |
| 独立路由标注 v2（241 条） | **0.801**（`router_eval-independent_v2-first-run.json`） | 0.842（暴露后） |
| 外部评审切片（对话，单轮修复） | r6 0.579 · r7 0.264 · r8 0.275 · r9 0.346 · r10 0.250 · r11 0.200 · r12 0.320 · r13 0.413 · **r14 0.160** | r6 0.816 · r7 0.830 · r8 0.686 · r9 0.788 · r10 0.792 · r11 0.733 · r12 0.840 · r13 0.978 · **r14 0.740**（`chat_heldout_r14-auto-nollm-after-fix.json`） |
| 盲收集真实说法（v2 / v3 / v4 / v5 / v6） | **0.500** / **0.540** / **0.740** / **0.680** / **0.650**（`claims_web_v2-first-run.json` … `claims_web_v6-first-run.json`） | 0.840 / 0.880 / 0.780 / 0.680 / 0.650（第 18 轮核查器） |
| 校验器压力测试（333 正例 / 5,631 篡改） | 严格模式误放 **0.92%**（允许派生数字 1.05%）、正例通过 100%、跨公司换数 0.0%（`verifier_stress-round18.json`） | CI 回归基线 |
| 提示注入红队（16 套攻击集） | 模板路径 **0 / 3,796** 成功（`redteam-offline-r18.json`）；全量 LLM 运行 0–3.4% 当作事实（`redteam-r8-llm.json`）；最新定向 LLM（holdout13）Agent 0/16、组织答案 0/13（`redteam-r16-holdout13-llm.json`） | CI 持续回归 |
| 故障注入（11 场景） | **11/11** 体面降级（`fault_injection.json`） | — |

**切片来源说明**：第 8–14 轮修复后运行读取的是外部评审目录切片，切片在运行完成后提交入库；由于 git 无法审计本地文件访问，每轮均由 `slice_overlap` 额外报告剔除文本/内容/特征重合后的保守通过率。

## 完整的局限说明

- **模型覆盖范围**：在线消融仅基于两家 Flash 级模型（DeepSeek V4.1 Flash、GLM-5.3 Flash）经单一兼容网关测试。
- **小规模未暴露切片**：每轮评审切片含 38–60 题且仅对当轮修复为样本外；比较语法仍基于词类规则，新问法仍可能漏判（如第十四轮修复后仍有 13/50 失败）。
- **新问法路由误差**：独立路由标注首次准确率 `0.801`（约 20% 误判），集中在无标的建议与澄清/拒答边界。
- **Agent 相对组织答案优势有限**：DeepSeek 上单次领先（`+0.038`）但 `pass^3` 不显著，GLM 上持平且延迟/成本更高。
- **可追溯不等于外部客观真实**：校验器确保数字源自本句所引证据，但无法仅凭校验判断新闻正文中投毒数字的真伪（依赖基本面交叉核验与输出安全层补防）。
- **LLM 路径残留注入转述风险**：输出安全层可标注或剔除可识别模式，但无法彻底阻止模型转述无数字的恶意叙事。
- **规则化追问与离线覆盖边界**：指代消解与声明核查依赖词表规则，离线快照仅覆盖 22 只行情标的与 13 家基本面公司。
- **人工评测尚未执行**：人工质量打分、问财/豆包/Kimi 盲测对比、可用性研究与所有者真实说法集均已备好脚本（`evaluation/human/`），尚待人工执行。

## CI、推送前检查与人工评测工具包

- **GitHub Actions CI**（`.github/workflows/ci.yml`）：执行 `ruff` 检查、全历史 `gitleaks`、前端检查与构建一致性、含 Postgres + Playwright E2E + `axe-core` + 分支覆盖率下限（`agent ≥ 88%`、`api ≥ 90%`、`answer_guards ≥ 81%`）的全量 `pytest`、发布默认数据全量测试（`tests-shipped`）、离线评测门禁（`dev`/`holdout`/`ext`/`shipped` + `claim_bench`）、`verifier_stress` + 离线红队、保留切片校验 + `report --check`、Docker 只读冒烟、Kubernetes 清单校验及 `kind` 集群冒烟。
- **本地推送前镜像**：[`scripts/prepush.sh`](../../scripts/prepush.sh) 本地运行全部核心步骤并校验 CI 时间预算（见 [development.md](development.md#ci-时间预算)）。
- **人工评测工具包**：[`evaluation/human/README.md`](../../evaluation/human/README.md) 提供 SHA-256 锁定的评分模板与自动化计分脚本。
