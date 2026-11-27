# 评测摘要（中文）

语言：[自动生成完整英文版](../agent-eval.md) | [评测历史](evaluation-history.md) | 中文

本页为 [`docs/agent-eval.md`](../agent-eval.md) 的精简中文导读，汇总 FinSight 的核心评测口径、对比基线、校验器压力测试、提示注入红队与复现命令。

## 评测口径

- **一票否决（dealbreaker）**：每题必须同时满足（1）行为正确（回答/澄清/拒答）；（2）全部必需事实陈述且带有效 `[evidence_id]` 引用；（3）调用必需工具；（4）判断类问题带限定语；（5）全文无买卖或仓位指令。
- **统计方法**：在线评测每题重复 3 次，报告 95% bootstrap 置信区间、配对 bootstrap 差值与精确 McNemar 检验；独立任务集永久保留未暴露首次结果。

## 核心指标总览

| 维度 | 核心指标 | 证据来源 |
|---|---|---|
| **论文 Query Intelligence（10,000 双语查询）** | 金融召回 **0.975**、越界准确率 **0.9729**、意图 F1 **0.9407**、主题 F1 **0.9208**、Recall@10 **0.9664** | [FAIA 2026](https://ebooks.iospress.nl/doi/10.3233/FAIA261107) |
| **独立测试集 v3（DeepSeek 首次，130 题）** | Agent **0.869** `[0.81, 0.92]`、LLM 组织答案 0.831、确定性流程 0.769、纯 LLM（无工具）**0/130** | [`ablation-final4-deepseek-testv3-holdout.json`](../agent-eval.md#test-set-v3-independent-author-untouched-first-runs) |
| **校验器压力测试（333 正例，5,631 篡改）** | 严格误放 **0.92%**（允许派生数字 1.05%）、正例通过 **100%**、跨公司换数 **0.0%** | [`verifier_stress-round18.json`](../agent-eval.md#verifier-stress-test-333-gold-answers-5631-corrupted-variants-verifier_stress-round18json) |
| **提示注入红队（16 套攻击集）** | 模板路径 **0 / 3,796** 成功；全量 LLM 运行 0–3.4% 当作事实；最新定向切片（holdout13）Agent 0/16、组织答案 0/13 | [`redteam-offline-r18.json`](../agent-eval.md#prompt-injection-red-team-offline-template-path-after-round-18-all-sixteen-sets-ci-baseline) |
| **声明核查（Claim Check）** | 离线开发集（311）**1.000**、保留集（47）**1.000**；盲收集真实公开说法首次得分 **0.500–0.740**（v2–v6） | [claim-check.md](claim-check.md) |
| **离线 CI 门禁基线** | `dev`（558）**1.000**、`holdout`（53）**0.943**、`ext`（123）**1.000**、`shipped`（21）**1.000** | [`docs/agent-eval.md`](../agent-eval.md#offline-gate-baselines-ci-compares-against-these) |
| **故障注入（11 场景）** | **11/11** 平稳降级（超时、5xx、空数据、超大文档、LLM 宕机、死循环） | [`fault_injection.json`](../agent-eval.md) |

## 关键统计结论

1. **工具检索 + 校验是核心增益**：纯 LLM（无工具）在严格引用口径下通过 `0/130`；仅靠确定性规划 + 模板即可达 `0.769`；加入规划预取 LLM Agent 提升至 `0.869`（`+0.100`，`p = 0.007`）。
2. **混合路由性价比最优**：简单事实查询走确定性流程零成本且 P95 `< 2 s`；复杂多跳/归因/对比问题走规划预取 Agent；慢速推理模型（如 GLM）通过 `QI_AGENT_SLOW_MODEL_POLICY=on` 自动走工作流 + LLM 组织答案，避免 `82 s` 级长尾。
3. **整句删除修复优于子句裁剪**：校验失败后执行整句级删除修复，可读性与二次校验通过率均达 `100%`（旧子句裁剪仅 `26%` 可读）。

## 复现命令

```bash
# 离线门禁、路由、校验器与红队
python -m evaluation.agent_eval.gate
python -m evaluation.agent_eval.router_eval
python -m evaluation.agent_eval.verifier_stress
python -m evaluation.agent_eval.redteam
python -m evaluation.agent_eval.fault_injection

# 在线消融（需配置 OpenAI 兼容端点 DEEPSEEK_*）
python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2

# 校验并重新生成 docs/agent-eval.md 与 README 指标区块
python -m evaluation.agent_eval.report --check
```
