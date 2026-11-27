# 开发规范与推送前检查

语言：[English](../development.md) | 中文

代码改动推送到 `master` 前，使用 [`scripts/prepush.sh`](../../scripts/prepush.sh) 在本地完整运行与 GitHub Actions 一致的校验套件。环境初始化与架构约束请参阅 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 本地 CI 镜像（`scripts/prepush.sh`）

```bash
scripts/prepush.sh                          # 运行全部本地 CI 步骤
PYTHON=.venv/bin/python scripts/prepush.sh  # 指定虚拟环境解释器
SKIP=e2e,extras scripts/prepush.sh          # 按步骤名跳过部分耗时项
```

| 步骤 | 命令 | 对应 CI Job |
|---|---|---|
| `ruff` | `ruff check . && ruff format --check .` | `lint` |
| `actionlint` | `actionlint .github/workflows/*.yml`（若已安装） | — |
| `pytest` | `python -m pytest -q tests`（v1 种子基线 + 默认数据单测） | `tests` |
| `pytest-shipped` | `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests`（扩展层 + 更正层） | `tests-shipped` |
| `fe-*` 与 `dist-diff` | `frontend/` 下 `pnpm install/typecheck/lint/test/build` 且 `query_intelligence/web/dist` 无未提交漂移 | `frontend` |
| `e2e` | `pnpm run e2e`（Playwright 驱动离线服务） | `tests` |
| `gate-dev-holdout`、`gate-ext`、`gate-shipped` | `python -m evaluation.agent_eval.gate` 校验各基线 | `agent-eval-gate` |
| `claim-holdout` | `python -m evaluation.claim_bench.run --set holdout` | `agent-eval-gate` |
| `extras` | `verifier_stress`、`redteam`、`gate --extras-only` | `agent-eval-extras` |
| `verify-slices` | `python -m evaluation.agent_eval.verify_slices` | `heldout-verify` |
| `report-check` | `python -m evaluation.agent_eval.report --check`（校验 `docs/agent-eval.md` 与 README 区块） | `heldout-verify` |

日志写入 `outputs/prepush/<commit>/<step>.log`，摘要写入 `summary.md`（已归档摘要见 `docs/results/ci/`）。仅在 CI 或手动执行的额外检查包括：全历史 `gitleaks`、Postgres 多副本测试（`QI_TEST_POSTGRES_DSN`）、分支覆盖率下限（`agent ≥ 88%`、`api ≥ 90%`、`answer_guards ≥ 81%`）、Docker 构建与 `deploy/k8s-smoke/smoke.sh`。

## CI 时间预算

[`scripts/ci_budget.py`](../../scripts/ci_budget.py) 在 `prepush.sh` 每一步结束后，结合 `.github/workflows/ci.yml` 中的 `timeout-minutes` 估算对应 CI Job 的云端耗时；当累计预估超过 `CI_BUDGET_FRACTION`（默认 `0.6`）时标记 `FAIL (CI time budget)`。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CI_BUDGET_FRACTION` | `0.6` | 允许占用的超时比例上限（设为 `0` 或 `off` 仅报告不报错） |
| `CI_SLOWDOWN` | `3` | 云端 runner 相对本机的默认耗时倍率 |
| `CI_SLOWDOWN_STEPS` | `pytest=5` | 分步骤倍率（CI 的 pytest 开启分支覆盖率） |
| `CI_BUDGET_BUSY_LOAD` | 性能核数（或半数 CPU） | 高负载宿主机超时误报时标注 `"host busy"` |

映射完整性由 `tests/test_ci_budget.py` 强制校验。

## 种子基线与发布默认数据双轨测试

离线评测集默认锁定 v1 种子（`tests/conftest.py` 关闭扩展层与更正层）以保持纵向可比性；生产默认行为（三层全开）由四项独立测试保障：

- `tests/test_shipped_default.py` 与 `tests/test_data_layers.py`（默认设置、年报更正值、Dockerfile 文件清单、缺失层 `/ready` 503）；
- `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests`（CI `tests-shipped`，排除 `tests/seed_data_tests.txt` 中的种子断言）；
- `python -m evaluation.agent_eval.gate --set ext --set shipped`；
- Playwright E2E（`frontend/playwright.config.ts` 开启扩展层与更正层）。

## 自动生成的评测页与 README 区块

`README.md` 与 `README_CN.md` 中位于 `<!-- generated:<region> (python -m evaluation.agent_eval.readme) -->` 标记内的指标由 [`evaluation/agent_eval/readme.py`](../../evaluation/agent_eval/readme.py) 根据已提交结果自动渲染：

```bash
python -m evaluation.agent_eval.report          # 重新生成 docs/agent-eval.md 与 README 生成区块
python -m evaluation.agent_eval.report --check  # 检查区块与引用是否最新
python scripts/record_ci_run.py                 # 通过 gh CLI 记录最新 master CI 结果
```
