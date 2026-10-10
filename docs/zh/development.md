# 开发：推送前的检查

语言：[English](../development.md) | 中文

## 规则

**在准备推送到 `master` 的那个提交上运行 `scripts/prepush.sh`，通过了才推送。** 合并提交也一样：合并之后的代码必须跑通浏览器
e2e 测试，不能只看各分支各自通过。

原因：第 13 轮把更正层设为产品默认，但 e2e 用例仍断言 v1 种子数值，合并提交没跑 e2e 就进了 `master`，于是 `master` 的 CI 在
`7f4f06c` 上是红的（run 38061714150），直到只改用例的修复 `f8f5546`（run 38064588334，通过）。这个脚本在本地跑 CI 跑的东西，
避免同样的事情再悄悄发生（第十轮评审 J1）。

## 运行内容

```bash
scripts/prepush.sh                          # 全部步骤，笔记本上约 15–25 分钟
PYTHON=.venv/bin/python scripts/prepush.sh  # 指定装了 requirements-dev.txt 的解释器
SKIP=e2e,extras scripts/prepush.sh          # 按名字跳过步骤；汇总里会标为 skipped
```

| 步骤 | 命令 | 对应的 CI job |
|---|---|---|
| `ruff` | `ruff check .` 和 `ruff format --check .` | lint |
| `actionlint` | `actionlint .github/workflows/*.yml`（没装 actionlint 时跳过） | – |
| `pytest` | `python -m pytest -q tests`（包括下文的 `tests/test_shipped_default.py`） | tests |
| `fe-install`、`fe-typecheck`、`fe-lint`、`fe-test`、`fe-build` | 在 `frontend/` 下依次 `pnpm install --frozen-lockfile`、`pnpm typecheck`、`pnpm lint`、`pnpm test`、`pnpm build` | frontend |
| `dist-diff` | `git status --porcelain query_intelligence/web/dist` 必须为空（提交的构建产物与源码一致） | frontend |
| `e2e` | `pnpm run e2e`：Playwright 用真实 Chrome 访问离线服务 | tests |
| `gate-dev-holdout` | `python -m evaluation.agent_eval.gate --set dev --set holdout`（固定在 v1 种子） | agent-eval-gate |
| `gate-ext` | `gate --set ext`（开扩展层，关更正层） | agent-eval-gate |
| `gate-shipped` | `gate --set shipped`（扩展层**和**更正层都开：产品实际提供的数据） | agent-eval-gate |
| `extras` | `verifier_stress`、`redteam`、`gate --extras-only` | agent-eval-gate |
| `claim-holdout` | `claim_bench.run --set holdout`，用 CI 的下限 | agent-eval-gate |
| `verify-slices` | 八个保留切片的 verify 脚本（r9 带 `QI_EVAL_SNAPSHOT_EXT=1`） | agent-eval-gate |
| `report-check` | `python -m evaluation.agent_eval.report --check` | agent-eval-gate |

前面的步骤失败后，后面的步骤照样运行；只有所有运行了的步骤都通过，退出码才是 0。与 CI 一样，实时数据源和 trace 都关闭。
e2e 服务由脚本挑一个空闲端口启动：在 CI 之外，Playwright 会复用已经在该端口上监听的服务，那可能是另一个工作目录的代码。

本地不覆盖的：gitleaks 全历史扫描、依赖 Postgres 的 checkpointer 测试（需要 `QI_TEST_POSTGRES_DSN`）、覆盖率下限、Docker
镜像和 kind 集群。后两项这样跑：

```bash
docker build -f docker/Dockerfile -t finsight:$(git rev-parse --short=7 HEAD) .
deploy/k8s-smoke/smoke.sh        # 需要 kind 和 kubectl；断言 /ready、三个数据层和一个更正后的数字
```

## 输出

日志写到 `outputs/prepush/<commit>/<step>.log`，汇总写到 `outputs/prepush/<commit>/summary.md`（或 `SUMMARY=<路径>`）：
提交、工作区是否干净、Python / Node / pnpm 版本，以及每一步的命令、结果、秒数和日志里的结果行（pytest 的通过数、门禁的任务
成功率等）。提交到仓库的汇总在 [`docs/results/ci/`](../results/ci/)，按运行时的提交命名。

## 测的是发布默认配置，而不只是固定的种子数据

评测集和大多数测试固定在 v1 种子上（`tests/conftest.py` 设了 `QI_OFFLINE_SNAPSHOT_EXT=false` 和
`QI_OFFLINE_SNAPSHOT_FIX=false`；评测框架要用 `QI_EVAL_SNAPSHOT_EXT=1` / `QI_EVAL_SNAPSHOT_FIX=1` 才会打开），因为它们的
标注是按种子写的。部署实际提供的数据由以下几处覆盖：

* `tests/test_shipped_default.py`：删掉这两个变量，按 `Settings` 的默认值构建服务，向 API 询问只有这两层才会给出的数字（茅台
  市盈率 21.44、五粮液 ROE（加权） 6.89% 和营业收入 405.29 亿元、中国平安 58.03、宁德时代 ROE（期末净资产） 21.42%、相差
  25.64 个百分点），并禁止出现种子数值。把更正层的默认值改成关闭，8 个测试全部失败。
* 门禁集 `shipped`（`tasks/agent_eval_shipped_v1.jsonl`，21 个任务 / 22 轮，由
  `evaluation/agent_eval/build_shipped_tasks.py` 生成，基线 `evaluation/results/gate-shipped.json`）：两层都开，不用回放，
  期望值取自 `data/snapshot/structured_data_fix.json`。两层都开时 21/21，关掉更正层时 2/21。
* `tests/test_data_layers.py`：Dockerfile 复制了加载器读取的每个数据层文件；开启了却缺失的数据层会记一条警告，并让
  `GET /ready` 返回 503。
