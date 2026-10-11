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
scripts/prepush.sh                          # 全部步骤：在 278f7cc 上，Apple 芯片笔记本耗时 35 分 40 秒
PYTHON=.venv/bin/python scripts/prepush.sh  # 指定装了 requirements-dev.txt 的解释器
SKIP=e2e,extras scripts/prepush.sh          # 按名字跳过步骤；汇总里会标为 skipped
CI_BUDGET_FRACTION=0.5 scripts/prepush.sh   # CI 时间预算比例（默认 0.6；off 表示只报告不判失败）
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
| `extras` | `verifier_stress`、`redteam`、`gate --extras-only` | agent-eval-extras |
| `claim-holdout` | `claim_bench.run --set holdout`，用 CI 的下限 | agent-eval-gate |
| `verify-slices` | 十个保留切片的 verify 脚本（r9 带 `QI_EVAL_SNAPSHOT_EXT=1`，r10、r11 另带 `QI_EVAL_SNAPSHOT_FIX=1`） | heldout-verify |
| `report-check` | `python -m evaluation.agent_eval.report --check` | heldout-verify |

前面的步骤失败后，后面的步骤照样运行；只有所有运行了的步骤都通过，退出码才是 0。与 CI 一样，实时数据源和 trace 都关闭。
e2e 服务由脚本挑一个空闲端口启动：在 CI 之外，Playwright 会复用已经在该端口上监听的服务，那可能是另一个工作目录的代码。

本地不覆盖的：gitleaks 全历史扫描、依赖 Postgres 的 checkpointer 测试（需要 `QI_TEST_POSTGRES_DSN`）、覆盖率下限、Docker
镜像和 kind 集群。后两项这样跑：

```bash
docker build -f docker/Dockerfile -t finsight:$(git rev-parse --short=7 HEAD) .
deploy/k8s-smoke/smoke.sh        # 需要 kind 和 kubectl；断言 /ready、三个数据层和一个更正后的数字
```

## CI 时间预算

一个步骤在本地通过，仍可能因为时间让 CI 失败。run 38095861905（`bb99360`）里每个步骤都通过了，但 `agent-eval-gate` job
在它 45 分钟的 `timeout-minutes` 处被取消：第 15 轮的切片让它的最后一步变长，而这个 job 在 `4ad3380` 上已经用掉 45 分钟里的
43 分 51 秒。推送前检查只知道通过或失败，看不到这一点（第十二轮评审 L16）。

从第 16 轮起，`scripts/prepush.sh` 每跑完一步就调用 [`scripts/ci_budget.py`](../../scripts/ci_budget.py)：

1. **job 预算**：从 `.github/workflows/ci.yml` 读出每个 job 的 `timeout-minutes`。没写的按 GitHub 的默认值 360。
2. **步骤到 job**：`STEP_JOBS` 写明每个推送前步骤对应的 CI job（即上表）。有测试检查脚本里的每一步都有映射、映射到的 job
   都存在，并且该 job 确实运行这一步的命令。
3. **估算**：一个 job 的 CI 时间估为 runner 的准备秒数，加上这个 job 至今每一步的本地秒数 × 减速系数。系数是同一份代码上
   托管 runner 秒数与笔记本秒数之比（CI run 38077019773 对 `03dd10d` 的推送前运行，38095861905 对 `7cc9a4c`）。门禁、声明、
   verify 和 e2e 在 runner 上慢 2.4–3.1 倍，所以默认系数是 3。pytest 慢 4.2–4.8 倍，因为 CI 加了分支覆盖率，所以它的系数是 5。
4. **判失败**：job 的估算超过 `timeout-minutes` 的 `CI_BUDGET_FRACTION`（默认 0.6）时，越线的那一步记为
   `FAIL (CI time budget)`。估算是累计的，所以越线的那一步即使本身很短也会失败。38095861905 那样每步都快、合起来太慢的
   情况就是这样被拦住的。

汇总里多一列，写每一步所属 job 的估算，最后附一张按 job 的表。表里列出超时、步骤、本地秒数、估算的 CI 分钟数和占预算的比例。

| 变量 | 默认 | 含义 |
|---|---|---|
| `CI_BUDGET_FRACTION` | `0.6` | job 的估算最多可用其 `timeout-minutes` 的比例；`off` 或 `0` 只报告 |
| `CI_SLOWDOWN` | `3` | 没有单独系数的步骤，每本地秒对应的 runner 秒数 |
| `CI_SLOWDOWN_STEPS` | `pytest=5`（内置） | 按步骤的系数，例如 `pytest=5,e2e=2.5`；换到 Apple 芯片笔记本以外的机器上要重新标定 |
| `CI_BUDGET_BUSY_LOAD` | 性能核数（这台 M 系列笔记本为 4），否则为 CPU 数的一半 | 负载平均值超过它时，预算失败会标为「host busy」 |

**机器繁忙会抬高估算。** 耗时是墙钟秒数。系数是在没有其他负载的笔记本上标定的。第 16 轮里，同一台 4 个性能核的笔记本上另有五个评测进程在跑，
pytest 用了 1186 秒，而平时约 550 秒，`tests` job 于是被估成预算的 113%。因此每一步都记录机器的负载平均值（开始时的 1 分钟值，以及结束时
窗口覆盖该步的平均值）。负载高于 `CI_BUDGET_BUSY_LOAD` 时的预算失败仍然算失败，但提示信息和汇总里的「Max host load」列会写明这一点。
先在空闲的机器上重新测量这个 job，再决定相信还是忽略它。

用 `7cc9a4c` 的本地耗时估算，被取消的那种布局下 `agent-eval-gate` 是 48.5 / 45 分钟（108%），`tests` 是 49.7 / 60（83%），
所以两者都会在推送前失败。为了留出余量，第 16 轮把校验器压力测试和红队移到单独的 job `agent-eval-extras`（60 分钟），并把
`tests` 的超时提到 90 分钟。现在的估算是：tests 49.7 / 90（55%），agent-eval-gate 11.8 / 45（26%），agent-eval-extras
28.6 / 60（48%），heldout-verify 10.1 / 45（22%）。`tests/test_ci_budget.py` 检查这两种布局。跳过的步骤不计入，所以用了 `SKIP`
时估算只是下限。

改动前的最后一次运行 38098598453（`c39008b`，heldout-verify 已拆出）通过了，但 `agent-eval-gate` 用了 45 分钟里的 41.8 分钟
（其中校验器和红队一步就是 30.2 分钟），`tests` 用了 60 分钟里的 47.5 分钟。

## 输出

日志写到 `outputs/prepush/<commit>/<step>.log`，汇总写到 `outputs/prepush/<commit>/summary.md`（或 `SUMMARY=<路径>`）：
提交、工作区是否干净、Python / Node / pnpm 版本，以及每一步的命令、结果、秒数、所属 CI job 至今的估算时间与预算，和日志里的结果行（pytest 的通过数、门禁的任务
成功率等），最后是按 job 的预算表。提交到仓库的汇总在 [`docs/results/ci/`](../results/ci/)，按运行时的提交命名：第 14 轮在 `278f7cc` 上（r14-ops 分支，第 14 轮合并之前，所以是 dev 473 / ext 24）17 个步骤全部通过
（[`prepush-278f7cc.md`](../results/ci/prepush-278f7cc.md)：2508 个测试通过 / 70 个跳过，vitest 152 个，e2e 2 个，门禁 dev 1.000、
holdout 0.9434、ext 1.000、shipped 1.000）。

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
