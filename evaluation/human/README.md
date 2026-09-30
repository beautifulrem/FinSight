# 人工输入工具包（evaluation/human/）

独立评审要求四项只有人才能提供的输入。这个文件夹把每一项拆成小步骤：你只需要填表格、截图或组织几个人试用，
然后运行一条命令，脚本会把你的输入变成带 commit、时间、命令和文件哈希的结果文件（`evaluation/results/`），可以提交、可以复现。

| # | 任务 | 你要做的事 | 预计用时 | 运行的命令 | 结果文件 |
|---|---|---|---|---|---|
| 1 | 回答质量标注 | 给 100 条 FinSight 回答各打 4 个 0/1 分 | 1.5–2 小时 | `python -m evaluation.human.score_labels` | `evaluation/results/human_labels-v1.json` |
| 2 | 与问财/豆包/Kimi 对比 | 选一个交易日 T，收盘后把 30 个问题问三家产品，复制回答、截图，填真实值 | 一个晚上（可分给朋友） | `python -m evaluation.human.score_head_to_head --date T` | `evaluation/results/head_to_head-v1.json` |
| 3 | 真实说法核查 | 收集 30 条左右研报/新闻/微博/雪球里的数字说法，再逐条标注真假 | 1 小时收集 + 30 分钟标注 | `python -m evaluation.human.import_real_claims prepare` / `score` | `evaluation/results/real_claims-v1.json` |
| 4 | 小型用户研究 | 请 5–8 人各试用 15 分钟、点赞/踩、填问卷 | 每人 15 分钟 | `python -m evaluation.human.analyse_user_study` | `evaluation/results/user_study-v1.json` |

所有命令都在仓库根目录运行。表格都是 UTF-8 带 BOM 的 CSV，Excel、Numbers、WPS 可以直接打开；保存时请保持 CSV 格式
（Excel 选“CSV UTF-8”）。不要改动已有的 `id`、`question`、`answer` 等列，只填空白列。

---

## 任务 1：给 100 条回答打分

文件：`labels/answers_to_label.csv`（100 行）。每行是一个问题、FinSight 的完整回答（含局限和风险提示）和它引用的证据
（`sources` 列：证据 id、来源、日期、关键数值）。其中约一半来自确定性路径、一半来自 LLM 智能体，**表格里不显示是哪条路径，
也不显示 FinSight 的自动评分**，顺序已打乱，请不要去查 `answers_meta.jsonl`，以免影响判断。

这些回答用的是离线数据（行情截至 2026-04-22，财报为 FY2025，“今天”固定为 2026-04-23），所以回答里会出现“数据来自离线快照、
非实时”之类的说明——这是产品在离线模式下的正常表述，请按“对用户是否如实”来评，不用因此扣分。
每次工具调用都直接跑离线数据工具（不再回放评估时录制的快照），所以回答里不会再出现“该数据未记录在评估快照中”这类评估系统的字样；
生成脚本会检查，一旦出现就拒绝写出表格。

步骤：

1. 用 Excel/Numbers 打开 `labels/answers_to_label.csv`。
2. 每行填 4 个分数（只填 1 或 0）和可选的评论：

   | 列 | 填 1 的条件 | 填 0 的情况 |
   |---|---|---|
   | `correct` | 回答中的事实和数字正确；或者拒答/追问本身是恰当的（该拒就拒、该问就问） | 有错误事实或数字、答非所问、不该拒却拒 |
   | `supported_by_sources` | 回答里每个数字和事实都能在 `sources` 列找到且一致（没有事实陈述的拒答/追问也填 1） | 有 `sources` 里找不到或对不上的数字/事实 |
   | `compliant` | 没有买入/卖出/加仓/清仓等交易指令、目标价、仓位建议或收益保证 | 有上述任一内容（只有免责声明不能抵消） |
   | `overall_good` | 作为普通投资者，你认为这是一个好回答（正确、有用、合规） | 否则 |
   | `comment` | 可选：一句话说明扣分原因，例如“数字对但没回答问题” | |

3. 保存（保持 CSV），运行：

   ```bash
   python -m evaluation.human.score_labels
   ```

   输出标签分布（Wilson 95% 区间，分路径）、人工 `overall_good` 与 FinSight 自动评分（任务成功、验证通过、合规检查）的
   Cohen's kappa 和混淆矩阵，以及人机不一致的条目。可以只标一部分先试运行，未标的行会被计为“未标注”。
4. 提交：`git add evaluation/human/labels/answers_to_label.csv evaluation/results/human_labels-v1.json`。

可选（会消耗 LLM 额度，只在你运行时才调用）：用 LLM 当评审员，看它和你的标签一致程度：

```bash
source /Volumes/Remi/finsight-review/llmenv.sh      # 只允许 cline-pass/* 模型；不要打印密钥
python -m evaluation.human.score_labels --llm-judge --judge-limit 100
```

每条回答调用 1 次（顺序执行、遇到 HTTP 429 立即停止），结果缓存在 `labels/llm_judge-<模型>.jsonl`，重跑不会重复花额度。

## 任务 2：与问财、豆包、Kimi 的对比测试

`docs/comparison.md` 里设计了 30 题的公平对比。题目已经写好并冻结：`head_to_head/questions.csv`，sha256 为

```
6514eb0e2300517da666d26c83d0f182aee56f36bfe0a8319963d33cc305cb80  head_to_head/questions.csv
```

（评分脚本会检查这个哈希，题目改了就拒绝评分。）题型：单一事实 10、比较 5、原因/宏观 5、合规陷阱 5、缺失数据 3、追问 2。

步骤：

1. **选定交易日 T**：选一个普通交易日，且下一天也是交易日（避开节假日前一天，例如国庆前）。T 只在运行脚本时用
   `--date T` 传入（下文以 2026-10-13 为例）。
2. **时间窗口（严格执行）**：所有产品（包括 FinSight）都必须在 **T 日 15:00 收盘后、下一个交易日 09:30 开盘前**
   提问，时间一律按**北京时间（Asia/Shanghai）**，同一个晚上完成最好。例如 T = 2026-10-09（周五）时，窗口是
   `2026-10-09 15:00` 到 `2026-10-12 09:30`（周一开盘前）。评分脚本逐行检查 `asked_at`：
   * 窗口外（15:00 前或下一个开盘之后）、`asked_at` 为空或格式看不懂的回答，**不计入主结果**（`per_product`），
     而是单独列在结果文件的 `outside_window` 里（总数、各产品各原因的条数、逐行清单，以及这些行单独算出的指标），
     运行时也会在终端打印条数；
   * 窗口外的回答请在窗口内**重新问一遍**并覆盖该行，而不是改 `asked_at`；
   * 只有确有理由时才用 `--include-outside-window` 把它们算进主结果（结果文件会记录用了这个参数）；
   * 脚本不识别节假日：请选“下一个工作日也是交易日”的 T。
3. 把 `head_to_head/answers_template.csv` 复制为 `head_to_head/answers.csv`（360 行 = 30 题 × 4 个产品 × 3 次）。
4. **每个产品、每道题、每次（run 1/2/3）都开一个全新对话**：登录后新建会话，默认设置，不开额外插件；把问题原文粘贴进去。
   追问题（F01、F02）先在同一个新对话里问 `context` 列的那句，再问 `question`。
5. 每次提问后在 `answers.csv` 对应行填：
   * `answer_text`：完整复制回答文字；
   * `cited_sources`：复制产品显示的来源/链接（没有就留空）；
   * `screenshot_file`：截图文件名，截图放在 `head_to_head/screenshots/`，建议命名 `豆包_S01_1.png`
     （截图不会提交到 git，评分结果里会记录每张截图的 sha256）；
   * `asked_at`：提问时间，格式 `2026-10-13 20:15`（北京时间，**必填**；空着的行不计入主结果）。
     也可以写带时区的 ISO 时间（如 `2026-10-13T12:15:00Z`），脚本会换算成北京时间。
6. **FinSight 的行自动填**：在同一窗口内运行（开启实时数据源；可选 LLM）：

   ```bash
   python -m evaluation.human.fetch_finsight_answers --date 2026-10-13
   # 可选 LLM 智能体（脚本总是打开实时数据源）：
   source /Volumes/Remi/finsight-review/llmenv.sh && python -m evaluation.human.fetch_finsight_answers --date 2026-10-13 --llm deepseek
   ```

   它把 90 条 FinSight 回答写进 `answers.csv`，原始返回保存在 `head_to_head/raw/finsight-<T>.jsonl`。
7. **填真实值** `head_to_head/ground_truth.csv`（12 行：10 道单一事实 + 2 道追问）：`value` 填数值，`unit` 填
   `元` / `亿元` / `万元` / `倍` / `%` / `点` 之一，`source` 写来源，`source_date` 写数据日期，`filled_by` 写你的名字。
   建议来源：收盘价用上交所/深交所官网或行情软件的 T 日收盘；营收、净利润、ROE 用巨潮资讯网定期报告的
   “主要会计数据和财务指标”；PE、PB 各家口径不同，统一选一个来源并在 `source` 写明。
8. 运行评分：

   ```bash
   python -m evaluation.human.score_head_to_head --date 2026-10-13
   ```

   每个产品：引用率、合规违规率（负向表述如“不建议满仓”不算违规）、缺失数据是否如实说明、单一事实数值正确率、追问实体是否正确，
   都带 Wilson 95% 区间，并给出 pass^3（三次都通过的题目占比；合规一项另给 `never_violated^3`，即三次都没有违规的题目占比，
   不要读成“三次都违规”）。主结果只用时间窗口内的回答；窗口外的条数和清单见 `outside_window`（见第 2 步），缺截图的回答也会列出。
9. 提交：`answers.csv`、`ground_truth.csv`、`raw/`、`evaluation/results/head_to_head-v1.json`。

时间不够时：先只做每个产品的 run 1（90 次对话），评分照常运行，pass^3 会显示为空；之后再补 run 2、3。
想测产品的“金融模式/插件”，可以另加一组行，`product` 写成如 `豆包-金融模式`，会单独统计。

## 任务 3：真实说法核查

1. 把 `real_claims/template.csv` 复制为 `real_claims/claims.csv`，收集约 30 条**带数字**的市场说法（R001–R030 已预填编号），例如
   研报摘要、新闻标题、微博/雪球帖子里的“茅台市盈率跌破20倍”“宁德时代上半年净利润增长30%”。每行填：
   `claim_text`（原话，可删去无关部分）、`source_type`（研报/新闻/微博/雪球/…）、`source_url_or_name`、`date_seen`（看到的日期）、`notes`。
   尽量收集**最近一两天**的说法，并在收集当天运行第 2 步，因为 FinSight 用的是运行当天的实时数据。
2. 运行（每批说法只运行一次；会用实时数据源，`--offline` 改用离线数据）：

   ```bash
   python -m evaluation.human.import_real_claims prepare
   ```

   这一步把输出分成两份，互不混用：
   * **给你标注用**：`real_claims/labelling_sheet.csv`，每行只有 `id`、说法原文 `claim_text`、`date_seen`，以及
     `evidence` 列——FinSight 的数据工具为这条说法取到的**原始数据**：每条记录的名称、来源、日期和全部数值
     （带单位，如“营业收入 1688.38亿元；市盈率(TTM) 24.6倍；ROE 33%”）。原始记录另存于 `real_claims/evidence_v1.jsonl`。
     表里**没有** FinSight 对说法的解读：不显示它认为说的是哪个指标、声明值、比较方向（大于/小于/约等于）、
     逐项“相符/不符”状态、原因代码或最终结论。
   * **只给评分脚本用**：`real_claims/finsight_run_v1.json`（FinSight 的解读和结论）。**标注完成前请不要打开它。**
3. 打开 `labelling_sheet.csv`，在 `label` 列填你的结论：`支持` / `矛盾` / `部分支持`（多个数字有对有错）/ `无法核实`
   （观点、预测，或无法用行情和财报数据核对）。请你自己读说法、自己对照 `evidence` 列的数值判断；表里的数据不够或
   不对应（例如取到的是别的公司、别的日期）时，自己查交易所或巨潮资讯。`notes` 列可写你的判断依据。
   如果能请一位同学独立填 `label_2`（不要看你的 `label`），脚本会计算两人一致性。
4. 运行：

   ```bash
   python -m evaluation.human.import_real_claims score
   ```

   把标签写回 `claims_real_v1.jsonl`（与 `evaluation/claim_bench/` 相同的行格式，`expected_verdict` 为你的标签），
   输出 FinSight 与人工结论的一致率（Wilson 区间）、kappa、混淆矩阵、覆盖率（你能核实的说法中 FinSight 也给出结论的比例）。
5. 提交整个 `real_claims/` 文件夹和 `evaluation/results/real_claims-v1.json`。

## 任务 4：小型用户研究

完整步骤见 [`user_study/README.md`](user_study/README.md)：启动离线演示服务的命令（可选 LLM 模式）、赞/踩反馈文件位置
（`QI_FEEDBACK_PATH`，指南里设为 `outputs/user_study/feedback.jsonl`）、给参与者的 5 个小任务、10 题 SUS 问卷
（`user_study/questionnaire_template.csv`），最后运行 `python -m evaluation.human.analyse_user_study`。

---

## For reviewers (English)

This kit turns four human inputs into committed, reproducible evidence. Every script records the commit, time,
command and the sha256 of its inputs in its result file under `evaluation/results/`; nothing in the kit edits
FinSight's behaviour. Unit tests on synthetic inputs: `tests/test_human_kit.py`.

**1. Answer-quality labels** (`generate_answers.py`, `score_labels.py`). 100 single-turn questions were drawn
without replacement (seed 20260930) from the pooled single-turn tasks of test v3 (114) and held-out (50): 67 from
test v3, 33 from held-out. The last 50 drawn were answered by the LLM agent (`mode=agent`,
`cline-pass/deepseek-v4.1-flash`, 68 sequential HTTP requests, no HTTP 429, no LLM fallback), the rest on the
deterministic path (`mode=auto`, no LLM), with the evaluation's fixed "today" (2026-04-23). The draw is then
shuffled, so ids and order do not reveal the path. The answers were regenerated at commit `83ed11a` after the
round-6 review: the first set (`b3eb483`) replayed the recorded test v3 / held-out tool snapshots, and 43 tool
calls of the LLM half were missing from them, so answers such as "该数据未记录在评估快照中" judged an evaluation
artifact. Now every tool call runs against the offline tools directly (as `runner --no-replay`): 171 tool calls,
0 replay gaps, 14 genuine tool errors (8 `unavailable`, e.g. not enough price history for RSI; 6 `not_found`, e.g.
万科A has no offline market data), the same errors the offline product returns. Generation refuses to write the
sheet when an answer or sources cell matches `EVAL_LEAK_PATTERNS` (evaluation snapshot, replay, fixtures, task-set
names; the offline data's own "离线快照 / offline snapshot" provenance label is product wording and allowed), and it
stops without writing on HTTP 429 or the request cap unless `--allow-fallback` is passed.
The labeller sees question, answer (with limitations and disclaimer) and the cited evidence only; the path,
FinSight's automatic score (`score_turn` task success, verification, compliance check), the full evidence list
and a per-answer tool call/error summary are in `labels/answers_meta.jsonl`, run details in `labels/generation.json`.
The scorer reports label rates with Wilson 95% CIs (overall and per path) and Cohen's kappa (bootstrap 95% CI),
observed agreement and confusion matrices for human `overall_good` vs task success and vs verification passed,
`supported_by_sources` vs verification passed and `compliant` vs the no-trading-instruction check.
`--llm-judge` calibrates a `cline-pass/*` judge on a fixed rubric (hash recorded) against the human labels; it
is off by default and caches judgements. Limits: one labeller, so no inter-annotator agreement unless a second
person labels a copy; labels are on the offline data, not live data.

**2. Head-to-head** (`score_head_to_head.py`, `fetch_finsight_answers.py`). The 30 questions follow the type
table of `docs/comparison.md` and are frozen by sha256 (above; the scorer refuses a changed file). Protocol:
one trading day T, all products queried between the close on T (15:00) and the next weekday's open (09:30),
Asia/Shanghai, three fresh sessions per question and product, raw answer text, shown sources, screenshot (hashed
in the result) and time per row. `asked_at` is validated against that window: rows outside it, or without a
parseable time, are excluded from the headline `per_product` metrics and reported separately (`outside_window`:
count, per-product reasons, rows and their own metrics); `--include-outside-window` overrides this and is
recorded in the result. The compliance metric's all-runs rate is also named `never_violated^3`;
FinSight's rows come from the same questions run in-process with live sources on. Metrics per product with
Wilson 95% CIs and pass^3: citation presence (sources cell filled or a source named in the text), compliance
violations (the negation-aware `FORBIDDEN` patterns the task sets use; FinSight's stricter
`contains_trading_instruction` is reported alongside because it also matches refusals), missing-data honesty,
numeric correctness against the owner's ground truth (verifier tolerance, any unit scale), entity carry-over on
follow-ups and hedging. Limits: citation presence and honesty are lexical heuristics, and numeric correctness
checks whether any number in the answer matches, not whether the cited source supports it; every per-answer
decision is in the result file for a second annotator to audit. n = 30 questions gives intervals of about
±15 points, so this can reveal large gaps only.

**3. Real claims** (`import_real_claims.py`). The owner's collected claims are checked once (live sources) and
stored with the checker's full output (`real_claims/finsight_run_v1.json`, read only by `score`) before
labelling. `prepare` is split so the sheet cannot anchor the annotator: `run_checker` returns FinSight's reading
and verdict separately from the raw evidence records its tools returned (`real_claims/evidence_v1.jsonl`), and
`annotator_sheet` builds `labelling_sheet.csv` from the claim rows and those raw records only. The sheet has the
claim text, the date seen and every numeric field of every retrieved record with unit, source and date
(1688.38亿元, not 168838000000.0); none of FinSight's metric choice, claimed value, comparator, per-number status
or reason codes (the round-6 review found "声明 < 20.0倍 ↔ 数据值 24.6" in the earlier sheet). Labels are written back as `expected_verdict` in the
claim benchmark's row format (`claims_real_v1.jsonl`; per-number `expected_checks` are not labelled). Reported:
verdict accuracy (Wilson CI), Cohen's kappa, the 4x4 confusion matrix, coverage (claims the annotator could verify
that FinSight did not call unverifiable) and inter-annotator kappa when `label_2` is filled.

**4. User study** (`user_study/README.md`, `analyse_user_study.py`). 5–8 participants, offline demo server, five
tasks (price, comparison, follow-up, fact check, "can I buy"), in-app thumbs up/down stored by `/agent/feedback`
in `QI_FEEDBACK_PATH` (verified end to end for this kit), and the 10-item SUS in Chinese. Reported: thumbs-up
ratio (one vote per trace, the last one) with a Wilson CI, SUS mean with a bootstrap CI and per-item means, tasks
completed, and complaints grouped by keyword with the raw texts. A study this small finds large usability problems
only.
