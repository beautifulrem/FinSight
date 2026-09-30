# 小型用户研究：操作指南

目标：请 5–8 位参与者（最好有人炒股、有人不炒股）各用 FinSight 做 5 个小任务（约 15 分钟），
每条回答点赞或点踩，最后填一份 10 题的 SUS 可用性问卷。产出：点赞率（带 95% 置信区间）、SUS 分数、主要吐槽。

你需要准备：一台能运行本仓库的电脑、一个浏览器、本文件夹里的问卷模板。

## 1. 启动演示服务（每次研究开始前）

在仓库根目录执行（macOS / Linux）：

```bash
cd /path/to/FinSight
export QI_USE_LIVE_MARKET=0 QI_USE_LIVE_NEWS=0 QI_USE_LIVE_ANNOUNCEMENT=0 QI_USE_LIVE_MACRO=0   # 用仓库自带的离线快照
export QI_FEEDBACK_PATH="$PWD/outputs/user_study/feedback.jsonl"   # 赞/踩写到这个文件
unset DEEPSEEK_API_KEY                                               # 离线确定性模式，不调用 LLM
uvicorn query_intelligence.api.app:create_app --factory --host 127.0.0.1 --port 8765
```

看到 `Uvicorn running on http://127.0.0.1:8765` 后，用浏览器打开 <http://127.0.0.1:8765>。

Windows PowerShell 写法：

```powershell
$env:QI_USE_LIVE_MARKET="0"; $env:QI_USE_LIVE_NEWS="0"; $env:QI_USE_LIVE_ANNOUNCEMENT="0"; $env:QI_USE_LIVE_MACRO="0"
$env:QI_FEEDBACK_PATH="$PWD\outputs\user_study\feedback.jsonl"
Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
uvicorn query_intelligence.api.app:create_app --factory --host 127.0.0.1 --port 8765
```

**可选：LLM 模式。** 想让参与者体验 LLM 智能体，在启动前额外设置（只能用 `cline-pass/*` 模型）：

```bash
export DEEPSEEK_API_KEY=...                              # 不要写进任何文件或截图
export DEEPSEEK_BASE_URL=https://api.cline.bot/api/v1
export DEEPSEEK_MODEL=cline-pass/deepseek-v4.1-flash
```

（本机也可以直接 `source /Volumes/Remi/finsight-review/llmenv.sh`，它会设置同样的变量；之后仍要设置上面的
`QI_FEEDBACK_PATH` 再启动。）然后在页面的模式切换处选「自动」或「LLM 智能体」。每个问题约用 1–3 次 LLM 调用，
8 人 × 5 个任务大约 40–120 次。一次研究里所有参与者用同一种模式，并在问卷的 `comment` 或你的记录里写明模式。

**离线数据只覆盖这些标的**（截至 2026-04-22 的快照）：贵州茅台、五粮液、中国平安、沪深300ETF、沪深300 指数、
创业板ETF、证券ETF，以及 CPI、PMI、M2 等宏观指标。下面的任务都在这个范围内。页面上会提示数据不是实时的，这是正常的。

## 2. 赞/踩反馈存在哪里

* 每条回答下方有「赞」「踩」两个按钮（大拇指图标）。点「踩」会自动展开评论框，请参与者写一句原因。
* 服务端收到后，**追加一行 JSON** 到 `QI_FEEDBACK_PATH` 指定的文件。按上面的命令启动时就是
  `outputs/user_study/feedback.jsonl`（仓库根目录下；`outputs/` 不会被 git 提交）。
  如果启动时没有设置 `QI_FEEDBACK_PATH`，默认写到**启动服务时所在目录**下的 `outputs/feedback/feedback.jsonl`。
* 每行包含：时间、trace_id、赞/踩、评论、问题原文、路由、浏览器的匿名 id（哈希，不含个人信息）。
* **研究进行中不要重启服务。** 反馈要靠服务端内存里的运行记录匹配；重启后再点旧回答的赞/踩，页面会提示
  “服务端已找不到这次运行，反馈已保存在本浏览器”，这条反馈就不会写进文件。
* 自检：启动后自己问一个问题并点一次赞，然后 `tail -1 outputs/user_study/feedback.jsonl` 应该能看到这一行。
  正式开始前可以删掉这行，或在分析时用 `--since` 只统计正式开始之后的反馈。

## 3. 每位参与者的流程（约 15 分钟）

1. 打开一个**新的无痕/隐私窗口**访问 <http://127.0.0.1:8765>（这样每人是独立的匿名身份和会话）。
   记下参与者编号（P01、P02…）和开始时间。
2. 读给参与者听：“这是一个查询 A 股数据的研究助手。请完成下面 5 个小任务，可以用自己的话提问。
   每看完一个回答，请点赞或点踩；点踩时写一句原因。我不会帮你操作，想到什么可以说出来。”
3. 参与者依次完成 5 个任务（可以打印下面的任务卡）。你只观察和记录，不提示。
4. 记录参与者**自己认为完成了**几个任务（0–5），填到问卷的 `tasks_completed`。
5. 请参与者填写 SUS 问卷（第 4 节），再问两个开放问题：最喜欢什么、最不满意什么。

### 任务卡（给参与者）

1. **问价格**：问一下贵州茅台最新的收盘价。
2. **比较**：比较茅台和五粮液的市盈率，哪个更高？
3. **追问**：在同一个对话里接着问，比如“那它们的 ROE 呢？”或“那五粮液的市净率呢？”（不要重复写公司名）。
4. **核查一个说法**：有人说“中国平安市盈率只有 5 倍”。用页面上的「核查」页检查这句话（或者在对话里问
   “听说中国平安市盈率只有5倍，是真的吗？”）。
5. **能不能买**：问“现在能买五粮液吗？”，看看它怎么回答。

## 4. SUS 问卷

把 `questionnaire_template.csv` 复制为 `questionnaire.csv`，每位参与者一行（已预填 P01–P10，不用的行留空即可）。
每题 1–5 分：1 = 非常不同意，2 = 不同意，3 = 一般，4 = 同意，5 = 非常同意。也可以打印纸质版，事后录入。

| 题号 | 题目 |
|---|---|
| q1 | 我愿意经常使用 FinSight |
| q2 | 我觉得 FinSight 没必要这么复杂 |
| q3 | 我觉得 FinSight 用起来很容易 |
| q4 | 我觉得需要有懂技术的人帮忙才能用 FinSight |
| q5 | 我觉得 FinSight 的各项功能整合得很好 |
| q6 | 我觉得 FinSight 有太多前后不一致的地方 |
| q7 | 我觉得大多数人能很快学会用 FinSight |
| q8 | 我觉得 FinSight 用起来很别扭 |
| q9 | 用 FinSight 时我很有信心 |
| q10 | 用 FinSight 之前我需要先学很多东西 |

其他列：`date`（日期）、`invests_in_stocks`（是/否）、`tasks_completed`（0–5）、`most_liked`、`most_disliked`、
`comment`（可写使用的模式，如“离线确定性”或“LLM 智能体”）。

## 5. 分析与提交

```bash
python -m evaluation.human.analyse_user_study
# 只统计正式开始后的反馈：
python -m evaluation.human.analyse_user_study --since 2026-10-10T06:00   # UTC 时间，北京时间减 8 小时
```

输出 `evaluation/results/user_study-v1.json`：点赞率与 Wilson 95% 区间、按路由拆分、SUS 平均分（bootstrap 95% 区间，
68 分是行业平均线）、每题均分、平均完成任务数、按关键词归类的主要吐槽（附原话）。

提交：`git add evaluation/human/user_study/questionnaire.csv evaluation/results/user_study-v1.json`。
反馈原始文件在 `outputs/`（不提交）；结果 JSON 里会记录它的 sha256，以及吐槽原话。提交前检查评论里没有姓名、手机号等个人信息。

注意：5–8 人的研究只能发现明显的可用性问题，置信区间会很宽，报告时请照实写出人数。
