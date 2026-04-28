# FinSight / ARIN Query Intelligence 使用的机器学习方法

本文按模块列出项目中实际使用的机器学习方法。重点区分两类内容：核心 Query Intelligence 后端中的传统、可解释机器学习方法，以及下游模块中的 Transformer / LLM 例外。

## 1. 总体边界

FinSight 中的机器学习方法可以分为两类：

- **核心 Query Intelligence 后端**：以传统、可解释机器学习和稀疏特征为主。覆盖 NLU、检索排序、数据源规划、实体边界识别、错别字链接、澄清判断等模块。
- **下游例外模块**：只有在系统已经生成紧凑证据之后，才允许使用 Transformer 或 LLM。主要包括文档情感分析，以及最终回答和追问生成。

核心 NLU / Retrieval 主链路不使用 BERT、Transformer embedding、向量数据库或 LLM 推理作为 backbone。

## 2. NLU 模块

代码路径：`query_intelligence/nlu/`、`training/train_*.py`、`models/*.joblib`

| 子模块 | 机器学习方法 | 特征 | 输出 artifact | 作用 |
|---|---|---|---|---|
| 产品类型分类器 | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` | word n-gram `(1,2)`、char n-gram `(2,4)`、查询标记 token | `models/product_type.joblib` | 预测 `stock`、`etf`、`fund`、`index`、`macro`、`generic_market`、`unknown` 或 `out_of_scope`。 |
| 意图分类器 | 多标签 `MultiOutputClassifier(SGDClassifier(loss="log_loss"))` | 与产品类型分类器相同的稀疏 n-gram 特征 | `models/intent_ovr.joblib` | 预测一个或多个意图，例如 `market_explanation`、`hold_judgment`、`buy_sell_timing`、`fundamental_analysis`、`event_news_query`。 |
| 主题分类器 | 多标签 `MultiOutputClassifier(SGDClassifier(loss="log_loss"))` | 与产品类型分类器相同的稀疏 n-gram 特征 | `models/topic_ovr.joblib` | 预测所需证据主题，例如 `price`、`news`、`industry`、`macro`、`fundamentals`、`valuation`、`risk`。 |
| 问题类型分类器 | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` | word / char n-gram 稀疏文本特征 | `models/question_style.joblib` | 预测 `fact`、`why`、`compare`、`advice` 或 `forecast`。 |
| 用户情绪 / 语气分类器 | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` | word / char n-gram 稀疏文本特征 | `models/sentiment.joblib` | 预测用户语气，例如 `neutral`、`bullish`、`bearish`、`anxious`。这是 NLU 中的用户语气识别，不是文档情感分析。 |
| 越界检测器 | `HashingVectorizer` + 带 class weight 的 `SGDClassifier(loss="log_loss")` | word n-gram、char n-gram、金融锚点标记、工具类任务标记、对话标记 | `models/out_of_scope_detector.joblib` | 识别并拒绝非金融问题，例如天气、编程、笑话、写诗等。 |
| 实体边界识别 | `sklearn_crfsuite.CRF` 线性链 CRF，优化算法为 `lbfgs` | 字符级特征：当前字符、前后字符、数字/字母/大写/空格标记、BOS/EOS | `models/entity_crf.joblib` | 在中英文 query 中识别实体 mention span，标签为 `B-ENT / I-ENT / O`。 |
| 实体链接 / 候选实体排序 | 基于人工数值特征的 `SGDClassifier(loss="log_loss")` | fuzzy match 分数、partial ratio、token overlap、symbol hit、标准名称/行业名称特征 | 运行时基于实体 catalog 构建；`models/` 中没有单独 shipped artifact | 在 alias 或 fuzzy matching 之后对候选实体排序；数据不足时退回 heuristic 排序。 |
| 错别字链接器 | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | Levenshtein similarity、RapidFuzz ratio、partial ratio、长度差、前后缀匹配、ETF/LOF 标记、heuristic score | `models/typo_linker.joblib` | 将错别字或变体 mention 链接到已知 alias。 |
| 澄清 gate | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | 产品类型、时间范围、实体数量、比较对象数量、intent/topic 标记、marker 特征 | `models/clarification_gate.joblib` | 判断 query 是否缺少关键信息，是否需要触发澄清。 |
| 问题类型 reranker | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | 初始问题类型、产品类型、intent/topic 标记、实体数量、compare/why/advice/forecast marker | `models/question_style_reranker.joblib` | 在 heuristic 或初始分类之后修正 question style。 |
| Source plan reranker | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | 候选 source、规则 source plan 中的 rank、产品类型、时间范围、intent/topic 标记、compare/why/advice marker | `models/source_plan_reranker.joblib` | 对候选证据源打分和重排，例如 `market_api`、`news`、`announcement`、`fundamental_sql`、`macro_sql`、`faq`。 |

NLU 中的辅助非 ML 方法：

- Query normalization 规则。
- Alias table、精确匹配和 fuzzy matching。
- 风险标记与默认 source plan 的规则逻辑。
- 基于 marker 的 question-style heuristic。

## 3. Retrieval 模块

代码路径：`query_intelligence/retrieval/`、`training/train_ranker.py`

| 子模块 | ML / 统计方法 | 特征 | 输出 artifact | 作用 |
|---|---|---|---|---|
| 文档检索 | `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` + `linear_kernel` 相似度 | title、summary、body 的字符级 TF-IDF；query bundle 中的 normalized query、实体、关键词和 symbol | `DocumentRetriever` 运行时对象 | 检索候选新闻、公告、研报、FAQ 和产品文档。 |
| 文档特征构建 | 人工特征工程 | lexical score、trigram similarity、实体匹配、alias 匹配、title hit、关键词覆盖、intent/source 兼容性、topic/content 兼容性、产品匹配、可信度、时效性、公告标记、长度、时间窗口匹配、ticker hit | 运行时 feature JSON | 将候选 evidence 转换成可解释排序特征。 |
| 文档 ranker | `StandardScaler` + `SGDClassifier(loss="log_loss")`，通过 `MLRanker` 使用 | 上述人工排序特征 | `models/ranker.joblib` 和 `models/ranker.json` | 预测候选 evidence 的相关性概率。 |
| Hybrid ranker | 规则 baseline 与 ML ranker 的加权融合 | baseline score + ML probability | `HybridRanker` 运行时对象 | 将稳定的规则排序与学习到的相关性分数结合；当前有 ML ranker 时 baseline 权重为 0.35。 |
| 去重 / 聚类 | `TfidfVectorizer(char_wb, 2-4)` + cosine similarity | 候选文档的 title 和 summary 文本 | `Deduper` 运行时对象 | 对相似度超过阈值的 near-duplicate evidence 进行聚类或删除。 |

Retrieval 中的辅助非 ML 方法：

- 从 NLU artifact 构建 query bundle。
- 按 `source_plan` 过滤数据源。
- 对 entity-bound 文档进行 symbol 过滤。
- 基于规则的 term-hit bonus 和 source diversity selection。
- 调用结构化 provider 获取 market、fundamental、industry、announcement、macro 数据。

## 4. Market Analysis 模块

代码路径：`query_intelligence/retrieval/market_analyzer.py`

该模块主要是**统计分析**，不是 supervised ML。

使用的方法包括：

- 价格收益率计算。
- 移动均线，例如 MA5、MA20。
- RSI 风格的动量信号。
- MACD 风格的趋势信号。
- 波动率计算。
- Bollinger band 风格的区间信号。
- PE、PB、ROE 等估值和基本面摘要。
- 对 market、fundamental、macro、news、structured evidence 的 data readiness 检查。

作用：

- 生成 `analysis_summary`。
- 总结证据信号，但不做确定性买卖建议。

## 5. 文档情感分析模块

代码路径：`sentiment/preprocessor.py`、`sentiment/classifier.py`

| 子模块 | 机器学习方法 | 模型 | 作用 |
|---|---|---|---|
| 按语言路由的文档情感分析 | Transformer sequence classification，FinBERT 系列模型 | 英文：`ProsusAI/finbert`；中文：`yiyanghkust/finbert-tone-chinese` | 将检索到的文档分类为 `positive`、`negative` 或 `neutral`。 |
| 逐句推理 | FinBERT inference + softmax probability | `AutoTokenizer` + `AutoModelForSequenceClassification` | 对每个句子分别打分，更符合 FinBERT 的句级训练分布。 |
| 文档级聚合 | majority label + 平均 sentiment score / confidence | 对句子预测结果做规则聚合 | 生成文档级 `SentimentItem`。 |
| 混合/未知语言处理 | 同时运行中文和英文 FinBERT，并平均概率 | 两个 FinBERT 模型 | 处理双语或语言不确定的文档。 |

重要边界：

- 这是下游模块，消费 Query Intelligence artifact 和检索出的文档。
- 它不替代 NLU 的实体识别、意图分类或 retrieval ranking。

## 6. 回答生成和追问生成模块

代码路径：`scripts/llm_response.py`

该模块使用 LLM，但只在系统已经生成紧凑证据之后使用。

| 子模块 | 机器学习方法 | 后端选项 | 作用 |
|---|---|---|---|
| 回答生成 | Chat LLM generation | 默认 OpenAI-compatible backend：`deepseek-v4-flash`；也支持其他 OpenAI-compatible provider、Anthropic Messages API、本地 HuggingFace `AutoModelForCausalLM` | 基于紧凑证据生成前端可用的 `answer_generation` JSON。 |
| 追问生成 | Chat LLM generation | 与回答生成相同的后端体系 | 生成 `next_question_prediction` JSON 和后续问题建议。 |
| 本地模型模式 | Causal language model inference | `AutoTokenizer` + `AutoModelForCausalLM`；默认示例包括 finance Llama / Qwen 风格模型 | 在配置本地模型时支持本地 transformer 推理。 |
| JSON 校验和 grounding guard | 确定性校验，不是 ML | schema 检查、evidence ID 检查、免责声明强制等 | 防止无依据或格式错误的结果传给前端。 |

重要边界：

- LLM 不负责 intent detection、entity linking、source planning、retrieval 或 market analysis。
- LLM 必须基于 compact evidence 回答，不能编造缺失的 market、fundamental、macro、news、sentiment 或 statistical facts。

## 7. Training 模块

代码路径：`training/`

训练脚本负责生成随仓库发布的 `models/*.joblib` artifacts。

| 训练脚本 | 训练的方法 | 输出 |
|---|---|---|
| `training.train_product_type` | 稀疏 n-gram 文本特征 + 线性 logistic SGD 分类器 | `models/product_type.joblib` |
| `training.train_intent` | 多标签稀疏 n-gram 分类器 + one-vs-rest 风格线性 SGD | `models/intent_ovr.joblib` |
| `training.train_topic` | 多标签稀疏 n-gram 分类器 + one-vs-rest 风格线性 SGD | `models/topic_ovr.joblib` |
| `training.train_question_style` | 稀疏 n-gram 文本特征 + 线性 logistic SGD 分类器 | `models/question_style.joblib` |
| `training.train_sentiment` | 针对用户语气的稀疏 n-gram 文本特征 + 线性 logistic SGD 分类器 | `models/sentiment.joblib` |
| `training.train_entity_crf` | 字符级 CRF | `models/entity_crf.joblib` |
| `training.train_clarification_gate` | 字典特征 + 线性 logistic SGD | `models/clarification_gate.joblib` |
| `training.train_question_style_reranker` | 字典特征 + 线性 logistic SGD | `models/question_style_reranker.joblib` |
| `training.train_source_plan_reranker` | 字典特征 + 线性 logistic SGD | `models/source_plan_reranker.joblib` |
| `training.train_out_of_scope_detector` | 稀疏 n-gram 文本特征 + 线性 logistic SGD 分类器 | `models/out_of_scope_detector.joblib` |
| `training.train_ranker` | 标准化人工排序特征 + 线性 logistic SGD | `models/ranker.joblib` |
| `training.train_typo_linker` | 字典特征 + 线性 logistic SGD | `models/typo_linker.joblib` |

## 8. 总结

项目的核心系统使用传统、可解释机器学习：文本分类主要是稀疏 n-gram 特征加线性 logistic SGD 分类器，实体边界识别使用 CRF，澄清判断、错别字链接、source planning 和排序使用人工特征加线性分类器，文档检索和去重使用 TF-IDF / cosine similarity。Transformer 模型是下游例外：FinBERT 用于文档情感分析，LLM 用于在已有证据基础上生成最终回答和追问 JSON。
