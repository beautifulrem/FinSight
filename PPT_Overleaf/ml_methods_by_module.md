# ML Methods Used in FinSight / ARIN Query Intelligence

This file lists the machine-learning methods used by the project, grouped by module. It separates the classical ML backbone from downstream Transformer or LLM components.

## 1. Overall Boundary

FinSight uses two different categories of ML:

- **Core Query Intelligence backbone**: classical, explainable ML and sparse features. This covers NLU, retrieval ranking, source planning, entity boundary detection, typo linking, and clarification decisions.
- **Downstream exceptions**: Transformer or LLM models are allowed only after compact evidence has already been produced. This covers document sentiment classification and answer/next-question generation.

The core backend does **not** use BERT, Transformer embeddings, vector databases, or LLM reasoning as the main NLU/Retrieval backbone.

## 2. NLU Module

Code paths: `query_intelligence/nlu/`, `training/train_*.py`, `models/*.joblib`

| Submodule | ML method | Features | Output artifact | Purpose |
|---|---|---|---|---|
| Product type classifier | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` | Word n-grams `(1,2)`, char n-grams `(2,4)`, query marker tokens | `models/product_type.joblib` | Predicts `stock`, `etf`, `fund`, `index`, `macro`, `generic_market`, `unknown`, or `out_of_scope`. |
| Intent classifier | Multi-label `MultiOutputClassifier(SGDClassifier(loss="log_loss"))` | Same sparse n-gram feature union as product type | `models/intent_ovr.joblib` | Predicts one or more intents such as `market_explanation`, `hold_judgment`, `buy_sell_timing`, `fundamental_analysis`, and `event_news_query`. |
| Topic classifier | Multi-label `MultiOutputClassifier(SGDClassifier(loss="log_loss"))` | Same sparse n-gram feature union as product type | `models/topic_ovr.joblib` | Predicts evidence topics such as `price`, `news`, `industry`, `macro`, `fundamentals`, `valuation`, and `risk`. |
| Question style classifier | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` | Word/char n-gram sparse text features | `models/question_style.joblib` | Predicts `fact`, `why`, `compare`, `advice`, or `forecast`. |
| User sentiment / tone classifier | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` | Word/char n-gram sparse text features | `models/sentiment.joblib` | Predicts user tone, such as `neutral`, `bullish`, `bearish`, or `anxious`. This is NLU user-tone detection, not document sentiment. |
| Out-of-scope detector | `HashingVectorizer` + `SGDClassifier(loss="log_loss")` with class weights | Word n-grams, char n-grams, finance-anchor markers, utility-task markers, dialogue markers | `models/out_of_scope_detector.joblib` | Rejects non-financial questions, such as weather, coding, jokes, or poems. |
| Entity boundary detection | Linear-chain CRF from `sklearn_crfsuite.CRF` using `lbfgs` | Character-level features: current char, previous/next chars, digit/alpha/upper/space flags, BOS/EOS | `models/entity_crf.joblib` | Detects entity mention spans in Chinese and English queries using `B-ENT / I-ENT / O` labels. |
| Entity linker / candidate ranker | `SGDClassifier(loss="log_loss")` over handcrafted numeric features | Fuzzy match scores, partial ratios, token overlap, symbol hits, canonical-name/industry-name features | Built in runtime from entity catalog; no shipped separate artifact in `models/` | Ranks candidate entities after alias or fuzzy matching. Falls back to heuristic ranking if insufficient data. |
| Typo linker | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | Levenshtein similarity, RapidFuzz ratio, partial ratio, length gap, prefix/suffix match, ETF/LOF flags, heuristic score | `models/typo_linker.joblib` | Links misspelled or variant mentions to known aliases. |
| Clarification gate | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | Product type, time scope, entity count, comparison target count, intent/topic flags, marker features | `models/clarification_gate.joblib` | Decides whether the query lacks enough information and should trigger clarification. |
| Question style reranker | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | Base question style, product type, intent/topic flags, entity count, compare/why/advice/forecast markers | `models/question_style_reranker.joblib` | Corrects question style after the first heuristic/classifier pass. |
| Source plan reranker | `DictVectorizer` + `SGDClassifier(loss="log_loss")` | Candidate source, rule-plan rank, product type, time scope, intent/topic flags, compare/why/advice markers | `models/source_plan_reranker.joblib` | Scores and reorders candidate evidence sources, such as `market_api`, `news`, `announcement`, `fundamental_sql`, `macro_sql`, and `faq`. |

Supporting non-ML methods in NLU:

- Query normalization rules.
- Alias table and exact/fuzzy matching.
- Rule-based risk flags and source-plan defaults.
- Heuristic question-style markers.

## 3. Retrieval Module

Code paths: `query_intelligence/retrieval/`, `training/train_ranker.py`

| Submodule | ML / statistical method | Features | Output artifact | Purpose |
|---|---|---|---|---|
| Document retrieval | `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` + `linear_kernel` similarity | Character-level TF-IDF from title, summary, and body; query bundle text from normalized query, entities, keywords, and symbols | Runtime in `DocumentRetriever` | Retrieves candidate news, announcements, research notes, FAQs, and product documents. |
| Document feature builder | Handcrafted feature engineering | Lexical score, trigram similarity, entity match, alias match, title hit, keyword coverage, intent/source compatibility, topic/content compatibility, product match, credibility, recency, disclosure flag, length, time-window match, ticker hit | Runtime feature JSON | Converts retrieved candidates into explainable ranking features. |
| Document ranker | `StandardScaler` + `SGDClassifier(loss="log_loss")`; used through `MLRanker` | The handcrafted ranking features listed above | `models/ranker.joblib` and `models/ranker.json` | Predicts relevance probability for candidate evidence. |
| Hybrid ranker | Weighted blend of rule baseline and ML ranker | Baseline score + ML probability | Runtime in `HybridRanker` | Combines stable rule ranking with learned relevance. Current default blend gives the baseline part a 0.35 weight when ML ranker is available. |
| Deduplication | `TfidfVectorizer(char_wb, 2-4)` + cosine similarity | Candidate title and summary text | Runtime in `Deduper` | Clusters or removes near-duplicate evidence when similarity is above the threshold. |

Supporting non-ML retrieval methods:

- Query building from NLU artifacts.
- Source filtering by `source_plan`.
- Entity-symbol filtering for entity-bound documents.
- Rule-based term-hit bonus and source diversity selection.
- Structured provider calls for market, fundamental, industry, announcement, and macro data.

## 4. Market Analysis Module

Code path: `query_intelligence/retrieval/market_analyzer.py`

This module is mostly **statistical analysis**, not supervised ML.

Methods used:

- Price return calculation.
- Moving averages, such as MA5 and MA20.
- RSI-style momentum signal.
- MACD-style trend signal.
- Volatility calculation.
- Bollinger-band-style range signal.
- Valuation and fundamental summaries, such as PE, PB, and ROE.
- Data readiness checks across market, fundamental, macro, news, and structured evidence.

Purpose:

- Produce `analysis_summary`.
- Summarize evidence signals without making deterministic buy/sell decisions.

## 5. Document Sentiment Module

Code paths: `sentiment/preprocessor.py`, `sentiment/classifier.py`

| Submodule | ML method | Model | Purpose |
|---|---|---|---|
| Language-routed document sentiment | Transformer sequence classification with FinBERT-family models | English: `ProsusAI/finbert`; Chinese: `yiyanghkust/finbert-tone-chinese` | Classifies retrieved documents as `positive`, `negative`, or `neutral`. |
| Per-sentence inference | FinBERT inference + softmax probability | `AutoTokenizer` + `AutoModelForSequenceClassification` | Scores each sentence separately, matching FinBERT's sentence-level training assumption. |
| Document aggregation | Majority label + average sentiment score/confidence | Rule aggregation over sentence predictions | Produces document-level `SentimentItem` records. |
| Mixed/unknown language handling | Runs both Chinese and English FinBERT models and averages probabilities | Both FinBERT models | Handles bilingual or uncertain-language documents. |

Important boundary:

- This is a downstream module. It consumes Query Intelligence artifacts and retrieved documents.
- It does not replace NLU entity detection, intent classification, or retrieval ranking.

## 6. Answer Generation and Next-Question Module

Code path: `scripts/llm_response.py`

This module uses LLMs, but only after compact evidence has already been generated.

| Submodule | ML method | Backend options | Purpose |
|---|---|---|---|
| Answer generation | Chat LLM generation | Default OpenAI-compatible backend: `deepseek-v4-flash`; optional OpenAI-compatible providers, Anthropic Messages API, or local HuggingFace `AutoModelForCausalLM` | Produces frontend-ready `answer_generation` JSON from compact evidence. |
| Next-question prediction | Chat LLM generation | Same backend family as answer generation | Produces `next_question_prediction` JSON with follow-up questions. |
| Local model mode | Causal language model inference | `AutoTokenizer` + `AutoModelForCausalLM`; default local examples include finance Llama/Qwen-style models | Allows local transformer inference if configured. |
| JSON validation and grounding guard | Deterministic validation, not ML | Schema checks, evidence ID checks, disclaimer enforcement | Prevents unsupported or malformed output from being passed to the frontend. |

Important boundary:

- The LLM does not own intent detection, entity linking, source planning, retrieval, or market analysis.
- The LLM must answer from compact evidence and should not invent missing market, fundamental, macro, news, sentiment, or statistical facts.

## 7. Training Module

Code path: `training/`

Training scripts produce the shipped `models/*.joblib` artifacts.

| Training script | Method trained | Output |
|---|---|---|
| `training.train_product_type` | Sparse n-gram text classifier + linear logistic SGD | `models/product_type.joblib` |
| `training.train_intent` | Multi-label sparse n-gram classifier + one-vs-rest style linear SGD | `models/intent_ovr.joblib` |
| `training.train_topic` | Multi-label sparse n-gram classifier + one-vs-rest style linear SGD | `models/topic_ovr.joblib` |
| `training.train_question_style` | Sparse n-gram text classifier + linear logistic SGD | `models/question_style.joblib` |
| `training.train_sentiment` | Sparse n-gram text classifier + linear logistic SGD for user tone | `models/sentiment.joblib` |
| `training.train_entity_crf` | Character-level CRF | `models/entity_crf.joblib` |
| `training.train_clarification_gate` | Dict features + linear logistic SGD | `models/clarification_gate.joblib` |
| `training.train_question_style_reranker` | Dict features + linear logistic SGD | `models/question_style_reranker.joblib` |
| `training.train_source_plan_reranker` | Dict features + linear logistic SGD | `models/source_plan_reranker.joblib` |
| `training.train_out_of_scope_detector` | Sparse n-gram text classifier + linear logistic SGD | `models/out_of_scope_detector.joblib` |
| `training.train_ranker` | Standardized handcrafted ranking features + linear logistic SGD | `models/ranker.joblib` |
| `training.train_typo_linker` | Dict features + linear logistic SGD | `models/typo_linker.joblib` |

## 8. Summary

The core system uses classical, explainable ML: sparse n-gram features with linear logistic SGD classifiers for text classification, CRF for entity boundary detection, handcrafted feature-based linear classifiers for clarification, typo linking, source planning, and ranking, plus TF-IDF/cosine methods for retrieval and deduplication. Transformer models are downstream exceptions: FinBERT handles document sentiment, and LLMs generate final answer and next-question JSON over already retrieved evidence.
