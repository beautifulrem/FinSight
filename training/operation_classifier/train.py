"""Train and evaluate the comparison operation classifier (char n-gram TF-IDF + lexicon hits → logistic regression).

    python -m training.operation_classifier.train                # train, evaluate, write model + results
    python -m training.operation_classifier.train --no-lexicon   # ablation: character n-grams only (not shipped)

Data
----
* Generated wordings: for every operation (``difference``, ``ratio``, ``relative``, ``share``, ``share_total``,
  ``sum``, ``mean``, ``spread``, ``which``, ``total_vs_one``) and for ``none`` (a single lookup, a list of values, an
  opinion), Chinese and English templates written for round 15 (``TEMPLATES``) are filled with targets and metric
  names (seeded). None of the templates copies a reviewer probe or a held-out slice text
  (``tests/test_agent_eval.py`` checks them).
* The templates are split by template, not by sentence: every fifth template of each operation and language is
  **held out** (never trained on), so the held-out accuracy measures wordings the model has not seen.
* Dev wordings: the questions of the dev and ext task sets (``evaluation/agent_eval/tasks/agent_eval_v1.jsonl``,
  ``agent_eval_ext_v1.jsonl``) labelled with the comparison grammar's reading (``none`` where it reads no operation).
  They are training data only (in-sample), so the classifier agrees with the grammar where the grammar is right.

Evaluation (``evaluation/results/operation_classifier-r16.json``; round 15: ``operation_classifier-r15.json``)
-------------------------------------------------------------------
* accuracy and per-class recall on the held-out templates, with a Wilson 95% interval;
* the grammar alone on the same held-out wordings (``frame.parse_clause``), and the grammar with the classifier as
  it is used at run time (fallback when the grammar reads nothing, tie-breaker between matched productions);
* the ``--no-lexicon`` ablation (character n-grams only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from query_intelligence.agent.operation_classifier import (
    ARTIFACT_NAME,
    FALLBACK_MIN,
    OPERATIONS,
    OperationClassifier,
    lexicon_hits,
    normalise,
    structure_hits,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261011
VERSION = "operation-clf-r16"
HELD_OUT_EVERY = 5
PER_TEMPLATE = 24
RESULTS = ROOT / "evaluation" / "results" / "operation_classifier-r16.json"
INDEPENDENT = ROOT / "evaluation" / "operation_wordings_independent_v1.jsonl"
SWEEP = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6)

# (Chinese name, English name) of targets
TARGETS = [
    ("贵州茅台", "Kweichow Moutai"),
    ("茅台", "Moutai"),
    ("五粮液", "Wuliangye"),
    ("中国平安", "Ping An"),
    ("宁德时代", "CATL"),
    ("比亚迪", "BYD"),
    ("招商银行", "China Merchants Bank"),
    ("招行", "CMB"),
    ("工商银行", "ICBC"),
    ("工行", "ICBC"),
    ("紫金矿业", "Zijin Mining"),
    ("美的集团", "Midea"),
    ("长江电力", "China Yangtze Power"),
    ("中信证券", "CITIC Securities"),
    ("平安银行", "Ping An Bank"),
    ("中芯国际", "SMIC"),
    ("格力电器", "Gree"),
    ("海天味业", "Haitian"),
]
METRICS_ZH = [
    "净利润",
    "营收",
    "营业收入",
    "总市值",
    "市盈率",
    "市净率",
    "ROE",
    "毛利率",
    "成交额",
    "扣非净利润",
    "股息率",
]
METRICS_EN = ["net profit", "revenue", "market cap", "P/E", "P/B", "ROE", "gross margin", "turnover", "earnings"]

# operation → language → templates ({A} {B} {C} targets, {m} a metric). Every fifth template is held out.
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "difference": {
        "zh": [
            "{A}的{m}比{B}高出多少",
            "{A}和{B}的{m}差距有多大",
            "{A}的{m}领先{B}多少",
            "{B}的{m}落后{A}几个点",
            "{A}{m}比{B}多了多少",
            "{A}与{B}{m}的差额是多少",
            "{A}的{m}超出{B}多少",
            "{A}比{B}少赚了多少",
            "{A}和{B}在{m}上差多少",
            "{A}的{m}减去{B}的是多少",
            "{A}的{m}和{B}隔着多少",
            "{A}的{m}压过{B}多少",
            "{A}和{B}的{m}还差着多少",
            "{A}{m}跟{B}之间差了多少钱",
            "{A}比{B}的{m}多出来的部分是多少",
            # (round 16) own wording, written after the round-12 review
            "{A}和{B}在{m}上相去多少",
            "{A}{m}减{B}{m}还剩多少",
            "{A}与{B}的{m}隔了多远",
            "{B}{m}比{A}少掉多少",
            "{A}的{m}多{B}一截，这一截是多少",
        ],
        "en": [
            "How much higher is {A}'s {m} than {B}'s?",
            "What is the gap between {A} and {B} on {m}?",
            "By how much does {A} lead {B} in {m}?",
            "How far ahead of {B} is {A} on {m}?",
            "What's the difference in {m} between {A} and {B}?",
            "By how much does {A}'s {m} exceed {B}'s?",
            "How far behind {A} is {B} on {m}?",
            "Subtract {B}'s {m} from {A}'s",
            "How big is {A}'s lead over {B} in {m}?",
            "{A} versus {B}: how much apart is the {m}?",
            "What separates {A} and {B} on {m}?",
            "How wide is the {m} distance between {A} and {B}?",
            "How much daylight is between {A}'s {m} and {B}'s?",
            "{A} minus {B}, {m}?",
            "How much more {m} does {A} bring in than {B}?",
            # (round 16) own wording, written after the round-12 review
            "How big a gap separates {A} from {B} on {m}?",
            "{A} less {B}, in {m} terms?",
            "Work out {A}'s {m} minus {B}'s",
            "What's the shortfall of {B} versus {A} in {m}?",
            "How much does {B} lag {A} by on {m}?",
        ],
    },
    "ratio": {
        "zh": [
            "{A}的{m}是{B}的几倍",
            "{A}的{m}相当于几个{B}",
            "{A}和{B}的{m}之比是多少",
            "{A}的{m}除以{B}的是多少",
            "{A}的{m}大约是{B}的两倍吗",
            "{A}{m}与{B}{m}的比值",
            "{A}的{m}顶得上几个{B}",
            "{A}的{m}有{B}的三倍吗",
            "{A}的{m}是{B}的多少倍",
            "{B}的{m}能抵几个{A}",
            "{A}的{m}翻了{B}几番",
            "{A}的{m}是{B}的多少倍左右",
            "{A}的{m}够{B}的几倍",
            "拿{B}的{m}做基数，{A}是几倍",
            "{A}{m}对{B}{m}是几比一",
            # (round 16) own wording, written after the round-12 review
            "{A}{m}抵得过几个{B}",
            "{A}的{m}约莫是{B}的几倍",
            "{A}的{m}除{B}的{m}得几",
            "{A}的{m}能装下几个{B}",
            "{B}的{m}乘几倍才到{A}",
        ],
        "en": [
            "How many times {B}'s {m} is {A}'s?",
            "What is the ratio of {A}'s {m} to {B}'s?",
            "Is {A}'s {m} more than twice {B}'s?",
            "By what multiple does {A}'s {m} exceed {B}'s?",
            "Divide {A}'s {m} by {B}'s",
            "How many {B}s is {A} worth on {m}?",
            "Is {A}'s {m} roughly three times {B}'s?",
            "{A}'s {m} as a multiple of {B}'s?",
            "What multiple of {B}'s {m} is {A}'s?",
            "How many times bigger is {A}'s {m} than {B}'s?",
            "{A}'s {m} is how many-fold {B}'s?",
            "How many {B}s fit into {A}'s {m}?",
            "{A} to {B} on {m}, what multiple?",
            "Is {A}'s {m} over four times {B}'s?",
            "Express {A}'s {m} as a multiple of {B}'s",
            # (round 16) own wording, written after the round-12 review
            "{A}'s {m} over {B}'s, as a multiple?",
            "How many {B}s would match {A} on {m}?",
            "Roughly how many times does {A} out-{m} {B}?",
            "Is {A}'s {m} a multiple of {B}'s, and which?",
            "Give the {m} multiple of {A} relative to {B}",
        ],
    },
    "relative": {
        "zh": [
            "{A}的{m}比{B}高百分之多少",
            "{A}的{m}比{B}低多少个百分比",
            "{A}相对{B}的{m}溢价多少",
            "{A}的{m}较{B}折价百分之几",
            "{A}比{B}的{m}多出几成",
            "{A}的{m}高出{B}百分之几",
            "{B}的{m}比{A}少百分之多少",
            "{A}的{m}相对{B}高多少%",
            "以{B}为基数，{A}的{m}高百分之几",
            "{A}{m}比{B}多出多少百分比",
            "{A}的{m}高出{B}几个百分比",
            "{A}的{m}比{B}贵了百分之多少",
            "以{B}为准，{A}的{m}多出百分之几",
            "{A}{m}相较{B}上浮了多少%",
            "{A}比{B}的{m}便宜了百分之几",
            # (round 16) own wording, written after the round-12 review
            "{A}的{m}比{B}溢价几成",
            "{A}{m}较{B}偏高的幅度按百分比是多少",
            "{A}的{m}超出{B}的百分点是多少成",
            "{B}的{m}比{A}打了几折",
            "以{B}作基准，{A}的{m}多出百分之几",
        ],
        "en": [
            "By what percentage is {A}'s {m} above {B}'s?",
            "How many percent higher is {A}'s {m} than {B}'s?",
            "What premium does {A} trade at over {B} on {m}?",
            "In percentage terms, how much lower is {B}'s {m} than {A}'s?",
            "Percent difference of {A}'s {m} versus {B}'s?",
            "How much bigger in percent is {A}'s {m} than {B}'s?",
            "By what percent does {A}'s {m} beat {B}'s?",
            "What discount is {B} at relative to {A} on {m}, in percent?",
            "Percentage-wise, how far is {A}'s {m} above {B}'s?",
            "As a percentage, how much higher is {A}'s {m}?",
            "{A}'s {m}, how many percent over {B}'s?",
            "What's the percentage gap of {A} over {B} on {m}?",
            "By what percentage does {B} trail {A} on {m}?",
            "How much cheaper in percent is {A} than {B} on {m}?",
            "Relative to {B}, {A}'s {m} is up by what percent?",
            # (round 16) own wording, written after the round-12 review
            "How much richer, in percent, is {A}'s {m} than {B}'s?",
            "{A} over {B} on {m}: the percent markup?",
            "In relative terms, by what percent does {A} beat {B} on {m}?",
            "What's the percent discount of {B} to {A} on {m}?",
            "{A} versus {B}, percentage difference in {m}",
        ],
    },
    "share": {
        "zh": [
            "{A}的{m}相当于{B}的百分之多少",
            "{A}的{m}只有{B}的几成",
            "{A}的{m}占{B}的比例是多少",
            "{A}的{m}是{B}的百分之几",
            "{A}的{m}才有{B}的多大比例",
            "{A}{m}为{B}的几分之几",
            "{A}的{m}等于{B}的百分之几",
            "{A}的{m}是不是{B}的一半左右",
            "{A}的{m}有没有{B}的六成",
            "{A}的{m}相当于{B}的几成",
            "{A}的{m}够得上{B}的几成",
            "{A}的{m}顶{B}的百分之几",
            "{A}的{m}约占{B}的多大比例",
            "{A}的{m}相当于{B}的多少比例",
            "拿{B}当整体，{A}的{m}占多少",
            # (round 16) own wording, written after the round-12 review
            "拿{B}当一百，{A}的{m}是多少",
            "{A}的{m}够不够{B}的一半，具体百分比多少",
            "{A}的{m}是{B}的几分之几",
            "按{B}为分母，{A}的{m}到了百分之几",
            "{A}的{m}能顶{B}的几成",
        ],
        "en": [
            "{A}'s {m} works out to what percentage of {B}'s?",  # (round 16) reworded
            "What percent of {B}'s {m} is {A}'s?",
            "What fraction of {B}'s {m} does {A} have?",
            "As a share of {B}'s, how big is {A}'s {m}?",
            "{A}'s {m} as a percentage of {B}'s",
            "How many percent of {B}'s {m} does {A} reach?",
            "What proportion of {B}'s {m} is {A}'s?",
            "{A}'s {m} is what share of {B}'s?",
            "Is {A}'s {m} about half of {B}'s?",
            "What portion of {B}'s {m} does {A} make?",
            "{A}'s {m} comes to what percent of {B}'s?",
            "Measured against {B}, what percent is {A}'s {m}?",
            "How large is {A}'s {m} relative to {B}'s, as a percent?",
            "{A}'s {m} equals what fraction of {B}'s?",
            "What percentage of {B}'s {m} does {A} make up?",
            # (round 16) own wording, written after the round-12 review
            "With {B} as the base, what percent is {A}'s {m}?",
            "Is {A}'s {m} at least a third of {B}'s? What percent exactly?",
            "{A}'s {m} measured as a fraction of {B}'s?",
            "Setting {B}'s {m} at 100%, where is {A}?",
            "What proportion of {B}'s {m} does {A} reach?",
        ],
    },
    "share_total": {
        "zh": [
            "{A}的{m}占{A}和{B}两家合计的多少",
            "{A}和{B}的{m}合计中，{A}占几成",
            "{A}的{m}在{A}与{B}之和里占比多少",
            "{A}、{B}、{C}三家{m}总和里{A}占百分之几",
            "{A}在{A}和{B}的{m}总量中占多大比例",
            "{A}占{A}、{B}两家{m}加起来的几成",
            "{A}和{B}的{m}加在一起，{A}占其中多少",
            "{A}的{m}占三家合计的百分比",
            "两家{m}合起来，{A}占比多少",
            "{A}的{m}占{A}{B}之和的比重",
            "{A}和{B}的{m}凑在一起，{A}占几成",
            "{A}在两家{m}里占多大分量",
            "两家{m}的盘子里{A}占多少",
            "{A}的{m}占这两家加总的比例",
            "{A}、{B}、{C}三家{m}合计，{A}的占比",
            # (round 16) own wording, written after the round-12 review
            "{A}和{B}的{m}凑起来，{A}拿了几成",
            "{A}、{B}两家{m}合并后，{A}贡献了多少比例",
            "{A}在{A}、{B}合计{m}中所占份额",
            "{A}和{B}的{m}加总，其中{B}占多大比重",
            "{A}、{B}、{C}的{m}放在一起，{C}分得多少比例",
        ],
        "en": [
            "What is {A}'s {m} as a percentage of {A} and {B} combined?",
            "What share of the combined {m} of {A} and {B} does {A} account for?",
            "Of {A}, {B} and {C}'s total {m}, what percentage is {A}'s?",
            "How much of {A} and {B}'s joint {m} comes from {A}?",
            "{A}'s portion of the {m} the two make together?",
            "What fraction of their combined {m} is {A}'s?",
            "Of the total {m} of {A} and {B}, how big a share is {A}'s?",
            "{A}'s {m} as a share of the sum of {A}, {B} and {C}",
            "What part of the combined {m} belongs to {A}?",
            "Percentage of {A} in the {m} of both together?",
            "Of the pooled {m} of {A} and {B}, how much is {A}'s?",
            "{A}'s slice of the {m} {A} and {B} make together?",
            "In the combined {m} of {A}, {B} and {C}, {A}'s percentage?",
            "How much of the two firms' total {m} is {A}'s?",
            "Share of {A} within {A} plus {B} {m}?",
            # (round 16) own wording, written after the round-12 review
            "Of the {m} {A} and {B} make together, how much is {A}'s part?",
            "{A} and {B} pooled: what slice of the {m} is {B}'s?",
            "Add {A} and {B}'s {m}; what fraction of that comes from {A}?",
            "What percent of the joint {m} of {A}, {B} and {C} is {C}'s?",
            "How much of the {A} plus {B} {m} total does {A} represent?",
        ],
    },
    "sum": {
        "zh": [
            "{A}和{B}的{m}加起来是多少",
            "{A}加{B}的{m}总共多少",
            "{A}与{B}{m}合计多少",
            "{A}、{B}、{C}三家{m}一共多少",
            "{A}和{B}的{m}总计是多少",
            "{A}和{B}两家{m}合并起来有多少",
            "两家{m}一起有多少",
            "{A}跟{B}的{m}汇总是多少",
            "{A}与{B}的{m}之和",
            "{A}、{B}的{m}统共多少",
            "{A}和{B}的{m}凑一块是多少",
            "{A}和{B}的{m}拢共多少",
            "把{A}和{B}的{m}加一加",
            "{A}、{B}两家的{m}一起算是多少",
            "{A}和{B}{m}的总数",
            # (round 16) own wording, written after the round-12 review
            "{A}跟{B}的{m}算到一起是多少",
            "{A}{B}两家{m}一并算多少",
            "{A}、{B}的{m}全加上",
            "{A}与{B}的{m}总量",
            "{A}加上{B}，{m}一共多少",
        ],
        "en": [
            "What is the combined {m} of {A} and {B}?",
            "How much {m} do {A} and {B} make between them?",
            "{A} and {B}'s {m} in total?",
            "Add up the {m} of {A}, {B} and {C}",
            "What do {A} and {B} earn together?",
            "What is {A}'s {m} plus {B}'s?",
            "Total {m} of {A} and {B}?",
            "How much {m} do {A}, {B} and {C} have collectively?",
            "Sum of {A} and {B}'s {m}",
            "What's the {m} of the two put together?",
            "What's {A} and {B}'s {m}, pooled?",
            "{A} and {B}'s {m} lumped together?",
            "The {m} of {A} and {B} taken as one?",
            "How much {m} do {A} and {B} have jointly?",
            "{A} and {B}: {m} in all?",
            # (round 16) own wording, written after the round-12 review
            "What's the aggregate {m} of {A} and {B}?",
            "Lump {A} and {B} together: {m}?",
            "Combine {A}'s and {B}'s {m}",
            "{A} plus {B}: how much {m} in all?",
            "What do {A}, {B} and {C} report in {m}, summed?",
        ],
    },
    "mean": {
        "zh": [
            "{A}、{B}、{C}三家{m}的平均值",
            "{A}和{B}的{m}平均是多少",
            "{A}、{B}、{C}的{m}均值是多少",
            "三家{m}平均下来是多少",
            "{A}和{B}{m}加起来再除以二",
            "{A}、{B}、{C}的{m}算个平均",
            "{A}与{B}的平均{m}",
            "{A}、{B}、{C}{m}的平均水平",
            "{A}、{B}的{m}平均一下",
            "{A}、{B}、{C}三家{m}合计后除以三是多少",
            "{A}、{B}、{C}的{m}拉平了是多少",
            "{A}和{B}的{m}平摊下来是多少",
            "{A}、{B}、{C}三家{m}的均数",
            "{A}、{B}的{m}取个平均",
            "{A}、{B}、{C}的{m}平均起来是多少",
            # (round 16) own wording, written after the round-12 review
            "{A}、{B}、{C}的{m}平均水准是多少",
            "{A}和{B}的{m}取中间值是多少",
            "{A}、{B}、{C}三家{m}均摊下来",
            "{A}与{B}的{m}平均数",
            "{A}、{B}的{m}二者平均",
        ],
        "en": [
            "What is the average {m} of {A}, {B} and {C}?",
            "Mean {m} of {A} and {B}?",
            "On average, what is the {m} of {A}, {B} and {C}?",
            "Average the {m} of {A} and {B}",
            "What do {A}, {B} and {C} average on {m}?",
            "Add up the {m} of {A}, {B} and {C} and divide by three",
            "What's the mean {m} across {A} and {B}?",
            "Average {m} for {A}, {B} and {C}?",
            "The {m} of {A} and {B}, averaged?",
            "Take the average of {A}'s and {B}'s {m}",
            "What's the {m} of {A}, {B} and {C} averaged out?",
            "{A}, {B} and {C}: mean {m}?",
            "Average out {A}'s and {B}'s {m}",
            "The {m} averaged across {A}, {B} and {C}?",
            "What's the arithmetic mean of {A} and {B}'s {m}?",
            # (round 16) own wording, written after the round-12 review
            "What does the {m} of {A}, {B} and {C} average to?",
            "Typical {m} across {A} and {B}, as a mean?",
            "Mean of {A}'s and {B}'s {m}?",
            "{A}, {B}, {C}: what's their average {m}?",
            "Average {A} and {B} on {m} for me",
        ],
    },
    "spread": {
        "zh": [
            "{A}、{B}、{C}里{m}最高的比最低的高多少",
            "{A}、{B}、{C}三家{m}最大和最小差多少",
            "三家里{m}最贵的和最便宜的差多少",
            "{A}、{B}、{C}的{m}极差是多少",
            "{A}、{B}、{C}中{m}最高的减去最低的",
            "{A}、{B}、{C}里{m}最低的和最高的相差多少",
            "{A}、{B}、{C}之中{m}最多的比最少的多多少",
            "{A}、{B}、{C}{m}最大值与最小值之差",
            "这几家{m}最高和最低的差距",
            "{A}、{B}、{C}的{m}最贵和最便宜差几倍",
            "{A}、{B}、{C}里{m}最高和最低隔了多少",
            "{A}、{B}、{C}三家{m}最高的比最低的多几",
            "{A}、{B}、{C}中{m}最大的减最小的",
            "{A}、{B}、{C}三家里{m}最便宜的和最贵的差多少",
            "{A}、{B}、{C}的{m}上下差多少",
            # (round 16) own wording, written after the round-12 review
            "{A}、{B}、{C}的{m}最高和最低隔多少",
            "{A}、{B}、{C}里{m}最强的比最弱的强多少",
            "{A}、{B}、{C}的{m}极值之差",
            "{A}、{B}、{C}三家{m}的落差有多大",
            "{A}、{B}、{C}的{m}高低差是多少",
        ],
        "en": [
            "Among {A}, {B} and {C}, how far apart are the highest and lowest {m}?",
            "What is the spread between the highest and lowest {m} of {A}, {B} and {C}?",
            "Highest minus lowest {m} among {A}, {B} and {C}?",
            "Of {A}, {B} and {C}, the most and least expensive on {m}: what separates them?",
            "Range between the top and bottom {m} of {A}, {B} and {C}?",
            "Between the cheapest and the priciest of {A}, {B} and {C}, how wide is the {m} gap?",
            "Largest versus smallest {m} among {A}, {B} and {C}?",
            "How far is the lowest {m} from the highest among {A}, {B} and {C}?",
            "Biggest to smallest {m} of {A}, {B} and {C}, what's the gap?",
            "{A}, {B}, {C}: highest and lowest {m}, difference?",
            "Top-to-bottom {m} gap among {A}, {B} and {C}?",
            "Of {A}, {B} and {C}, highest {m} less the lowest?",
            "{A}, {B}, {C}: lowest to highest {m}, how far?",
            "What's the max-min spread of {m} across {A}, {B} and {C}?",
            "Priciest versus cheapest of {A}, {B} and {C} on {m}?",
            # (round 16) own wording, written after the round-12 review
            "Among {A}, {B} and {C}, how wide is the {m} range?",
            "Best minus worst {m} for {A}, {B} and {C}?",
            "The {m} span from top to bottom across {A}, {B} and {C}?",
            "{A}, {B} and {C}: maximum less minimum {m}?",
            "What's the high-low {m} gap for {A}, {B} and {C}?",
        ],
    },
    "which": {
        "zh": [
            "{A}和{B}谁的{m}更高",
            "{A}的{m}比{B}高还是低",
            "{A}和{B}哪个{m}更大",
            "{A}与{B}哪家{m}更低",
            "{A}、{B}谁的{m}比较高",
            "{A}和{B}哪一家{m}更多",
            "{A}的{m}是高于还是低于{B}",
            "{A}跟{B}哪只{m}更便宜",
            "{A}和{B}，{m}哪个高",
            "{A}还是{B}的{m}更高",
            "{A}还是{B}的{m}更高",
            "{A}和{B}比，谁的{m}大",
            "{A}和{B}哪边{m}更强",
            "{A}、{B}哪个{m}低一些",
            "{A}的{m}和{B}比起来哪个高",
            # (round 16) own wording, written after the round-12 review
            "{A}的{m}是不是比{B}高",
            "{A}和{B}，{m}谁占上风",
            "{A}、{B}、{C}里谁的{m}最大",
            "{A}与{B}比，{m}哪家更胜一筹",
            "{A}的{m}有没有{B}高",
        ],
        "en": [
            "Which is higher, {A}'s {m} or {B}'s?",
            "Is {A}'s {m} higher or lower than {B}'s?",
            "Which of {A} and {B} has the bigger {m}?",
            "Who has the lower {m}, {A} or {B}?",
            "Is {A}'s {m} above or below {B}'s?",
            "Which one has more {m}, {A} or {B}?",
            "Between {A} and {B}, which has the higher {m}?",
            "{A} or {B}: which {m} is larger?",
            "Which is cheaper on {m}, {A} or {B}?",
            "Which of the two has the higher {m}?",
            "{A} or {B}, whose {m} is higher?",
            "Which wins on {m}, {A} or {B}?",
            "Does {A} or {B} have more {m}?",
            "Higher {m}: {A} or {B}?",
            "Which of {A} and {B} is lower on {m}?",
            # (round 16) own wording, written after the round-12 review
            "Does {A} have a higher {m} than {B}?",
            "Who comes out ahead on {m}, {A} or {B}?",
            "Among {A}, {B} and {C}, which tops the {m} table?",
            "Is {B}'s {m} lower than {A}'s?",
            "Whose {m} is bigger, {A}'s or {B}'s?",
        ],
    },
    "total_vs_one": {
        "zh": [
            "{A}和{B}的{m}加起来比{C}多吗",
            "{C}一家的{m}，比{A}、{B}两家加起来还多多少",
            "{A}加{B}的{m}合计能超过{C}吗",
            "{A}和{B}两家{m}合计，比{C}少多少",
            "{C}的{m}比{A}、{B}加在一起还高吗",
            "{A}、{B}的{m}总和与{C}相比差多少",
            "{A}和{B}合起来的{m}是不是高于{C}",
            "{C}{m}比{A}和{B}合计多出多少",
            "{A}与{B}一共的{m}比{C}高还是低",
            "{A}和{B}{m}总计能比得上{C}吗",
            "{A}和{B}的{m}凑一起比{C}高吗",
            "{A}、{B}合计的{m}超过{C}了吗",
            "{C}的{m}比{A}和{B}两家加总还多吗",
            "{A}加上{B}的{m}比{C}还少多少",
            "两家{m}加一起比{C}多多少",
            # (round 16) own wording, written after the round-12 review
            "跟{C}相比，{A}和{B}的{m}加起来够不够",
            "{A}与{B}的{m}合计，和{C}一比谁大",
            "{A}加上{B}的{m}能不能赶得上{C}",
            "{C}的{m}比{A}、{B}两家凑一起高出多少",
            "拿{A}和{B}的{m}总和去比{C}，差多少",
        ],
        "en": [
            "Do {A} and {B} together earn more than {C}?",
            "Is {C}'s {m} bigger than {A} and {B} combined?",
            "Combined, do {A} and {B} beat {C} on {m}?",
            "How much more {m} does {C} have than {A} and {B} put together?",
            "Are {A} and {B}'s {m} together larger than {C}'s?",
            "Does {C} out-earn {A} and {B} combined?",
            "By how much do {A} and {B} combined trail {C} on {m}?",
            "Is the total {m} of {A} and {B} more than {C}'s?",
            "{A} plus {B} versus {C} on {m}: who is ahead, combined?",
            "Do {A} and {B} combined exceed {C} in {m}?",
            "Do {A} and {B} combined out-earn {C}?",
            "Is {C}'s {m} more than {A} plus {B}?",
            "Taken together, are {A} and {B} ahead of {C} on {m}?",
            "Combined {m} of {A} and {B} versus {C}: by how much?",
            "Does {C} beat {A} and {B} put together on {m}?",
            # (round 16) own wording, written after the round-12 review
            "Against {C}, how do {A} and {B} fare on {m} combined?",
            "Can {A} plus {B} together outweigh {C} in {m}?",
            "How far short of {C} are {A} and {B} combined on {m}?",
            "Relative to {C}, is the {m} of {A} and {B} together bigger?",
            "Put {A} and {B} together: do they top {C} on {m}?",
        ],
    },
    "none": {
        "zh": [
            "帮我查{A}的{m}数值",
            "{A}和{B}的{m}分别是多少",
            "{A}最近走势怎么样",
            "{A}、{B}的{m}各是多少",
            "介绍一下{A}和{B}",
            "{A}的{m}怎么算的",
            "{A}值得买吗",
            "{A}的{m}最新数据",
            "列出{A}、{B}、{C}的{m}",
            "{A}和{B}最近有什么新闻",
            "{A}的{m}去年是多少",
            "我有1000股{A}，值多少钱",
            "{A}属于哪个行业",
            "{m}这个指标是什么含义",
            "{A}和{B}的{m}都给我看看",
            "{A}的{m}现在多少",
            "{A}和{B}的{m}分别列一下",
            "{A}的{m}是怎么变化的",
            "{A}今天涨了吗",
            "{A}有什么风险",
            # (round 16) own wording, written after the round-12 review
            "{A}和{B}的{m}各自报一下",
            "{A}最近{m}有什么变化",
            "{m}这个指标高好还是低好",
            "{A}和{B}哪个更值得长期拿着",
            "帮我看看{A}的{m}历史走势",
        ],
        "en": [
            "What is {A}'s {m}?",
            "Show me the {m} of {A} and {B}",
            "How has {A} been trading lately?",
            "Tell me about {A} and {B}",
            "What is the meaning of the term {m}?",
            "Is {A} worth buying at this level?",
            "List the {m} of {A}, {B} and {C}",
            "Any news on {A} and {B}?",
            "What was {A}'s {m} last year?",
            "I hold 500 {A} shares, what are they worth?",
            "Which industry is {A} in?",
            "Give me {A}'s and {B}'s {m}, each",
            "What's the latest close of {A}?",
            "Explain how {m} is calculated",
            "{A} and {B}: {m} for each, please",
            "Show {A}'s {m}",
            "What are {A}'s and {B}'s {m} figures?",
            "How is {A} doing?",
            "Did {A} close up today?",  # (round 16) reworded: a fill near-copied a router label
            "What are the risks for {A}?",
            # (round 16) own wording, written after the round-12 review
            "Report {A}'s and {B}'s {m} separately",
            "Has {A}'s {m} been improving?",
            "Is a high {m} good or bad?",
            "Which is the better long-term hold, {A} or {B}?",
            "Show {A}'s {m} history",
        ],
    },
}


def _fill(template: str, rng: random.Random, zh: bool) -> str:
    a, b, c = rng.sample(TARGETS, 3)
    names = [item[0] if zh else item[1] for item in (a, b, c)]
    metric = rng.choice(METRICS_ZH if zh else METRICS_EN)
    return template.format(A=names[0], B=names[1], C=names[2], m=metric)


def generated(split: str, rng: random.Random) -> list[tuple[str, str, str]]:
    """``(text, label, template id)``; ``split`` is ``train`` or ``held_out`` (every fifth template)."""
    rows = []
    for label, languages in TEMPLATES.items():
        for language, templates in languages.items():
            for index, template in enumerate(templates):
                held = index % HELD_OUT_EVERY == HELD_OUT_EVERY - 1
                if held != (split == "held_out"):
                    continue
                for _ in range(PER_TEMPLATE):
                    rows.append((_fill(template, rng, language == "zh"), label, f"{label}/{language}/{index}"))
    return rows


def dev_wordings() -> list[tuple[str, str]]:
    """Dev and ext task questions labelled with the grammar's reading (training only)."""
    from query_intelligence.agent.frame import comparison_clause, parse_clause

    rows = []
    for name in ("agent_eval_v1.jsonl", "agent_eval_ext_v1.jsonl"):
        path = ROOT / "evaluation" / "agent_eval" / "tasks" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for turn in json.loads(line).get("turns") or []:
                query = str(turn.get("query") or "")
                clause = comparison_clause(query)
                operation = parse_clause(clause).operation if clause else None
                rows.append((query, operation or "none"))
    return sorted(set(rows))


def build_pipeline(lexicon: bool = True, c: float = 4.0, words: bool = True) -> Pipeline:
    parts: list[tuple[str, Any]] = [
        (
            "chars",
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=20000),
        )
    ]
    if words:
        # (round 16, review-12 L17) English word 1-2-grams: "stack up", "short of", "how much of"
        parts.append(
            (
                "words",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)\b[a-z][a-z'/-]+\b",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=20000,
                ),
            )
        )
    if lexicon:
        parts.append(("lexicon", Pipeline([("hits", FunctionTransformer(lexicon_hits)), ("dict", DictVectorizer())])))
        parts.append(
            ("structure", Pipeline([("hits", FunctionTransformer(structure_hits)), ("dict", DictVectorizer())]))
        )
    return Pipeline([("features", FeatureUnion(parts)), ("model", LogisticRegression(C=c, max_iter=4000))])


def wilson(successes: int, total: int, z: float = 1.959964) -> dict[str, Any]:
    if not total:
        return {"rate": None, "low": None, "high": None, "n": 0}
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"rate": round(p, 4), "low": round(centre - half, 4), "high": round(centre + half, 4), "n": total}


def _grammar(text: str) -> str:
    from query_intelligence.agent.frame import comparison_clause, parse_clause

    clause = comparison_clause(text)
    return (parse_clause(clause).operation if clause else None) or "none"


def _combined(text: str, model: OperationClassifier) -> str:
    """The grammar with the classifier as used at run time: tie-breaker between matched productions, fallback when
    the grammar reads nothing (same thresholds as ``operation_classifier``)."""
    from query_intelligence.agent import operation_classifier as oc
    from query_intelligence.agent.frame import comparison_clause, parse_clause

    clause = comparison_clause(text)
    if clause:
        parsed = parse_clause(clause)
        probabilities = model.probabilities(clause)
        label = max(probabilities, key=probabilities.__getitem__)
        if (
            len(set(parsed.matched)) >= 2
            and label != parsed.operation
            and label in parsed.matched
            and probabilities[label] >= oc.TIE_MIN
            and probabilities.get(str(parsed.operation), 0.0) <= oc.TIE_MAX_OTHER
        ):
            return label
        return parsed.operation or "none"
    named = sum(1 for zh, en in TARGETS if zh in text or en in text)
    if not (named >= 2 or oc._SEVERAL.search(text)) or not oc._ASKS_QUANTITY.search(text):
        return "none"
    label, probability = model.read(text)
    ok = label != "none" and probability >= FALLBACK_MIN and oc.compatible(label, text)
    return label if ok else "none"


def evaluate(model: OperationClassifier, rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    by_class: dict[str, list[int]] = {label: [0, 0] for label in OPERATIONS}
    correct = grammar_correct = combined_correct = rescued = grammar_none = 0
    misses: list[dict[str, str]] = []
    for text, label, template in rows:
        predicted, _p = model.read(text)
        grammar = _grammar(text)
        combined = _combined(text, model)
        by_class[label][1] += 1
        if predicted == label:
            correct += 1
            by_class[label][0] += 1
        elif len(misses) < 25:
            misses.append({"template": template, "expected": label, "classifier": predicted, "grammar": grammar})
        grammar_correct += grammar == label
        combined_correct += combined == label
        if grammar == "none" and label != "none":
            grammar_none += 1
            rescued += combined == label
    total = len(rows)
    return {
        "examples": total,
        "templates": len({row[2] for row in rows}),
        "classifier_accuracy": wilson(correct, total),
        "grammar_accuracy": wilson(grammar_correct, total),
        "grammar_plus_classifier_accuracy": wilson(combined_correct, total),
        "grammar_missed_operations": grammar_none,
        "rescued_by_fallback": rescued,
        "recall_by_class": {label: wilson(hit, n) for label, (hit, n) in by_class.items() if n},
        "classifier_misses_sample": misses,
    }


EVALUATION_SETS = (
    "agent_eval_holdout_v1.jsonl",
    "agent_eval_test_v2.jsonl",
    "agent_eval_multiturn_v1.jsonl",
    "agent_eval_test_v3.jsonl",
    "router_labels_independent_v1.jsonl",
    "router_labels_independent_v2.jsonl",
)


def fallback_activations(model: OperationClassifier) -> dict[str, Any]:
    """How often the fallback would read an operation into the questions of the committed evaluation sets (none of
    them is training data) that the grammar reads none in: counts per set and per operation only (no text). Each
    question is run through the offline NLU for the number of targets it names, as at run time."""
    import os

    from query_intelligence.agent import operation_classifier as oc
    from query_intelligence.agent.frame import comparison_clause

    os.environ.setdefault("QI_USE_LIVE_MARKET", "0")
    from evaluation.agent_eval.runner import build_offline_service

    service = build_offline_service()
    report: dict[str, Any] = {}
    for name in EVALUATION_SETS:
        path = ROOT / "evaluation" / "agent_eval" / "tasks" / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        queries = (
            [row["query"] for row in rows]
            if "query" in rows[0]
            else [turn["query"] for row in rows for turn in row.get("turns") or []]
        )
        none = fired = 0
        by_operation: dict[str, int] = {}
        for query in queries:
            if comparison_clause(query) is not None:
                continue
            none += 1
            nlu = service.analyze_query(query)
            nlu = nlu if isinstance(nlu, dict) else nlu.model_dump()
            named = len([e for e in nlu.get("entities") or [] if e.get("symbol")])
            if not (named >= 2 or oc._SEVERAL.search(query)) or not oc._ASKS_QUANTITY.search(query):
                continue
            label, probability = model.read(query)
            if label != "none" and probability >= FALLBACK_MIN and oc.compatible(label, query):
                fired += 1
                by_operation[label] = by_operation.get(label, 0) + 1
        report[name] = {
            "questions": len(queries),
            "grammar_none": none,
            "fallback_fires": fired,
            "by_operation": by_operation,
        }
    return report


def fallback_sweep(model: OperationClassifier, rows: list[tuple[str, str, int]]) -> dict[str, Any]:
    """(round 16, review-12 L17) The fallback at several thresholds on the rows the grammar reads no operation in:
    how often it fires and how often it is right (``rows``: text, label, targets named)."""
    from query_intelligence.agent import operation_classifier as oc

    eligible = [
        (text, label)
        for text, label, named in rows
        if _grammar(text) == "none" and (named >= 2 or oc._SEVERAL.search(text)) and oc._ASKS_QUANTITY.search(text)
    ]
    readings = [(model.read(text), text, label) for text, label in eligible]
    report: dict[str, Any] = {"grammar_none_eligible": len(eligible),
                              "operations_among_them": sum(1 for _t, label in eligible if label != "none")}  # fmt: skip
    for threshold in SWEEP:
        fired = [
            (predicted, label)
            for (predicted, probability), text, label in readings
            if predicted != "none" and probability >= threshold and oc.compatible(predicted, text)
        ]
        right = sum(1 for predicted, label in fired if predicted == label)
        report[f"{threshold:.2f}"] = {"fires": len(fired), "right": right, "precision": wilson(right, len(fired))}
    return report


def _independent_rows() -> list[tuple[str, str, int]]:
    rows = [json.loads(line) for line in INDEPENDENT.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row["question"], row["operation"], len(row.get("companies") or [])) for row in rows]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-lexicon", action="store_true", help="ablation: character n-grams only (not written)")
    parser.add_argument("--no-words", action="store_true", help="ablation: without word n-grams (not written)")
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args(argv)
    rng = random.Random(SEED)
    train = generated("train", rng)
    held_out = generated("held_out", random.Random(SEED + 1))
    dev = dev_wordings()
    texts = [normalise(text) for text, _label, _template in train] + [normalise(text) for text, _label in dev]
    labels = [label for _text, label, _template in train] + [label for _text, label in dev]
    ablation = args.no_lexicon or args.no_words
    pipeline = build_pipeline(lexicon=not args.no_lexicon, words=not (args.no_lexicon or args.no_words))
    pipeline.fit(texts, labels)
    model = OperationClassifier(pipeline=pipeline, version=VERSION)
    report: dict[str, Any] = {
        "version": VERSION,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python -m training.operation_classifier.train"
        + (" --no-lexicon" if args.no_lexicon else " --no-words" if args.no_words else ""),
        "features": "char_wb 1-3-gram TF-IDF (max 20000)"
        + ("" if args.no_lexicon or args.no_words else " + English word 1-2-gram TF-IDF (max 20000)")
        + (
            ""
            if args.no_lexicon
            else " + grammar lexicon hits + structure (order of total and comparison words, list of three, yes/no)"
        ),
        "model": "LogisticRegression(C=4, multinomial, lbfgs)",
        "seed": SEED,
        "train": {
            "generated": len(train),
            "generated_templates": len({row[2] for row in train}),
            "dev_wordings": len(dev),
            "dev_labels": {label: sum(1 for _t, item in dev if item == label) for label in OPERATIONS},
        },
        "held_out": evaluate(model, held_out),
        "in_sample_dev": evaluate(model, [(text, label, "dev") for text, label in dev]),
        "thresholds": {"fallback_min": FALLBACK_MIN},
        # (round 16, review-12 L17) the fallback threshold swept on the rows the grammar reads nothing in: held-out
        # templates, and the independent wording set (read by the engineers after its first run: after exposure)
        "fallback_threshold_sweep": {
            "held_out_templates": fallback_sweep(model, [(text, label, 2) for text, label, _t in held_out]),
            "independent_v1_after_exposure": fallback_sweep(model, _independent_rows()),
        },
    }
    dev_hash = hashlib.sha256(json.dumps(sorted(dev), ensure_ascii=False).encode("utf-8")).hexdigest()
    report["train"]["dev_wordings_sha256"] = dev_hash
    if not ablation:
        report["fallback_on_evaluation_sets"] = fallback_activations(model)
        path = ROOT / "models" / ARTIFACT_NAME
        # (round 16, review-12 L16) the training commit and the dev wordings' hash travel with the artifact
        provenance = {"trained_at_commit": report["commit"], "dirty": report["dirty"], "dev_wordings_sha256": dev_hash}
        joblib.dump({"pipeline": pipeline, "version": VERSION, **provenance}, path, compress=3)
        report["artifact"] = {
            "path": f"models/{ARTIFACT_NAME}",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **provenance,
        }
        out = Path(args.out)
    else:
        suffix = "nolexicon" if args.no_lexicon else "nowords"
        out = Path(args.out).with_name(f"operation_classifier-r16-{suffix}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("version", "train")}, ensure_ascii=False))
    held = report["held_out"]
    print(
        json.dumps(
            {
                "held_out_classifier": held["classifier_accuracy"],
                "held_out_grammar": held["grammar_accuracy"],
                "held_out_combined": held["grammar_plus_classifier_accuracy"],
                "rescued": f"{held['rescued_by_fallback']}/{held['grammar_missed_operations']}",
            }
        )
    )
    return report


if __name__ == "__main__":
    main()
