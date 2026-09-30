"""Output-side safety layer: runs on every answer (LLM and template) before the compliance guard.

The input-side defenses (the injection filter and classifier, the untrusted-data envelope, the system prompt) keep
a poisoned document from *steering* the model, but a model that summarises the document faithfully can still
relay what it says. This layer looks at the answer sentence by sentence, together with the run's evidence:

1. **Scrub.** A sentence with private contact details (phone, QQ / WeChat / Telegram / ``@handle``, non-allowlisted
   link), solicitation, guarantee or hype wording (``text_safety.find_prohibited_promotion``) or a trading call
   (``compliance.contains_trading_instruction``) that came from a document (it cites a document, or the flagged
   fragment occurs in a document of the run) is replaced by a neutral note, once per answer:
   "一篇文档含有未经核实的推广/联系方式内容，已省略。" A flagged sentence that did not come from a document is left
   to the compliance guard, which removes it with its own note.
2. **Document-only claims.** A sentence whose only support is document text and that states
   * a regulatory action (立案调查, 行政处罚, ST, 退市, 停牌, 强制清算, 风险名单, delisting, CSRC probe ...) that no
     second, differently worded document corroborates, or
   * a share-capital action or a dividend-plan change (送转, 高送转, 每10股送10股, 转增, 分红方案调整/取消, bonus
     shares, stock split; round 8) that no second, differently worded document corroborates, or
   * a reported fundamental (ROE, EPS, dividend or book value per share) that another document or another sentence
     of the answer states differently, or
   * a reported amount (net profit, revenue; round 8) whose only support is one document and that another answer
     sentence or a document states differently for the same period and company, e.g. a planted "更正公告：归母净利润
     应为912.6亿元" next to the annual report's 823.20 亿元. This works on news questions, where no fundamentals
     were fetched; figures two different documents agree on are left alone, and so are prior-period and change
     figures ("上年同期", "同比下降4.53%"), or
   * (round 9, E3) any figure (a number with a unit: percent, amount, multiple, points, price) that the run's
     structured data does not contain and that exactly one document wording carries: an insider's "一季度净利润同比
     增长63.5%", a poll, a buyback size, a footnote "restatement", a statistic, and also an ordinary single-source
     figure (a dividend, a sales number). A sentence that cites only structured evidence is left to the verifier;
     figures two differently worded documents state are left alone; (round 10, F9) so are figures the named
     company's structured fundamentals confirm when the run did not fetch them (a news question quoting the annual
     report), looked up once through ``corroborate``
   is attributed with the layer's own marker, whatever the model wrote: "据一篇文档称，…（未经其他来源证实）" / "…
   (according to one document; not confirmed by other sources)". A sentence that already says the claim is
   unverified is left as it is; one that only names its source ("媒体报道称…") gets the suffix. A fundamental or
   amount that contradicts the run's structured data for the same metric (and period and company, where stated) is
   dropped with a note (the verifier flags the same conflict on LLM drafts, so this is the net for template answers
   and repaired drafts).

Target prices are trading calls (rule 1). The layer never adds a number or a claim; it only removes, replaces with
a fixed note, or wraps a sentence in attribution. ``evaluation/agent_eval/redteam.py`` records which detector
matches remain and whether they sit in attributed sentences.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..text_safety import find_prohibited_promotion, fold
from .compliance import contains_trading_instruction
from .evidence import _NUMBER as _NUMBER_TOKEN
from .evidence import AgentEvidence, EvidenceStore, _collect_numbers
from .verifier import (
    _BARE_SCALES,
    _CITATION,
    _CONNECTOR_OPENING,
    _UNIT_SCALES,
    _chinese_values,
    _cleaned,
    _is_supported,
    metric_claims,
    structured_metric_values,
    whole_sentences,
)

NOTE_PROMOTION_ZH = "一篇文档含有未经核实的推广/联系方式内容，已省略。"
NOTE_PROMOTION_EN = "A document contained unverified promotional content; omitted."
NOTE_TRADING_ZH = "一篇文档含有买卖建议类内容，已省略。"
NOTE_TRADING_EN = "A document contained a buy/sell call; omitted."
NOTE_CONFLICT_ZH = "一篇文档中的财务数据与结构化数据不一致，已省略。"
NOTE_CONFLICT_EN = "A document stated a financial figure that contradicts the fundamentals data; omitted."
ATTRIBUTION_PREFIX_ZH = "据一篇文档称，"
ATTRIBUTION_SUFFIX_ZH = "（未经其他来源证实）"
ATTRIBUTION_SUFFIX_EN = " (according to one document; not confirmed by other sources)"
LIMITATION_DISAGREE_ZH = "不同文档对同一财务指标给出的数值不一致，相关数值仅作为文档说法列出。"
LIMITATION_DISAGREE_EN = "Documents disagree on the same financial metric; those figures are reported only as claims."

# Compliance notes this layer adds, by the kind of edit (the ``kind`` label of
# ``finsight_output_safety_edits_total``, see ``agent/telemetry.py``).
EDIT_KINDS = {
    "omitted_document_promotion": "promotion_or_contact",
    "omitted_document_trading_call": "trading_call",
    "omitted_conflicting_document_figure": "conflicting_figure",
    "attributed_document_claim": "attribution",
}

_REGULATORY = re.compile(
    r"立案(?:调查|侦查)?|行政处罚|处罚决定|监管函|警示函|纪律处分|公开谴责|(?:退市)?风险警示|(?-i:(?<![A-Za-z])\*?ST(?![A-Za-z]))|戴帽|"
    r"退市|终止上市|暂停上市|摘牌|停牌|强制清算|风险名单|财务造假|欺诈发行|涉嫌(?:违法|违规|犯罪|信息披露)|"
    r"\bdelist(?:ed|ing|ment)?\b|\b(?:trading\s+)?(?:suspension|halt)\b|\bsuspended\s+from\s+trading\b|"
    r"\b(?:probe|investigation|penalt(?:y|ies)|fined|sanction(?:s|ed)?)\b|\bforced\s+liquidation\b|"
    r"\brisk\s+(?:list|watch\s*list)\b|\baccounting\s+fraud\b|\bspecial\s+treatment\b|"
    # corporate actions that move a price as much as a regulator's decision ("合并已获批准", "to be acquired")
    r"(?:合并|吸收合并|重大资产重组|借壳|被收购|要约收购|私有化)(?:[^，。；]{0,8}(?:获批|批准|通过|完成|落地))?|"
    r"\bmerger\b|\b(?:to\s+be\s+)?acquired\s+by\b|\btakeover\s+(?:bid|offer)\b|\bgo(?:ing)?\s+private\b|"
    # (round 8) share-capital actions and changes to a dividend plan: a "独家：拟每10股送10股" rumour moves a price
    # like a regulator's decision, and one document is not enough to state it
    r"(?:每\s*)?(?<!\d)(?:10|十)\s*股\s*(?:送|转增?|送转)\s*(?:红股\s*)?\d+(?:\.\d+)?\s*股|"
    r"(?<![\d.])(?:10|十)\s*(?:送|转)\s*(?:\d+|[一二三四五六七八九十]+)|"
    r"高送转|送转(?:方案|预案|股份)?|转增股本|送红股|(?:资本)?公积金?转增|"
    r"(?:分红|利润分配|派息|派现)(?:预案|方案|计划)?[^，,。；;]{0,8}"
    r"(?:调整|变更|修改|更正|取消|终止|撤回|撤销|上调|下调|提高至|大幅提高)|"
    r"\bbonus\s+(?:shares?|issue)\b|\bstock\s+split\b|\bscrip\s+dividend\b|"
    r"\b\d+[- ]for[- ]\d+\s+(?:bonus|stock\s+)?split\b|"
    r"\bdividend\s+(?:plan\s+)?(?:cut|cancel\w*|scrapp\w*|suspend\w*|rais\w*|hik\w*|revis\w*)",
    re.IGNORECASE,
)
# A negation earlier in the same clause ("未发现立案调查或停牌信息", "no sign of a probe or delisting",
# "不送红股，不以资本公积金转增股本").
_NEGATION = re.compile(
    r"未(?:发现|见|曾|被|涉及|检索到|提及|显示)|没有|并未|不存在|并非|不会|无(?:任何|相关)?|不(?:以|再|进行|实施)?\s*$|"
    r"\bno\b|\bnot\b|\bwithout\b|\bnever\b|\bnone\b",
    re.IGNORECASE,
)

# (round 8) Reported amounts a planted "更正公告" restates ("归母净利润应为912.6亿元"). Compared in yuan across the
# answer, the documents and the structured fundamentals (``get_fundamentals`` ``net_profit`` / ``revenue``).
_AMOUNT_METRICS: dict[str, re.Pattern[str]] = {
    "net_profit": re.compile(
        r"归母净利润|归属于(?:上市公司|母公司)?(?:股东|所有者)的净利润|扣非(?:后)?净利润|净利润|"
        r"\bnet\s+(?:profit|income|earnings)(?:\s+attributable\s+to\s+(?:[\w-]+\s+){0,4}?"
        r"(?:shareholders|owners|equity\s+holders))?|\bprofit\s+attributable\s+to\s+(?:[\w-]+\s+){0,4}?"
        r"(?:shareholders|owners|equity\s+holders)",
        re.IGNORECASE,
    ),
    "revenue": re.compile(r"营业总收入|营业收入|营收|总收入|\b(?:operating\s+|total\s+)?revenues?\b", re.IGNORECASE),
}
_AMOUNT_KEYS: dict[str, re.Pattern[str]] = {
    "net_profit": re.compile(r"^(?:net_profit|n_income(?:_attr_p)?|net_income|netprofit)(?:_\w+)?$", re.IGNORECASE),
    "revenue": re.compile(r"^(?:revenue|total_revenue|operating_revenue|total_operating_revenue)$", re.IGNORECASE),
}
# the first number after the metric name; a change ("同比下降4.53%", "较上年减少39.5亿元", "fell 4.5% to") or a
# prior-period figure ("上年同期", "a year earlier") is not a level
_AMOUNT_NUMBER = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>万亿元?|亿元|亿|万元|万|元|trillion|billion|bn|million|mn|yuan|rmb|cny)?",
    re.IGNORECASE,
)
_AMOUNT_UNITS = {"万亿": 1e12, "亿": 1e8, "万": 1e4, "元": 1.0, "trillion": 1e12, "billion": 1e9, "bn": 1e9}
_AMOUNT_UNITS |= {"million": 1e6, "mn": 1e6, "yuan": 1.0, "rmb": 1.0, "cny": 1.0}
_CHANGE_OR_PRIOR = re.compile(
    r"同比|环比|较|比上年|增加|减少|增长|下降|上升|下滑|增幅|降幅|变动|上年|去年|同期|前一年|"
    r"\b(?:up|down|rose|fell|increase[sd]?|decrease[sd]?|grew|declin\w*|chang\w*|prior|previous|last\s+year|"
    r"a\s+year\s+(?:earlier|ago)|from)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:FY\s*)?((?:19|20)\d{2})(?!\s*月|\d)(?:\s*(?:年度?|财年|fiscal|full[- ]year|annual))?", re.I)
_SUB_PERIOD = re.compile(
    r"(?P<q>一季度|第一季度|Q1|二季度|第二季度|Q2|三季度|第三季度|Q3|四季度|第四季度|Q4|上半年|半年度|中期|H1|前三季度|"
    r"first\s+half|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter|nine\s+months)",
    re.IGNORECASE,
)


def _negated(text: str, start: int) -> bool:
    clause = re.split(r"[，,；;:：。.!?！？]", text[:start])[-1]
    return bool(_NEGATION.search(clause))


# The sentence already tells the reader the claim is unverified: leave it.
_UNVERIFIED = re.compile(
    r"未经?(?:其他来源|官方)?(?:证实|核实|确认|核验)|(?:无法|无从|不能|难以)(?:核实|证实|核验|确认)|无可核验|"
    r"尚未(?:证实|核实|确认)|真实性(?:存疑|待|未|无法)|不能据此|传闻|网传|未(?:获|得到|在)?[^。；;]{0,16}(?:证实|印证|核实)|"
    r"不(?:作为|能作为|应视为|视为)(?:已确认)?事实|"
    r"\bnot\s+(?:treated|reported|taken)\s+(?:here\s+)?as\s+(?:a\s+)?fact\b|\bunsupported\b|\bno\s+official\s+\w+\s+corroborat|"
    r"\bunverified\b|\bunconfirmed\b|\bnot\s+(?:been\s+)?(?:independently\s+)?(?:corroborated|confirmed|verified)\b|"
    r"\bcould\s+not\s+be\s+(?:verified|confirmed)\b|\bcannot\s+be\s+(?:verified|confirmed)\b|\bunofficial\b|"
    r"\balleg(?:ed|es|ing|edly)\b|\bpurported(?:ly)?\b|\brumou?r",
    re.IGNORECASE,
)
# The sentence already names its source ("据…报道", "有媒体称", "according to"): only the suffix is added.
# (round 9) "另据…", "此外据…" open with their source too, and "透露" names one ("董秘透露…").
_REPORTED = re.compile(
    r"^\s*(?:(?:另外|此外|同时|另)[，,]?\s*)?(?:据|有(?:媒体|报道|文章|消息)|另有|一(?:篇|则|条)|报道|文档)|称|透露|"
    r"according to|reported",
    re.I,
)
_SENTENCE_TAIL = re.compile(
    r"(?P<tail>(?:\s*\[[^\[\]\s]{2,160}\])*\s*[。．.!?！？]?(?:\s*\[[^\[\]\s]{2,160}\])*\s*)$", re.DOTALL
)


def states_unverified(sentence: str) -> bool:
    """True when the sentence already tells the reader its claim is unverified or single-source."""
    return bool(_UNVERIFIED.search(sentence))


def scrub_answer(
    answer: dict[str, Any],
    store: EvidenceStore,
    *,
    zh: bool,
    corroborate: Callable[[], list[tuple[float, bool]]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(answer, notes)``; notes name the rules that changed the answer (see the module docstring).

    ``corroborate`` (round 10, F9) returns the numbers of the named companies' structured fundamentals when the run
    did not fetch them (a news question). It is called at most once, and only when a sentence states a
    single-document figure the run's structured data lacks: a figure the fundamentals confirm (the annual report's
    revenue quoted by a news item) is FinSight's own data, not a one-document claim, and is not attributed."""
    text = str(answer.get("answer") or "")
    context = _Context(store, zh=zh, texts=[text, *map(str, answer.get("key_points") or [])], corroborate=corroborate)
    guarded = dict(answer)
    guarded["answer"] = context.scrub_text(text, allow_note=True)
    points: list[str] = []
    for point in answer.get("key_points") or []:
        cleaned = context.scrub_text(str(point), allow_note=False)
        if cleaned.strip():
            points.append(cleaned)
    guarded["key_points"] = points
    # a key point replaced by a note: say so once in the answer text
    for note in context.pending_notes:
        if note not in guarded["answer"]:
            guarded["answer"] = f"{guarded['answer'].rstrip()}{'' if zh else ' '}{note}".strip()
    limitations = [
        str(item)
        for item in answer.get("limitations") or []
        if not find_prohibited_promotion(str(item)) and not contains_trading_instruction(str(item))
    ]
    if context.documents_disagree:
        limitations.append(LIMITATION_DISAGREE_ZH if zh else LIMITATION_DISAGREE_EN)
    guarded["limitations"] = list(dict.fromkeys(limitations))
    return guarded, sorted(context.notes)


class _Context:
    def __init__(
        self,
        store: EvidenceStore,
        *,
        zh: bool,
        texts: list[str] | None = None,
        corroborate: Callable[[], list[tuple[float, bool]]] | None = None,
    ) -> None:
        self.store = store
        self._corroborate = corroborate
        self._corroborating: list[tuple[float, bool]] | None = None
        self.zh = zh
        self.documents = [item for item in store.items() if item.kind == "document"]
        self.raw_texts = {item.evidence_id: _document_text(item) for item in self.documents}
        self.document_texts = {evidence_id: _compact(text) for evidence_id, text in self.raw_texts.items()}
        self.notes: set[str] = set()
        self.emitted: set[str] = set()
        self.pending_notes: list[str] = []
        self.documents_disagree = False
        # (round 8) every reported amount in the answer (by sentence) and in each document, for the conflict check
        self.names = _entity_names(store)
        self.answer_sentences = [sentence for text in texts or [] for sentence in _sentences(text) if sentence.strip()]
        self.answer_figures = [
            (_sentence_key(sentence), figure)
            for sentence in self.answer_sentences
            for figure in amount_figures(sentence, self.names)
        ]
        self.document_figures = [
            (item.evidence_id, figure)
            for item in self.documents
            for figure in amount_figures(_document_text(item), self.names)
        ]
        self.structured_figures = _structured_amounts(store, self.names)
        # (round 9) every number of the structured payloads, and every figure with a unit in each document
        self.structured_numbers = structured_numbers(store)
        self.document_unit_figures = {
            item.evidence_id: unit_figures(_document_text(item), with_context=True) for item in self.documents
        }

    # ---- per text field ----

    def scrub_text(self, text: str, *, allow_note: bool) -> str:
        if not text.strip():
            return text
        out: list[str] = []
        for sentence in _sentences(text):
            if not sentence.strip():
                out.append(sentence)
                continue
            replaced = self.scrub_sentence(sentence)
            if replaced is None:
                out.append(sentence)
                continue
            kind, value = replaced
            if kind == "rewrite":
                out.append(value)
                continue
            # a fixed note replaces the sentence (once per answer)
            trailing = sentence[len(sentence.rstrip()) :]
            if not allow_note:
                if value not in self.pending_notes:
                    self.pending_notes.append(value)
                continue
            if value in self.emitted:
                continue
            self.emitted.add(value)
            out.append(value + (trailing or ("" if self.zh else " ")))
        return "".join(out).strip() if allow_note else "".join(out)

    def scrub_sentence(self, sentence: str) -> tuple[str, str] | None:
        cited = [match.group(1) for match in _CITATION.finditer(sentence) if match.group(1) in self.store]
        doc_ids = [i for i in cited if self.store.get(i).kind == "document"]  # type: ignore[union-attr]
        structured_ids = [i for i in cited if i not in doc_ids]

        findings = find_prohibited_promotion(sentence)
        if findings and self._from_document(doc_ids, [finding.text for finding in findings]):
            self.notes.update({"removed_prohibited_promotion", "omitted_document_promotion"})
            return "note", NOTE_PROMOTION_ZH if self.zh else NOTE_PROMOTION_EN
        if contains_trading_instruction(sentence) and (
            doc_ids or any(contains_trading_instruction(text) for text in self.raw_texts.values())
        ):
            self.notes.update({"removed_trading_instruction", "omitted_document_trading_call"})
            return "note", NOTE_TRADING_ZH if self.zh else NOTE_TRADING_EN

        document_only = bool(doc_ids) and not structured_ids
        attribute = False
        for claim in metric_claims(sentence):
            reference = structured_metric_values(self.store, claim.metric)
            if reference:
                if not _is_supported(claim.value, reference, claim.scales, claim.rounding) and not structured_ids:
                    self.notes.add("omitted_conflicting_document_figure")
                    return "note", NOTE_CONFLICT_ZH if self.zh else NOTE_CONFLICT_EN
                continue
            if (document_only or not cited) and (
                self._documents_disagree(claim.metric, claim.value, claim.rounding)
                or self._answer_disagrees(sentence, claim.metric, claim.value, claim.rounding)
            ):
                self.documents_disagree = True
                attribute = True
        for figure in amount_figures(sentence, self.names):
            verdict = self._amount_conflict(sentence, figure, cited=bool(cited), document_only=document_only)
            if verdict == "structured" and not structured_ids:
                self.notes.add("omitted_conflicting_document_figure")
                return "note", NOTE_CONFLICT_ZH if self.zh else NOTE_CONFLICT_EN
            if verdict == "disputed":
                self.documents_disagree = True
                attribute = True
        if not attribute and (document_only or not cited):
            attribute = self._uncorroborated_regulatory_claim(sentence, cites_document=document_only)
        if not attribute and (doc_ids or not cited):
            # a sentence citing only structured evidence states FinSight's own figures (the verifier checks them)
            attribute = self._single_document_figure(sentence)
        if attribute and not states_unverified(sentence):
            self.notes.add("attributed_document_claim")
            return "rewrite", self._attribute(sentence)
        return None

    # ---- helpers ----

    def _from_document(self, doc_ids: list[str], fragments: list[str]) -> bool:
        if doc_ids:
            return True
        keys = [_compact(fragment) for fragment in fragments]
        return any(key and key in text for key in keys for text in self.document_texts.values())

    def _documents_disagree(self, metric: str, value: float, rounding: float) -> bool:
        """A document states ``metric`` with a value other than ``value`` (a planted dividend next to the real one)."""
        for item in self.documents:
            for claim in metric_claims(_document_text(item)):
                if claim.metric == metric and not _is_supported(value, [claim.value], claim.scales, rounding):
                    return True
        return False

    def _answer_disagrees(self, sentence: str, metric: str, value: float, rounding: float) -> bool:
        """Another sentence of the answer (or a key point) states ``metric`` with a different value."""
        own = _sentence_key(sentence)
        for other in self.answer_sentences:
            if _sentence_key(other) == own:
                continue
            for claim in metric_claims(other):
                if claim.metric == metric and not _is_supported(value, [claim.value], claim.scales, rounding):
                    return True
        return False

    def _amount_conflict(self, sentence: str, figure: Figure, *, cited: bool, document_only: bool) -> str | None:
        """``"structured"`` when the figure contradicts the run's fundamentals for the same metric, period and
        company; ``"disputed"`` when a sentence that only documents support (or that cites nothing but repeats a
        document figure) states an amount that another answer sentence or a document states differently and no
        second document corroborates; else ``None``."""
        if figure.entity is None and len(set(self.names.values())) > 1:
            return None  # several companies in the run and the sentence names none: cannot tell whose figure it is
        structured = [other for other in self.structured_figures if figure.comparable(other)]
        if any(not figure.agrees(other) for other in structured):
            return "structured"
        if structured:
            return None  # the fundamentals data confirm it
        if cited and not document_only:
            return None
        supporting = {
            self.document_texts[evidence_id]
            for evidence_id, other in self.document_figures
            if figure.comparable(other) and figure.agrees(other)
        }
        if not cited and not supporting:
            return None  # not a document figure: the verifier's business
        if len(supporting) >= 2:
            return None  # two different documents state it
        own = _sentence_key(sentence)
        others = [other for key, other in self.answer_figures if key != own]
        others += [other for _evidence_id, other in self.document_figures]
        if any(figure.comparable(other) and not figure.agrees(other) for other in others):
            return "disputed"
        return None

    def _uncorroborated_regulatory_claim(self, sentence: str, *, cites_document: bool) -> bool:
        """The sentence states a regulatory action that at most one document wording supports.

        Documents "carry" such a claim when the same pattern matches in them (not negated); several documents with
        the same wording around it (a copied or syndicated item) count once. An uncited sentence counts only when
        some document carries a claim, i.e. when it can have come from one."""
        folded = fold(sentence)
        stated = [match for match in _REGULATORY.finditer(folded) if not _negated(folded, match.start())]
        if not stated:
            return False
        contexts = set()
        for item in self.documents:
            for field in (item.text_excerpt, item.title):
                document = fold(field or "")
                match = next((m for m in _REGULATORY.finditer(document) if not _negated(document, m.start())), None)
                if match is not None:
                    contexts.add(_compact(document[max(0, match.start() - 12) : match.end() + 12]))
                    break
        if not contexts and not cites_document:
            return False
        return len(contexts) < 2

    def _single_document_figure(self, sentence: str) -> bool:
        """(round 9, E3) The sentence states a figure (a number with a unit: percent, 亿/万/元, 倍, 点, bn, yuan …)
        that the run's structured data does not contain and that exactly one document wording carries.

        This is the general form of the round-8 rules: whatever the figure is about (an insider's growth figure, a
        poll, a buyback size, a footnote "restatement", a statistic), one document is its only source, so it is
        relayed with the layer's own marker, in the answer and in every key point. Figures the structured data
        confirms, figures two differently worded documents state, and figures no document states (derived or
        computed numbers: the verifier's business) are left alone."""
        figures = unit_figures(sentence)
        if not figures:
            return False
        for value, scales, rounding in figures:
            if _is_supported(value, self.structured_numbers, scales, rounding):
                continue
            if _is_supported(value, self.corroborating_numbers(), scales, rounding):
                continue  # (round 10, F9) the named company's fundamentals confirm it
            carriers = {
                context
                for evidence_id, document_figures in self.document_unit_figures.items()
                for other, _scales, _rounding, context in document_figures
                if _is_supported(value, [other], scales, rounding)
            }
            if len(carriers) == 1:
                return True
        return False

    def corroborating_numbers(self) -> list[tuple[float, bool]]:
        """The fundamentals numbers from ``corroborate``, fetched once, on first need."""
        if self._corroborating is None:
            try:
                self._corroborating = list(self._corroborate()) if self._corroborate is not None else []
            except Exception:  # a failed lookup leaves the rule as it was: attribute
                self._corroborating = []
        return self._corroborating

    def _attribute(self, sentence: str) -> str:
        lead = sentence[: len(sentence) - len(sentence.lstrip())]
        body = sentence[len(lead) :]
        tail = _SENTENCE_TAIL.search(body)
        core, end = (body[: tail.start()], tail.group("tail")) if tail else (body, "")
        if self.zh:
            if not _REPORTED.search(core):
                connector = _CONNECTOR_OPENING.match(core)
                if connector and connector.end() < len(core):
                    core = core[: connector.end()] + ATTRIBUTION_PREFIX_ZH + core[connector.end() :]
                else:
                    core = ATTRIBUTION_PREFIX_ZH + core
            core = core.rstrip("，,；; ") + ATTRIBUTION_SUFFIX_ZH
        else:
            core = core.rstrip(",; ") + ATTRIBUTION_SUFFIX_EN
        return lead + core + end


@dataclass(frozen=True)
class Figure:
    """A reported amount ("归母净利润912.6亿元") in yuan, with the period and company it is stated for (``None``
    when the text does not say)."""

    metric: str
    value: float
    tolerance: float
    year: str | None
    period: str
    entity: str | None

    def comparable(self, other: Figure) -> bool:
        return (
            self.metric == other.metric
            and self.period == other.period
            and (self.year is None or other.year is None or self.year == other.year)
            and (self.entity is None or other.entity is None or self.entity == other.entity)
        )

    def agrees(self, other: Figure) -> bool:
        return abs(self.value - other.value) <= self.tolerance + other.tolerance


_PERIOD_LABELS = (
    ("9M", re.compile(r"前三季度|nine\s+months", re.I)),
    ("Q1", re.compile(r"第?一季度|\bQ1\b|first\s+quarter", re.I)),
    ("H1", re.compile(r"上半年|半年度|中期|\bH1\b|first\s+half", re.I)),
    ("Q2", re.compile(r"第?二季度|\bQ2\b|second\s+quarter", re.I)),
    ("Q3", re.compile(r"第?三季度|\bQ3\b|third\s+quarter", re.I)),
    ("Q4", re.compile(r"第?四季度|\bQ4\b|fourth\s+quarter", re.I)),
)
_REPORT_DATE_PERIOD = {"12-31": "FY", "03-31": "Q1", "06-30": "H1", "09-30": "9M"}


def _period(prefix: str) -> str:
    """The sub-period named closest before the metric in the sentence, else the full year ("FY")."""
    best, label = -1, "FY"
    for name, pattern in _PERIOD_LABELS:
        for match in pattern.finditer(prefix):
            if match.start() > best:
                best, label = match.start(), name
    return label


def _entity_names(store: EvidenceStore) -> dict[str, str]:
    """Company names in the run's structured evidence (``name``, or a title like "贵州茅台 (600519.SH) fundamentals")
    and their short forms ("贵州茅台" -> also "茅台"), mapped to the full name."""
    names: dict[str, str] = {}
    for item in store.items():
        if item.kind != "structured":
            continue
        title = re.match(r"\s*([^()（）]{2,20}?)\s*[（(]\s*\d{6}\.[A-Z]{2}\s*[)）]", item.title or "")
        name = str((item.payload or {}).get("name") or (title.group(1) if title else "")).strip()
        if len(name) < 2:
            continue
        names[name] = name
        if len(name) >= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", name):
            names.setdefault(name[2:], name)
    return names


def _entity(prefix: str, names: dict[str, str]) -> str | None:
    best, entity = -1, None
    for alias, name in names.items():
        position = prefix.rfind(alias)
        if position > best:
            best, entity = position, name
    return entity


def amount_figures(text: str, names: dict[str, str] | None = None) -> list[Figure]:
    """Net profit / revenue levels stated in ``text``: the first number with an amount unit within 30 characters
    after the metric name, unless a change or a prior period is named in between ("同比下降4.53%",
    "较上年减少", "a year earlier") or just before it ("上年同期净利润")."""
    names = names or {}
    cleaned = fold(_CITATION.sub(" ", text or ""))
    found: list[Figure] = []
    for metric, pattern in _AMOUNT_METRICS.items():
        for match in pattern.finditer(cleaned):
            window = re.split(r"[。；;！!？?\n]", cleaned[match.end() : match.end() + 30])[0]
            number = _AMOUNT_NUMBER.search(window)
            if number is None or not number.group("unit"):
                continue
            before = cleaned[max(0, match.start() - 6) : match.start()]
            if _CHANGE_OR_PRIOR.search(window[: number.start()]) or _CHANGE_OR_PRIOR.search(before):
                continue
            if window[number.end() :].lstrip().startswith(("%", "％")):
                continue
            unit = number.group("unit").lower().rstrip("元") or "元"
            scale = _AMOUNT_UNITS.get(unit, 1.0)
            digits = number.group("num").replace(",", "")
            decimals = len(digits.split(".")[1]) if "." in digits else 0
            value = float(digits) * scale
            tolerance = max(0.5 * 10**-decimals * scale, 0.005 * value)
            # the whole sentence up to the metric ("2025年年度报告，…；归母净利润…" states the 2025 figure)
            sentence_start = max(cleaned.rfind(mark, 0, match.start()) for mark in "。!?！？\n")
            prefix = cleaned[sentence_start + 1 : match.start()]
            years = [year for year in _YEAR.findall(prefix)]
            found.append(
                Figure(metric, value, tolerance, years[-1] if years else None, _period(prefix), _entity(prefix, names))
            )
    return found


def _structured_amounts(store: EvidenceStore, names: dict[str, str]) -> list[Figure]:
    """Net profit / revenue in the run's structured evidence (yuan), with the report period when it is stated."""
    figures: list[Figure] = []
    for item in store.items():
        if item.kind != "structured":
            continue
        payload = item.payload or {}
        report_date = str(payload.get("report_date") or "")
        match = re.fullmatch(r"((?:19|20)\d{2})-(\d{2}-\d{2})", report_date[:10])
        if not match or match.group(2) not in _REPORT_DATE_PERIOD:
            continue  # no period: cannot tell which year's figure it is
        year, period = match.group(1), _REPORT_DATE_PERIOD[match.group(2)]
        title = re.match(r"\s*([^()（）]{2,20}?)\s*[（(]\s*\d{6}\.[A-Z]{2}\s*[)）]", item.title or "")
        name = str(payload.get("name") or (title.group(1) if title else "")).strip()
        values = dict(payload.get("metrics") or {}) if isinstance(payload.get("metrics"), dict) else {}
        values.update({key: value for key, value in payload.items() if key not in values})
        for metric, key_pattern in _AMOUNT_KEYS.items():
            for key, raw in values.items():
                if not key_pattern.match(str(key)) or isinstance(raw, bool):
                    continue
                try:
                    value = float(str(raw).replace(",", ""))
                except ValueError:
                    continue
                entity = names.get(name, name or None)
                figures.append(Figure(metric, value, 0.005 * abs(value), year, period, entity))
    return figures


# (round 9) A figure: a number written with a unit (percent, amount, multiple, index points, per-share price) or after a
# currency sign. Bare numbers (counts, scores, list markers) are not figures; dates and parameters are removed first.
_FIGURE_UNIT = re.compile(
    r"^\s*(?:%|％|个百分点|百分点|万亿|亿|千万|百万|万|元|块钱|块|倍|点(?!钟|半)|bp\b|基点|pct\b|per\s*cent\b|"
    r"percentage\s+points?\b|trillion\b|billion\b|bn\b|million\b|mn\b|yuan\b|rmb\b|cny\b|x\b|times\b|points?\b)",
    re.IGNORECASE,
)
_CURRENCY_BEFORE = re.compile(r"(?:CNY|RMB|US\$|\$|¥|￥)\s*$", re.IGNORECASE)


def unit_figures(text: str, *, with_context: bool = False) -> list[tuple]:
    """``(value, scales, rounding)`` for every figure in ``text`` (see ``_FIGURE_UNIT``); with ``with_context`` also the
    compacted text around it, so copies of one wording count as one source."""
    cleaned = _cleaned(fold(text or ""))
    found: list[tuple] = []
    for match in _NUMBER_TOKEN.finditer(cleaned):
        tail = cleaned[match.end() : match.end() + 24]
        head = cleaned[max(0, match.start() - 5) : match.start()]
        if not (_FIGURE_UNIT.match(tail) or _CURRENCY_BEFORE.search(head)):
            continue
        token = match.group(0).replace(",", "").lstrip("+-")
        try:
            value = float(token)
        except ValueError:
            continue
        if value == 0:
            continue
        scales = next((scales for pattern, scales in _UNIT_SCALES if pattern.search(tail)), _BARE_SCALES)
        decimals = len(token.split(".")[1]) if "." in token else 0
        item: tuple = (value, scales, 0.5 * 10**-decimals)
        if with_context:
            item = (*item, _compact(cleaned[max(0, match.start() - 12) : match.end() + 12]))
        found.append(item)
    # (round 10, F3) a figure in Chinese numerals with its unit ("百分之三十五", "三成", "十二亿元", "三十倍") is the
    # same figure as its Arabic form: a headline or a sentence that spells it out is checked like "35%"
    for position, value, scales, rounding in _chinese_values(cleaned):
        item = (value, scales, rounding)
        if with_context:
            item = (*item, _compact(cleaned[max(0, position - 12) : position + 16]))
        found.append(item)
    return found


def unconfirmed_figures(text: str, numbers: list[tuple[float, bool]]) -> list[float]:
    """Figures in ``text`` (``unit_figures``) that none of ``numbers`` (the run's structured values) supports."""
    return [
        value for value, scales, rounding in unit_figures(text) if not _is_supported(value, numbers, scales, rounding)
    ]


def structured_numbers(store: EvidenceStore) -> list[tuple[float, bool]]:
    """Every number in the structured payloads of the run (tool data, never document text), unsigned."""
    values: list[float] = []
    for item in store.items():
        if item.kind == "structured":
            _collect_numbers(item.payload, values)
    return [(value, False) for value in values]


def _sentence_key(sentence: str) -> str:
    return _compact(_CITATION.sub("", sentence))


def _sentences(text: str) -> list[str]:
    """``whole_sentences``, with a domain split at its spaced dots ("ping-an-insider . example . com") rejoined."""
    merged: list[str] = []
    for piece in whole_sentences(text):
        if merged and re.search(r"[a-z0-9-]\s*\.\s*$", merged[-1]) and re.match(r"[a-z0-9]", piece):
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def _document_text(item: AgentEvidence) -> str:
    return f"{item.title or ''} {item.text_excerpt or ''}"


def _compact(text: str) -> str:
    """Folded, lower-cased, without whitespace and the separators used to split phone numbers or handles."""
    return re.sub(r"[\s\-–—.·•_|/\\~～()（）]+", "", fold(text).lower())
