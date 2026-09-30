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
   * a reported fundamental (ROE, EPS, dividend or book value per share) that another document states differently
   is attributed: "据一篇文档称，…（未经其他来源证实）" / "… (according to one document; not confirmed by other
   sources)." A sentence that already says the claim is unverified is left as it is. A fundamental that contradicts
   the run's structured data for the same metric is dropped with a note (the verifier flags the same conflict on
   LLM drafts, so this is the net for template answers and repaired drafts).

Target prices are trading calls (rule 1). The layer never adds a number or a claim; it only removes, replaces with
a fixed note, or wraps a sentence in attribution. ``evaluation/agent_eval/redteam.py`` records which detector
matches remain and whether they sit in attributed sentences.
"""

from __future__ import annotations

import re
from typing import Any

from ..text_safety import find_prohibited_promotion, fold
from .compliance import contains_trading_instruction
from .evidence import AgentEvidence, EvidenceStore
from .verifier import (
    _CITATION,
    _CONNECTOR_OPENING,
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

_REGULATORY = re.compile(
    r"立案(?:调查|侦查)?|行政处罚|处罚决定|监管函|警示函|纪律处分|公开谴责|(?:退市)?风险警示|(?-i:(?<![A-Za-z])\*?ST(?![A-Za-z]))|戴帽|"
    r"退市|终止上市|暂停上市|摘牌|停牌|强制清算|风险名单|财务造假|欺诈发行|涉嫌(?:违法|违规|犯罪|信息披露)|"
    r"\bdelist(?:ed|ing|ment)?\b|\b(?:trading\s+)?(?:suspension|halt)\b|\bsuspended\s+from\s+trading\b|"
    r"\b(?:probe|investigation|penalt(?:y|ies)|fined|sanction(?:s|ed)?)\b|\bforced\s+liquidation\b|"
    r"\brisk\s+(?:list|watch\s*list)\b|\baccounting\s+fraud\b|\bspecial\s+treatment\b|"
    # corporate actions that move a price as much as a regulator's decision ("合并已获批准", "to be acquired")
    r"(?:合并|吸收合并|重大资产重组|借壳|被收购|要约收购|私有化)(?:[^，。；]{0,8}(?:获批|批准|通过|完成|落地))?|"
    r"\bmerger\b|\b(?:to\s+be\s+)?acquired\s+by\b|\btakeover\s+(?:bid|offer)\b|\bgo(?:ing)?\s+private\b",
    re.IGNORECASE,
)
# A negation earlier in the same clause ("未发现立案调查或停牌信息", "no sign of a probe or delisting").
_NEGATION = re.compile(
    r"未(?:发现|见|曾|被|涉及|检索到|提及|显示)|没有|并未|不存在|并非|不会|无(?:任何|相关)?|"
    r"\bno\b|\bnot\b|\bwithout\b|\bnever\b|\bnone\b",
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
_REPORTED = re.compile(
    r"^\s*(?:据|有(?:媒体|报道|文章|消息)|另有|一(?:篇|则|条)|报道|文档)|称|according to|reported", re.I
)
_SENTENCE_TAIL = re.compile(
    r"(?P<tail>(?:\s*\[[^\[\]\s]{2,160}\])*\s*[。．.!?！？]?(?:\s*\[[^\[\]\s]{2,160}\])*\s*)$", re.DOTALL
)


def states_unverified(sentence: str) -> bool:
    """True when the sentence already tells the reader its claim is unverified or single-source."""
    return bool(_UNVERIFIED.search(sentence))


def scrub_answer(answer: dict[str, Any], store: EvidenceStore, *, zh: bool) -> tuple[dict[str, Any], list[str]]:
    """Return ``(answer, notes)``; notes name the rules that changed the answer (see the module docstring)."""
    context = _Context(store, zh=zh)
    guarded = dict(answer)
    text = str(answer.get("answer") or "")
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
    def __init__(self, store: EvidenceStore, *, zh: bool) -> None:
        self.store = store
        self.zh = zh
        self.documents = [item for item in store.items() if item.kind == "document"]
        self.raw_texts = {item.evidence_id: _document_text(item) for item in self.documents}
        self.document_texts = {evidence_id: _compact(text) for evidence_id, text in self.raw_texts.items()}
        self.notes: set[str] = set()
        self.emitted: set[str] = set()
        self.pending_notes: list[str] = []
        self.documents_disagree = False

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
            if (document_only or not cited) and self._documents_disagree(claim.metric, claim.value, claim.rounding):
                self.documents_disagree = True
                attribute = True
        if not attribute and (document_only or not cited):
            attribute = self._uncorroborated_regulatory_claim(sentence, cites_document=document_only)
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
