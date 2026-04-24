from __future__ import annotations


INDUSTRY_TERMS = [
    "白酒",
    "保险",
    "券商",
    "消费",
    "银行",
    "科技",
    "地产",
    "成长指数",
    "宽基指数",
]

BUCKET_MARKET_TERMS = [
    "行业",
    "板块",
    "大盘",
    *INDUSTRY_TERMS,
]


def contains_bucket_market_term(query: str) -> bool:
    for term in BUCKET_MARKET_TERMS:
        if term not in query:
            continue
        if term in INDUSTRY_TERMS and len(term) <= 2 and not _contains_standalone_short_industry(query, term):
            continue
        return True
    return False


def _contains_standalone_short_industry(query: str, term: str) -> bool:
    sector_markers = {"板块", "行业", "赛道", "方向", "主题"}
    start = 0
    while True:
        index = query.find(term, start)
        if index < 0:
            return False
        previous_char = query[index - 1] if index > 0 else ""
        next_text = query[index + len(term) : index + len(term) + 2]
        if not previous_char or not _is_cjk_char(previous_char) or next_text in sector_markers:
            return True
        start = index + len(term)


def _is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"
