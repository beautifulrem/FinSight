from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENTITY_FIELDS = [
    "entity_id",
    "canonical_name",
    "normalized_name",
    "symbol",
    "market",
    "exchange",
    "entity_type",
    "industry_name",
    "status",
]

ALIAS_FIELDS = [
    "alias_id",
    "entity_id",
    "alias_text",
    "normalized_alias",
    "alias_type",
    "priority",
    "is_official",
]

_NAME_CHARS = r"\u4e00-\u9fffA-Za-z0-9·\-\*"
_CODE_PATTERN = r"\d{6}(?:\.(?:SH|SZ|BJ)|\s*CH)?"
_NAME_CODE_RE = re.compile(rf"(?P<name>[{_NAME_CHARS}]{{2,24}})\s*[（(]\s*(?P<code>{_CODE_PATTERN})\s*[)）]", re.IGNORECASE)
_CODE_LABELED_NAME_RE = re.compile(rf"(?P<code>\d{{6}}\.(?:SH|SZ|BJ)|\d{{6}}\s+CH)\s+(?:股票简称|证券简称|基金简称|简称)\s+(?P<name>[{_NAME_CHARS}]{{2,24}})", re.IGNORECASE)
_CODE_NAME_RE = re.compile(rf"(?P<code>\d{{6}}\.(?:SH|SZ|BJ)|\d{{6}}\s+CH)\s+(?P<name>[{_NAME_CHARS}]{{2,24}})", re.IGNORECASE)

_STOCK_SUFFIX_RE = re.compile(r"(?:[-－](?:U|W|WD|B|S|R))+$", re.IGNORECASE)
_INVALID_NAME_PARTS = {
    "股票代码",
    "股票简称",
    "证券代码",
    "证券简称",
    "报告期",
    "联系人",
    "联系方式",
    "有限公司",
    "股份有限公司",
}
_INVALID_EXACT_NAMES = {
    "买入",
    "卖出",
    "增持",
    "减持",
    "中性",
    "推荐",
    "维持",
    "评级",
    "目标价",
    "收盘价",
}

CURATED_RUNTIME_ENTITIES = [
    {
        "canonical_name": "上证指数",
        "symbol": "000001.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["上证指数", "上证综指", "沪指", "大盘"],
    },
    {
        "canonical_name": "深证成指",
        "symbol": "399001.SZ",
        "market": "CN",
        "exchange": "SZSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["深证成指", "深成指", "深圳成指"],
    },
    {
        "canonical_name": "创业板指",
        "symbol": "399006.SZ",
        "market": "CN",
        "exchange": "SZSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["创业板指", "创业板指数", "创业板"],
    },
    {
        "canonical_name": "科创50",
        "symbol": "000688.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["科创50", "科创50指数", "上证科创板50"],
    },
    {
        "canonical_name": "沪深300",
        "symbol": "000300.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["沪深300", "沪深300指数", "CSI300"],
    },
    {
        "canonical_name": "中证500",
        "symbol": "000905.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["中证500", "中证500指数", "CSI500"],
    },
    {
        "canonical_name": "中证1000",
        "symbol": "000852.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["中证1000", "中证1000指数", "CSI1000"],
    },
    {
        "canonical_name": "上证50",
        "symbol": "000016.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["上证50", "上证50指数", "沪市50"],
    },
    {
        "canonical_name": "北证50",
        "symbol": "899050.BJ",
        "market": "CN",
        "exchange": "BSE",
        "entity_type": "index",
        "industry_name": "宽基指数",
        "aliases": ["北证50", "北证50指数", "北交所50"],
    },
    {
        "canonical_name": "恒生指数",
        "symbol": "HSI.HK",
        "market": "HK",
        "exchange": "HKEX",
        "entity_type": "index",
        "industry_name": "海外指数",
        "aliases": ["恒生指数", "恒指", "港股大盘"],
    },
    {
        "canonical_name": "纳斯达克综合指数",
        "symbol": "IXIC.US",
        "market": "US",
        "exchange": "NASDAQ",
        "entity_type": "index",
        "industry_name": "海外指数",
        "aliases": ["纳斯达克", "纳指", "纳斯达克指数", "纳斯达克综合指数"],
    },
    {
        "canonical_name": "标普500",
        "symbol": "SPX.US",
        "market": "US",
        "exchange": "NYSE",
        "entity_type": "index",
        "industry_name": "海外指数",
        "aliases": ["标普500", "标普500指数", "标普", "S&P500", "SP500"],
    },
    {
        "canonical_name": "道琼斯工业平均指数",
        "symbol": "DJI.US",
        "market": "US",
        "exchange": "NYSE",
        "entity_type": "index",
        "industry_name": "海外指数",
        "aliases": ["道琼斯", "道指", "道琼斯指数", "道琼斯工业平均指数"],
    },
]

for _sector_name, _aliases in {
    "白酒": ["白酒", "白酒板块", "白酒行业"],
    "半导体": ["半导体", "半导体板块", "芯片", "芯片板块"],
    "银行": ["银行", "银行板块", "银行业"],
    "保险": ["保险", "保险板块", "保险业"],
    "证券": ["证券", "证券板块", "券商", "券商板块"],
    "医药": ["医药", "医药板块", "医药行业"],
    "新能源": ["新能源", "新能源板块", "新能源行业"],
    "光伏": ["光伏", "光伏板块", "光伏行业"],
    "煤炭": ["煤炭", "煤炭板块", "煤炭行业"],
    "有色金属": ["有色", "有色板块", "有色金属", "有色金属板块"],
    "地产": ["地产", "地产板块", "房地产", "房地产板块"],
    "军工": ["军工", "军工板块", "国防军工"],
    "汽车行业": ["汽车板块", "汽车行业"],
    "人工智能": ["AI", "AI板块", "人工智能", "人工智能板块"],
    "机器人": ["机器人", "机器人板块"],
    "消费": ["消费", "消费板块", "大消费"],
    "科技": ["科技", "科技板块", "科技股"],
}.items():
    CURATED_RUNTIME_ENTITIES.append(
        {
            "canonical_name": _sector_name,
            "symbol": "",
            "market": "CN",
            "exchange": "",
            "entity_type": "sector",
            "industry_name": _sector_name,
            "aliases": _aliases,
        }
    )

for _macro_name, _aliases in {
    "CPI": ["CPI", "居民消费价格指数", "消费者价格指数"],
    "PPI": ["PPI", "工业生产者出厂价格指数"],
    "PMI": ["PMI", "采购经理指数", "制造业PMI"],
    "GDP": ["GDP", "国内生产总值"],
    "M2": ["M2", "广义货币", "广义货币供应量"],
    "社会融资规模": ["社融", "社会融资规模"],
    "LPR": ["LPR", "贷款市场报价利率"],
    "MLF": ["MLF", "中期借贷便利"],
    "逆回购": ["逆回购", "公开市场逆回购"],
    "十年期国债收益率": ["十年期国债收益率", "10年期国债收益率", "十年国债收益率"],
    "人民币汇率": ["汇率", "人民币汇率", "美元兑人民币"],
    "美元指数": ["美元指数", "DXY"],
    "通胀": ["通胀", "通货膨胀"],
}.items():
    CURATED_RUNTIME_ENTITIES.append(
        {
            "canonical_name": _macro_name,
            "symbol": "",
            "market": "CN",
            "exchange": "",
            "entity_type": "macro_indicator",
            "industry_name": "宏观指标",
            "aliases": _aliases,
        }
    )

for _policy_name, _aliases in {
    "降息": ["降息", "降低利率"],
    "降准": ["降准", "降低存款准备金率", "下调存款准备金率"],
}.items():
    CURATED_RUNTIME_ENTITIES.append(
        {
            "canonical_name": _policy_name,
            "symbol": "",
            "market": "CN",
            "exchange": "",
            "entity_type": "policy",
            "industry_name": "货币政策",
            "aliases": _aliases,
        }
    )


@dataclass(frozen=True)
class RuntimeEntitySummary:
    entity_count: int
    alias_count: int
    added_entity_count: int
    added_alias_count: int


class RuntimeEntityAssetBuilder:
    """Build runtime entity/alias assets from seed files plus public training assets."""

    def __init__(
        self,
        seed_entities: list[dict[str, str]],
        seed_aliases: list[dict[str, str]],
        training_assets_dir: str | Path,
        max_training_pairs: int = 80_000,
    ) -> None:
        self.seed_entities = [dict(row) for row in seed_entities]
        self.seed_aliases = [dict(row) for row in seed_aliases]
        self.training_assets_dir = Path(training_assets_dir)
        self.max_training_pairs = max_training_pairs

    def build(self, extra_universe_rows: Iterable[dict[str, str]] | None = None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        entities = [self._normalize_entity_row(row) for row in self.seed_entities]
        valid_entity_ids = {row["entity_id"] for row in entities if row.get("entity_id")}
        aliases = [
            self._normalize_alias_row(row)
            for row in self.seed_aliases
            if str(row.get("entity_id", "") or "") in valid_entity_ids
        ]

        entities_by_symbol = {row["symbol"]: row for row in entities if row.get("symbol")}
        entities_by_scope_key = {
            _scope_entity_key(row): row
            for row in entities
            if not row.get("symbol")
        }
        aliases_by_entity: dict[str, set[str]] = defaultdict(set)
        alias_owner_by_normalized: dict[str, str] = {}
        for row in aliases:
            aliases_by_entity[row["entity_id"]].add(row["normalized_alias"])
            alias_owner_by_normalized.setdefault(row["normalized_alias"], row["entity_id"])

        next_entity_id = self._next_int(entities, "entity_id")
        next_alias_id = self._next_int(aliases, "alias_id")
        next_entity_id, next_alias_id = self._add_curated_assets(
            entities=entities,
            entities_by_symbol=entities_by_symbol,
            entities_by_scope_key=entities_by_scope_key,
            aliases=aliases,
            aliases_by_entity=aliases_by_entity,
            alias_owner_by_normalized=alias_owner_by_normalized,
            next_entity_id=next_entity_id,
            next_alias_id=next_alias_id,
        )
        for pair in self._iter_candidate_pairs(extra_universe_rows or []):
            if not pair.get("symbol") or not pair.get("canonical_name"):
                continue
            symbol = pair["symbol"]
            canonical_name = pair["canonical_name"]
            if symbol in entities_by_symbol:
                entity = entities_by_symbol[symbol]
            else:
                entity = {
                    "entity_id": str(next_entity_id),
                    "canonical_name": canonical_name,
                    "normalized_name": canonical_name,
                    "symbol": symbol,
                    "market": "CN",
                    "exchange": _exchange_for_symbol(symbol),
                    "entity_type": pair.get("entity_type") or _infer_entity_type(symbol, canonical_name),
                    "industry_name": pair.get("industry_name", ""),
                    "status": "active",
                }
                entities.append(entity)
                entities_by_symbol[symbol] = entity
                next_entity_id += 1

            for alias_text, alias_type, priority in _aliases_for_pair(canonical_name, symbol, pair.get("raw_name", "")):
                normalized_alias = alias_text.strip()
                if not normalized_alias or normalized_alias in aliases_by_entity[entity["entity_id"]]:
                    continue
                existing_owner = alias_owner_by_normalized.get(normalized_alias)
                if existing_owner and existing_owner != entity["entity_id"]:
                    continue
                aliases.append(
                    {
                        "alias_id": str(next_alias_id),
                        "entity_id": entity["entity_id"],
                        "alias_text": alias_text,
                        "normalized_alias": normalized_alias,
                        "alias_type": alias_type,
                        "priority": str(priority),
                        "is_official": "true" if alias_type == "official_name" else "false",
                    }
                )
                aliases_by_entity[entity["entity_id"]].add(normalized_alias)
                alias_owner_by_normalized[normalized_alias] = entity["entity_id"]
                next_alias_id += 1

        entities.sort(key=lambda row: (row.get("entity_type", ""), row.get("symbol", ""), row.get("canonical_name", "")))
        aliases.sort(key=lambda row: int(row["alias_id"]))
        return entities, aliases

    def _add_curated_assets(
        self,
        *,
        entities: list[dict[str, str]],
        entities_by_symbol: dict[str, dict[str, str]],
        entities_by_scope_key: dict[tuple[str, str], dict[str, str]],
        aliases: list[dict[str, str]],
        aliases_by_entity: dict[str, set[str]],
        alias_owner_by_normalized: dict[str, str],
        next_entity_id: int,
        next_alias_id: int,
    ) -> tuple[int, int]:
        for row in CURATED_RUNTIME_ENTITIES:
            symbol = str(row.get("symbol", "") or "")
            key = (str(row["entity_type"]), str(row["canonical_name"]))
            if symbol and symbol in entities_by_symbol:
                entity = entities_by_symbol[symbol]
            elif not symbol and key in entities_by_scope_key:
                entity = entities_by_scope_key[key]
            else:
                entity = {
                    "entity_id": str(next_entity_id),
                    "canonical_name": str(row["canonical_name"]),
                    "normalized_name": str(row["canonical_name"]),
                    "symbol": symbol,
                    "market": str(row.get("market", "CN") or "CN"),
                    "exchange": str(row.get("exchange", "") or ""),
                    "entity_type": str(row["entity_type"]),
                    "industry_name": str(row.get("industry_name", "") or ""),
                    "status": "active",
                }
                entities.append(entity)
                if symbol:
                    entities_by_symbol[symbol] = entity
                else:
                    entities_by_scope_key[key] = entity
                next_entity_id += 1

            for index, alias_text in enumerate(row.get("aliases", [])):
                normalized_alias = str(alias_text).strip()
                if not normalized_alias or normalized_alias in aliases_by_entity[entity["entity_id"]]:
                    continue
                existing_owner = alias_owner_by_normalized.get(normalized_alias)
                if existing_owner and existing_owner != entity["entity_id"]:
                    continue
                alias_type = "official_name" if normalized_alias == entity["canonical_name"] else "common_alias"
                aliases.append(
                    {
                        "alias_id": str(next_alias_id),
                        "entity_id": entity["entity_id"],
                        "alias_text": normalized_alias,
                        "normalized_alias": normalized_alias,
                        "alias_type": alias_type,
                        "priority": "1" if index == 0 else "2",
                        "is_official": "true" if alias_type == "official_name" else "false",
                    }
                )
                aliases_by_entity[entity["entity_id"]].add(normalized_alias)
                alias_owner_by_normalized[normalized_alias] = entity["entity_id"]
                next_alias_id += 1
        return next_entity_id, next_alias_id

    def write(self, output_dir: str | Path, extra_universe_rows: Iterable[dict[str, str]] | None = None) -> RuntimeEntitySummary:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        entities, aliases = self.build(extra_universe_rows=extra_universe_rows)
        _write_csv(output_path / "entity_master.csv", entities, ENTITY_FIELDS)
        _write_csv(output_path / "alias_table.csv", aliases, ALIAS_FIELDS)
        return RuntimeEntitySummary(
            entity_count=len(entities),
            alias_count=len(aliases),
            added_entity_count=max(0, len(entities) - len(self.seed_entities)),
            added_alias_count=max(0, len(aliases) - len(self.seed_aliases)),
        )

    def _iter_candidate_pairs(self, extra_universe_rows: Iterable[dict[str, str]]) -> Iterable[dict[str, str]]:
        seen_keys: set[tuple[str, str]] = set()
        for row in extra_universe_rows:
            pair = _pair_from_universe_row(row)
            if pair and (pair["symbol"], pair["canonical_name"]) not in seen_keys:
                seen_keys.add((pair["symbol"], pair["canonical_name"]))
                yield pair

        pair_counter: Counter[tuple[str, str, str]] = Counter()
        raw_names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for symbol, name in self._iter_training_symbol_name_pairs():
            canonical_name = _canonicalize_security_name(name)
            if not _is_valid_security_name(canonical_name):
                continue
            entity_type = _infer_entity_type(symbol, canonical_name)
            key = (symbol, canonical_name, entity_type)
            pair_counter[key] += 1
            raw_names[key].add(name)
        for (symbol, canonical_name, entity_type), _count in pair_counter.most_common(self.max_training_pairs):
            yield {
                "symbol": symbol,
                "canonical_name": canonical_name,
                "raw_name": sorted(raw_names[(symbol, canonical_name, entity_type)], key=len)[-1],
                "entity_type": entity_type,
                "industry_name": "",
            }

    def _iter_training_symbol_name_pairs(self) -> Iterable[tuple[str, str]]:
        alias_path = self.training_assets_dir / "alias_catalog.jsonl"
        if alias_path.exists():
            with alias_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    symbol = _normalize_symbol(str(row.get("symbol") or row.get("security_code") or ""))
                    name = str(row.get("company_name") or row.get("alias_text") or "").strip()
                    if symbol and name:
                        yield symbol, name

        corpus_path = self.training_assets_dir / "retrieval_corpus.jsonl"
        if corpus_path.exists():
            with corpus_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = "\n".join(str(row.get(field) or "") for field in ("title", "body", "text"))
                    yield from _extract_pairs_from_text(text)

    @staticmethod
    def _normalize_entity_row(row: dict[str, str]) -> dict[str, str]:
        normalized = {field: str(row.get(field, "") or "") for field in ENTITY_FIELDS}
        normalized["symbol"] = _normalize_symbol(normalized["symbol"]) or normalized["symbol"]
        return normalized

    @staticmethod
    def _normalize_alias_row(row: dict[str, str]) -> dict[str, str]:
        normalized = {field: str(row.get(field, "") or "") for field in ALIAS_FIELDS}
        normalized["normalized_alias"] = normalized["normalized_alias"] or normalized["alias_text"]
        return normalized

    @staticmethod
    def _next_int(rows: list[dict[str, str]], field: str) -> int:
        return max((int(row.get(field, 0) or 0) for row in rows), default=0) + 1


def fetch_akshare_security_universe(ak_module: object | None = None) -> list[dict[str, str]]:
    if ak_module is None:
        import akshare as ak_module  # type: ignore[no-redef]

    rows: list[dict[str, str]] = []
    rows.extend(_fetch_akshare_table(ak_module, "stock_info_a_code_name", entity_type="stock"))
    rows.extend(_fetch_akshare_table(ak_module, "fund_etf_spot_em", entity_type="etf"))
    rows.extend(_fetch_akshare_table(ak_module, "fund_name_em", entity_type="fund"))
    return rows


def _fetch_akshare_table(ak_module: object, function_name: str, entity_type: str) -> list[dict[str, str]]:
    fn = getattr(ak_module, function_name, None)
    if fn is None:
        return []
    try:
        frame = fn()
    except Exception:  # noqa: BLE001
        return []
    records = frame.to_dict("records") if hasattr(frame, "to_dict") else []
    rows = []
    for record in records:
        code = _first_present(record, ["code", "代码", "基金代码", "证券代码"])
        name = _first_present(record, ["name", "名称", "基金简称", "简称", "证券简称"])
        symbol = _normalize_symbol(code)
        canonical_name = _canonicalize_security_name(name)
        if symbol and _is_valid_security_name(canonical_name):
            rows.append({"symbol": symbol, "canonical_name": canonical_name, "raw_name": name, "entity_type": entity_type})
    return rows


def _first_present(record: dict, keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _pair_from_universe_row(row: dict[str, str]) -> dict[str, str] | None:
    symbol = _normalize_symbol(str(row.get("symbol") or row.get("code") or ""))
    canonical_name = _canonicalize_security_name(str(row.get("canonical_name") or row.get("name") or ""))
    if not symbol or not _is_valid_security_name(canonical_name):
        return None
    return {
        "symbol": symbol,
        "canonical_name": canonical_name,
        "raw_name": str(row.get("raw_name") or canonical_name),
        "entity_type": row.get("entity_type") or _infer_entity_type(symbol, canonical_name),
        "industry_name": row.get("industry_name", ""),
    }


def _scope_entity_key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("entity_type", "") or ""), str(row.get("canonical_name", "") or ""))


def _extract_pairs_from_text(text: str) -> Iterable[tuple[str, str]]:
    if not text:
        return
    compact = re.sub(r"\s+", " ", text)
    for match in _NAME_CODE_RE.finditer(compact):
        symbol = _normalize_symbol(match.group("code"))
        name = _clean_security_name(match.group("name"))
        if symbol and name:
            yield symbol, name
    for match in _CODE_LABELED_NAME_RE.finditer(compact):
        symbol = _normalize_symbol(match.group("code"))
        name = _clean_security_name(match.group("name"))
        if symbol and name:
            yield symbol, name
    for match in _CODE_NAME_RE.finditer(compact):
        symbol = _normalize_symbol(match.group("code"))
        name = _clean_security_name(match.group("name"))
        if symbol and name:
            yield symbol, name


def _clean_security_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "", name).strip("，。；;:：、()（）[]【】")
    cleaned = re.sub(r"^(股票简称|证券简称|公司简称|标的公司)", "", cleaned)
    return cleaned


def _canonicalize_security_name(name: str) -> str:
    cleaned = _clean_security_name(name)
    return _STOCK_SUFFIX_RE.sub("", cleaned)


def _is_valid_security_name(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 18:
        return False
    if not any("\u4e00" <= char <= "\u9fff" for char in name):
        return False
    if any(part in name for part in _INVALID_NAME_PARTS):
        return False
    if name in _INVALID_EXACT_NAMES:
        return False
    if name.isdigit():
        return False
    return True


def _normalize_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.strip().upper().replace("_", ".")
    if not symbol:
        return ""
    symbol = re.sub(r"\s+", "", symbol)
    if symbol.endswith("CH"):
        symbol = symbol[:-2]
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", symbol):
        return symbol
    if re.fullmatch(r"\d{6}", symbol):
        return f"{symbol}.{_exchange_suffix_for_code(symbol)}"
    return ""


def _exchange_suffix_for_code(code: str) -> str:
    if code.startswith(("6", "5")):
        return "SH"
    if code.startswith(("8", "4")):
        return "BJ"
    return "SZ"


def _exchange_for_symbol(symbol: str) -> str:
    if symbol.endswith(".SH"):
        return "SSE"
    if symbol.endswith(".SZ"):
        return "SZSE"
    if symbol.endswith(".BJ"):
        return "BSE"
    return ""


def _infer_entity_type(symbol: str, name: str) -> str:
    upper_name = name.upper()
    code = symbol[:6]
    if "ETF" in upper_name or "LOF" in upper_name:
        return "etf"
    if "基金" in name:
        return "fund"
    if "指数" in name:
        return "index"
    if code.startswith(("51", "56", "58", "15")):
        return "etf"
    return "stock"


def _aliases_for_pair(canonical_name: str, symbol: str, raw_name: str) -> list[tuple[str, str, int]]:
    aliases = [
        (canonical_name, "official_name", 1),
        (symbol, "symbol", 1),
        (symbol[:6], "symbol", 2),
    ]
    if raw_name and raw_name != canonical_name:
        aliases.append((raw_name, "common_alias", 2))
    stripped = _STOCK_SUFFIX_RE.sub("", raw_name or canonical_name)
    if stripped and stripped != canonical_name:
        aliases.append((stripped, "common_alias", 2))
    deduped: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for alias_text, alias_type, priority in aliases:
        alias_text = alias_text.strip()
        if alias_text and alias_text not in seen:
            seen.add(alias_text)
            deduped.append((alias_text, alias_type, priority))
    return deduped


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
