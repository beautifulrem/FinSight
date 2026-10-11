"""A small curated glossary of A-share market concepts.

Questions such as "北向资金是啥" or "什么是两融" ask about a market mechanism, not about a security. The classical
NLU has no entity for them, so they used to be refused as out of scope (round-3 item C6). The entries below are
curated text, committed with the code, and they are handed to the agent as ordinary evidence (the
``explain_concept`` tool): every answer built from one cites ``glossary_<term>``, so the definition is traceable
like any other evidence item.

The glossary explains concepts only. FinSight has no data series for most of them, and the tool says so, so a
question asking for a number ("北向资金今天净流入多少") gets the definition plus a clear statement that the number
is not in the data, never an invented figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlossaryEntry:
    """One curated concept: ``term`` is the headword, ``aliases`` other spellings used in questions."""

    term: str
    zh: str
    en: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    has_data_series: bool = False
    #: The word also has an everyday meaning ("国家队队员名单" is about sport): it needs a definition or market cue.
    needs_cue: bool = False
    #: (round 15, K12) A label of a figure the product shows ("扣非", "ROE（加权）"): only a definition question is
    #: about the concept; "扣非净利润是多少" asks for a company's figure and is clarified as before.
    definition_only: bool = False
    #: (round 16, L11) A metric name that also appears in ordinary data questions ("市值最大的是什么公司",
    #: "涨跌幅是多少"):
    #: the definition cue must be right next to the word ("什么是市盈率", "市值是什么意思", "What does P/B mean?").
    adjacent_cue: bool = False

    @property
    def evidence_id(self) -> str:
        return f"glossary_{self.term}"

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.term, *self.aliases)


GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry(
        term="北向资金",
        aliases=("北上资金", "northbound funds", "northbound capital"),
        zh=(
            "北向资金指境外投资者通过沪股通、深股通买卖A股的资金，是沪深港通机制中从香港流入内地市场的方向；"
            "与之相对的南向资金指内地资金通过港股通买卖港股。市场常用其净买入额观察外资态度。"
        ),
        en=(
            "Northbound funds are overseas money buying and selling A-shares through the Shanghai and Shenzhen "
            "Stock Connect, i.e. the Hong Kong-to-mainland direction of Stock Connect; southbound funds are the "
            "opposite direction. Their net buying is often read as a gauge of foreign appetite."
        ),
    ),
    GlossaryEntry(
        term="南向资金",
        aliases=("南下资金", "southbound funds"),
        zh="南向资金指内地投资者通过港股通买卖港股的资金，是沪深港通机制中从内地流向香港市场的方向。",
        en=(
            "Southbound funds are mainland money buying and selling Hong Kong stocks through Stock Connect, the "
            "mainland-to-Hong Kong direction of the scheme."
        ),
    ),
    GlossaryEntry(
        term="融资融券",
        aliases=("两融", "margin trading", "margin financing"),
        zh=(
            "融资融券是券商向投资者出借资金买入证券（融资）或出借证券供其卖出（融券）的信用交易业务，俗称两融。"
            "融资余额反映用借来的钱买股的规模，融券余额反映借券做空的规模；两者合称两融余额，是常用的杠杆指标。"
        ),
        en=(
            "Margin trading and securities lending is credit trading in which a broker lends cash to buy "
            "securities (margin financing) or lends securities to sell (securities lending). The margin balance "
            "shows how much stock is held with borrowed money and is a common gauge of market leverage."
        ),
    ),
    GlossaryEntry(
        term="国家队",
        aliases=("平准基金", "national team"),
        zh=(
            "国家队是市场对具有稳定市场职能的国有背景资金的统称，通常指中央汇金、中国证券金融公司等机构，"
            "以及它们持有或增持的宽基ETF。它不是一个法律概念，也没有公开的统一持仓口径。"
        ),
        en=(
            'The "national team" is a market nickname for state-linked funds with a market-stabilising role, '
            "typically Central Huijin and China Securities Finance, along with the broad-based ETFs they hold. It "
            "is not a legal category and has no single published holdings definition."
        ),
        needs_cue=True,
    ),
    GlossaryEntry(
        term="涨跌停",
        aliases=("涨停", "跌停", "price limit", "limit up", "limit down"),
        zh=(
            "涨跌停是A股的单日价格涨跌幅限制：价格达到上限称为涨停，达到下限称为跌停。限制幅度按板块和股票类型"
            "不同（主板最窄，创业板和科创板更宽，风险警示股更窄），新股上市初期另有规则。"
        ),
        en=(
            'A-shares have a daily price limit: hitting the upper bound is "limit up", the lower bound "limit '
            'down". The width depends on the board and the type of share (narrowest on the main board, wider on '
            "ChiNext and the STAR Market, narrower for risk-flagged shares), with separate rules for new listings."
        ),
    ),
    GlossaryEntry(
        term="换手率",
        aliases=("turnover rate",),
        zh="换手率是一段时间内成交股数占流通股本的比例，用来衡量交易活跃度；同一只股票的换手率高低只能与自身历史或同业比较。",
        en=(
            "The turnover rate is the number of shares traded in a period divided by the free-float share count, a "
            "measure of trading activity that is only meaningful against the same stock's history or its peers."
        ),
    ),
    GlossaryEntry(
        term="ST股",
        aliases=("st股", "风险警示股", "退市风险警示"),
        zh=(
            "ST是交易所对财务或其他状况异常公司加注的风险警示标记，加注后股票简称前带ST；*ST表示存在退市风险。"
            "此类股票的涨跌幅限制更严格，风险显著高于普通股票。"
        ),
        en=(
            "ST marks a company whose financial or other condition the exchange flags as abnormal; *ST marks "
            "delisting risk. Such shares carry tighter daily price limits and materially higher risk."
        ),
    ),
    GlossaryEntry(
        term="打新",
        aliases=("新股申购", "IPO subscription"),
        zh="打新指参与新股发行的申购。A股新股申购按市值配售，中签后才能认购，中签率通常很低。",
        en=(
            '"Playing the new" means subscribing to IPOs. A-share subscriptions are allocated by the market value '
            "of holdings and only successful lots can be bought; hit rates are typically very low."
        ),
    ),
    GlossaryEntry(
        term="限售解禁",
        aliases=("解禁", "lock-up expiry"),
        zh="限售解禁指首发或定增等原因形成的限售股到期可以上市流通。解禁本身只增加可流通股份，并不必然带来抛售。",
        en=(
            "A lock-up expiry is the date restricted shares from an IPO or placement become tradable. It increases "
            "the tradable float; it does not by itself mean the shares will be sold."
        ),
    ),
    GlossaryEntry(
        term="龙虎榜",
        aliases=("龙虎榜数据", "public trading list"),
        zh="龙虎榜是交易所对异常波动或成交活跃的股票公布的买卖前几名营业部席位明细，常被用来观察短线资金行为。",
        en=(
            'The "dragon-tiger list" is the exchange disclosure of the top buying and selling brokerage seats for '
            "stocks with unusual moves or heavy turnover, often read as a window on short-term flows."
        ),
    ),
    GlossaryEntry(
        term="主力资金",
        aliases=("主力", "大单资金", "smart money"),
        zh=(
            "主力资金是市场对大额买卖单背后资金的统称，通常按成交单笔金额划分为大单、中单、小单后统计净额。"
            "它是数据商的口径，不是监管定义，不同数据源的结果可能不同。"
        ),
        en=(
            '"Main-force money" is a market label for the money behind large orders: vendors bucket trades by '
            "order size and net them. It is a data-vendor convention, not a regulatory definition, so different "
            "sources disagree."
        ),
        needs_cue=True,
    ),
    GlossaryEntry(
        term="科创板",
        aliases=("STAR Market",),
        zh="科创板是上海证券交易所面向科技创新企业的板块，实行注册制，日常涨跌幅限制宽于主板。",
        en=(
            "The STAR Market is the Shanghai exchange's board for technology companies, with a registration-based "
            "listing system and a wider daily price limit than the main board."
        ),
    ),
    GlossaryEntry(
        term="北交所",
        aliases=("北京证券交易所", "Beijing Stock Exchange"),
        zh="北交所是服务创新型中小企业的北京证券交易所，由新三板精选层改制而来，交易规则与沪深主板不同。",
        en=(
            "The Beijing Stock Exchange serves innovative small and medium-sized companies; it grew out of the "
            "NEEQ select tier and its trading rules differ from the Shanghai and Shenzhen main boards."
        ),
    ),
    GlossaryEntry(
        term="沪深港通",
        aliases=("沪港通", "深港通", "Stock Connect"),
        zh="沪深港通是内地与香港市场的互联互通机制，包含沪股通、深股通（北向）和港股通（南向）。",
        en=(
            "Stock Connect links the mainland and Hong Kong markets: the Shanghai and Shenzhen Connect legs "
            "(northbound) and the Hong Kong Connect leg (southbound)."
        ),
    ),
    GlossaryEntry(
        term="大宗交易",
        aliases=("block trade",),
        zh="大宗交易是达到规模标准的证券在盘后按约定价格成交的方式，成交价常与收盘价有折价或溢价，不计入当日连续竞价。",
        en=(
            "A block trade is an after-hours negotiated transaction above a size threshold, often at a discount or "
            "premium to the close, and it is not part of the day's continuous auction."
        ),
    ),
    GlossaryEntry(
        term="注册制",
        aliases=("registration-based IPO",),
        zh="注册制是发行上市审核方式：交易所审核、证监会注册，信息披露责任落在发行人和中介机构，与核准制相对。",
        en=(
            "Under the registration-based system the exchange reviews a listing and the CSRC registers it, with "
            "disclosure responsibility on the issuer and its intermediaries, as opposed to the older approval "
            "system."
        ),
    ),
    # (round 15, K12) the labels of the figures the fundamentals tiles show: 扣非 next to 归母 net profit, ROE（加权）
    GlossaryEntry(
        term="扣非净利润",
        aliases=(
            "扣非",
            "扣非后净利润",
            "扣非归母净利润",
            "扣除非经常性损益后的净利润",
            "扣除非经常性损益后净利润",
            "recurring net profit",
            "net profit after non-recurring items",
            "net profit excluding non-recurring items",
        ),
        zh=(
            "扣非净利润是扣除非经常性损益后、归属于母公司股东的净利润。非经常性损益指与主营业务无直接关系或偶发的收支，"
            "例如资产处置收益、政府补助、金融资产公允价值变动等。归母净利润包含这些项目，扣非净利润把它们剔除，"
            "因此更能反映主营业务的持续盈利能力：扣非明显低于归母，说明利润里一次性收益较多；扣非高于归母，说明当期有"
            "一次性损失。FinSight 财务数据中的“扣非”即这一口径，可与同一公司的归母净利润对照。"
        ),
        en=(
            "Net profit after non-recurring items (扣非净利润, recurring net profit) is the net profit attributable to "
            "shareholders of the parent with non-recurring gains and losses removed: items unrelated to the core "
            "business or one-off in nature, such as gains on asset disposals, government subsidies or fair-value "
            "changes. Attributable net profit (归母净利润) includes them, so a recurring figure well below the "
            "attributable one means much of the profit was one-off, and a recurring figure above it means a one-off "
            "loss. FinSight's 扣非 figure is this measure and can be read next to the same company's attributable "
            "net profit."
        ),
        has_data_series=True,
        definition_only=True,
    ),
    GlossaryEntry(
        term="归母净利润",
        aliases=(
            "归母",
            "归属于母公司股东的净利润",
            "归属于上市公司股东的净利润",
            "归属母公司净利润",
            "net profit attributable to shareholders",
            "attributable net profit",
            "net income attributable to shareholders",
            # (round 16, L11) the other net-profit tile: 净利润（含少数股东损益）
            "净利润（含少数股东损益）",
            "净利润(含少数股东损益)",
            "含少数股东损益",
            "少数股东损益",
            "net profit incl. minority interests",
            "net profit including minority interests",
            "minority interests",
        ),
        zh=(
            "归母净利润即归属于母公司股东（上市公司普通股股东）的净利润，是合并报表净利润减去少数股东损益后的部分，"
            "也是计算每股收益和市盈率常用的利润口径。它包含非经常性损益；剔除非经常性损益后的口径是扣非净利润，"
            "两者之差就是当期非经常性损益的影响。FinSight 数据中每个净利润字段的口径（归母或含少数股东）"
            "在数据定义中标明。"
        ),
        en=(
            "Attributable net profit (归母净利润) is the net profit attributable to shareholders of the parent, i.e. "
            "consolidated net profit minus the share of minority interests; it is the profit usually behind "
            "earnings per share and the P/E. It includes non-recurring items; removing them gives the recurring "
            "figure (扣非净利润), and the gap between the two is the effect of the period's non-recurring items. "
            "FinSight states for each net-profit field whether it is attributable or total (including minorities)."
        ),
        has_data_series=True,
        definition_only=True,
    ),
    GlossaryEntry(
        term="加权平均ROE",
        aliases=(
            "加权ROE",
            "ROE（加权）",
            "ROE(加权)",
            "加权平均净资产收益率",
            "加权净资产收益率",
            "摊薄ROE",
            "全面摊薄ROE",
            "摊薄净资产收益率",
            "全面摊薄净资产收益率",
            "期末ROE",
            "ROE（期末净资产）",
            "ROE(期末净资产)",
            "ROE（期末）",
            "weighted average roe",
            "weighted roe",
            "weighted average return on equity",
            "diluted roe",
            "period-end roe",
        ),
        zh=(
            "加权平均净资产收益率（加权ROE）是归母净利润除以报告期内加权平均的归母净资产：分母以期初净资产为基础，"
            "期内增发、分红等引起的增减按存续时间加权，是年报按披露规则列示的主要口径。全面摊薄（期末）ROE则用期末"
            "归母净资产作分母；期内净资产大幅增加（如增发）时，摊薄ROE通常低于加权ROE。两种口径的数值不能直接比较，"
            "FinSight 的ROE字段标注所用口径（加权或期末/摊薄）。"
        ),
        en=(
            "Weighted average ROE is attributable net profit divided by the weighted average attributable equity "
            "over the period: opening equity, with changes such as share issues or dividends weighted by the time "
            "they were outstanding. It is the main ROE in annual reports. Fully diluted (period-end) ROE divides by "
            "closing equity instead, so after a large equity increase it is usually lower than the weighted figure. "
            "The two are not directly comparable; FinSight labels which basis each ROE figure uses (weighted or "
            "period-end / diluted)."
        ),
        has_data_series=True,
        definition_only=True,
    ),
    # (round 16, review 12 L11) the other figures the price, fundamentals and indicator tiles show
    GlossaryEntry(
        term="市盈率",
        aliases=(
            "市盈率TTM",
            "市盈率 TTM",
            "市盈率(TTM)",
            "市盈率（TTM）",
            "滚动市盈率",
            "动态市盈率",
            "静态市盈率",
            "行业市盈率",
            "PE",
            "P/E",
            "PE TTM",
            "PE(TTM)",
            "P/E (TTM)",
            "pe ratio",
            "p/e ratio",
            "price-to-earnings",
            "price to earnings",
            "price-earnings ratio",
            "industry p/e",
        ),
        zh=(
            "市盈率（PE）是股价除以每股收益，等价于总市值除以归母净利润，可理解为按当前利润水平需要多少年的利润才能抵得"
            "上现在的市值。TTM（滚动）市盈率用最近十二个月（四个季度合计）的归母净利润计算；静态市盈率用上一完整财年的"
            "利润，动态市盈率用年化或预测的利润，三种口径的数值不能直接比较。FinSight 展示的是市盈率 TTM，按估值日的收"
            "盘价计算（估值日在数据中标明）；行业市盈率是所属行业的汇总值。市盈率的高低要结合行业、盈利增长和利润质量来"
            "看，亏损公司的市盈率为负或没有意义。"
        ),
        en=(
            "The price-to-earnings ratio (P/E) is the share price divided by earnings per share, or equivalently the "
            "market value divided by attributable net profit: how many years of current profit the market value "
            "represents. A TTM (trailing) P/E uses the last twelve months of profit (the latest four quarters); a "
            "static P/E uses the last full fiscal year and a forward or dynamic P/E an annualised or forecast profit, "
            "so the three are not directly comparable. FinSight shows the TTM P/E at the close of the valuation date "
            "(stated with the figure); the industry P/E is an aggregate of the company's industry. A P/E is read "
            "against the industry, growth and earnings quality; a loss-making company's P/E is negative or meaningless."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="市净率",
        aliases=(
            "破净",
            "行业市净率",
            "PB",
            "P/B",
            "pb ratio",
            "p/b ratio",
            "price-to-book",
            "price to book",
            "price-to-book ratio",
            "industry p/b",
            "below book value",
        ),
        zh=(
            "市净率（PB）是股价除以每股净资产，等价于总市值除以归母净资产（账面价值），表示市场为每单位净资产支付的价格"
            "。股价跌到每股净资产以下称为“破净”，即市值低于账面净资产。银行、保险、券商等资产以金融资产为主的行业常用市"
            "净率估值，盈利波动大的周期行业也常参考它。FinSight 的市净率按估值日的收盘价和最近一期报告的净资产计算；行"
            "业市净率是所属行业的汇总值。"
        ),
        en=(
            "The price-to-book ratio (P/B) is the share price divided by book value (net assets) per share, or the "
            "market value divided by equity attributable to shareholders: what the market pays for each unit of net "
            "assets. Shares priced below book value per share (a P/B under one) are said to trade below book (破净). "
            "Banks, insurers and brokers, whose assets are mostly financial, are usually valued on P/B, and so are "
            "cyclicals with volatile earnings. FinSight's P/B is at the close of the valuation date against the latest "
            "reported equity; the industry P/B is an aggregate of the company's industry."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="ROE",
        aliases=("净资产收益率", "return on equity"),
        zh=(
            "净资产收益率（ROE）是归母净利润除以归母净资产，衡量公司用股东投入的资本赚钱的效率。年报主要列示加权平均 "
            "ROE（分母为期内加权平均净资产）；全面摊薄 ROE 用期末净资产作分母。FinSight 的 ROE 标注口径：“ROE（加权）”"
            "是加权平均口径，“ROE（期末净资产）”是期末口径，两者数值不能直接比较。高 ROE 也可能来自高负债，需要结合杠杆"
            "一起看。"
        ),
        en=(
            "Return on equity (ROE) is attributable net profit divided by attributable equity: how much profit the "
            "company earns on the capital shareholders have put in. Annual reports mainly show the weighted average "
            "ROE (the period's weighted average equity as the denominator); the fully diluted ROE uses period-end "
            "equity. FinSight labels the basis: 'ROE (weighted average)' or 'ROE (period-end equity)', and the two are "
            "not directly comparable. A high ROE can also come from high leverage, so read it with the debt level."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="毛利率",
        aliases=("销售毛利率", "gross margin", "gross profit margin"),
        zh=(
            "毛利率是毛利润占营业收入的比例：（营业收入 − 营业成本）÷ 营业收入。它反映产品或服务本身的盈利空间，还没有"
            "扣除销售、管理、研发、财务等期间费用和税费。不同行业的毛利率差异很大，宜与同行或自身历史比较；银行、券商等"
            "金融企业通常不披露毛利率。"
        ),
        en=(
            "Gross margin is gross profit as a share of operating revenue: (revenue - cost of revenue) / revenue. It "
            "shows the margin on the products or services themselves, before selling, administrative, R&D and "
            "financial expenses and taxes. Gross margins differ widely across industries, so compare them with peers "
            "or the company's own history; banks and brokers usually report none."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="净利率",
        aliases=("销售净利率", "净利润率", "net margin", "net profit margin"),
        zh=(
            "净利率（销售净利率）是净利润占营业收入的比例，表示每单位收入最终留下多少利润，已经扣除成本、各项费用、税费"
            "和非经常性损益的影响。报告没有直接给出时，FinSight 用归母净利润 ÷ 收入计算，并在答案中写明所用的利润和收入"
            "口径。"
        ),
        en=(
            "Net margin is net profit as a share of operating revenue: how much of each yuan of revenue is left as "
            "profit after costs, expenses, taxes and non-recurring items. When the report gives none, FinSight "
            "computes it as attributable net profit / revenue and states the profit and revenue bases it used."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="营业总收入",
        aliases=(
            "营业收入",
            "营收",
            "total operating revenue",
            "total revenue",
            "operating revenue",
            "revenue",
        ),
        zh=(
            "营业收入是企业经营主营业务和其他业务取得的收入。营业总收入在营业收入之外，还包括金融类业务的利息收入、已赚"
            "保费、手续费及佣金收入等，所以多数非金融企业两者相同，集团内有财务公司等金融业务的企业营业总收入会大于营业"
            "收入。FinSight 的收入字段在数据定义中标明是“营业收入”还是“营业总收入”，比较两家公司的收入时应使用同一口径"
            "。"
        ),
        en=(
            "Operating revenue (营业收入) is the income from the company's main and other businesses. Total operating "
            "revenue (营业总收入) adds the income of financial operations, such as interest income, earned premiums "
            "and fee and commission income, so for most non-financial companies the two are equal, while a group with "
            "a finance arm reports a larger total. FinSight states for each revenue figure which of the two it is; "
            "compare companies on the same basis."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="每股收益",
        aliases=(
            "EPS",
            "基本每股收益",
            "稀释每股收益",
            "earnings per share",
            "basic eps",
            "diluted eps",
            "basic earnings per share",
            "diluted earnings per share",
        ),
        zh=(
            "每股收益（EPS）是归母净利润除以发行在外普通股的加权平均股数，表示每股普通股分到的利润。基本每股收益按实际"
            "发行在外的普通股计算；稀释每股收益假设可转换债券、股票期权、限制性股票等潜在普通股都已转为普通股，分母更大"
            "，所以通常不高于基本每股收益。FinSight 在数据定义中标明每股收益是基本还是稀释口径。"
        ),
        en=(
            "Earnings per share (EPS) is attributable net profit divided by the weighted average number of ordinary "
            "shares outstanding. Basic EPS uses the shares actually outstanding; diluted EPS assumes that potential "
            "shares (convertible bonds, options, restricted shares) have been converted, so its denominator is larger "
            "and it is usually no higher than basic EPS. FinSight states whether an EPS figure is basic or diluted."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="总市值",
        aliases=(
            "市值",
            "流通市值",
            "market cap",
            "market capitalisation",
            "market capitalization",
            "market value",
            "float market cap",
            "free-float market cap",
            "circulating market value",
        ),
        zh=(
            "总市值是公司全部股本乘以股价，是市场对整家公司股权的定价；流通市值只计算可以在二级市场自由交易的流通股（不"
            "含限售股）乘以股价，所以通常不大于总市值。两者都随股价每天变化。FinSight 的总市值和流通市值按估值日的收盘"
            "价计算，估值日在数据中标明。"
        ),
        en=(
            "Total market value (market cap) is the company's total share count times the share price: the market's "
            "price for the whole company's equity. The free-float market value counts only the shares that trade "
            "freely (restricted shares excluded), so it is usually no larger than the total. Both move with the price "
            "every day; FinSight's figures are at the close of the valuation date, stated with them."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="TTM",
        aliases=("滚动十二个月", "近十二个月", "trailing twelve months", "LTM", "last twelve months"),
        zh=(
            "TTM（Trailing Twelve Months）指最近十二个月，即截至最新报告期的四个季度合计。用 TTM 口径计算的指标（如市盈"
            "率 TTM）随每期定期报告滚动更新，比只用上一年年报更及时，也避免了只用单个季度带来的季节性偏差。"
        ),
        en=(
            "TTM (trailing twelve months) is the latest four reported quarters added together. Ratios on a TTM basis, "
            "such as the TTM P/E, roll forward with each periodic report, so they are more current than the last "
            "annual report and avoid the seasonality of a single quarter."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="收盘价",
        aliases=("closing price", "close price"),
        zh=(
            "收盘价是一个交易日最后的成交价格。A股的收盘价由收盘集合竞价产生，是计算当日涨跌幅、市值和多数技术指标的基"
            "准。FinSight 展示的收盘价注明交易日期，是最新可用交易日的价格，不是实时报价。"
        ),
        en=(
            "The closing price is the last traded price of a session. On the A-share market it is set by the closing "
            "call auction and is the basis for the day's change, the market value and most technical indicators. "
            "FinSight's close is dated: it is the latest available session's price, not a live quote."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="涨跌幅",
        aliases=("日涨跌幅", "day change", "daily change", "percentage change"),
        zh=(
            "涨跌幅是收盘价相对前一交易日收盘价的变动幅度：（当日收盘价 − 前收盘价）÷ 前收盘价，以百分比表示。A股有单日"
            "涨跌幅限制，幅度按板块和股票类型不同（主板最窄，创业板和科创板更宽），新股上市初期另有规则。"
        ),
        en=(
            "The day change is the close's move from the previous session's close: (close - previous close) / previous "
            "close, in percent. A-shares have a daily price limit whose width depends on the board and the type of "
            "share (narrowest on the main board, wider on ChiNext and the STAR Market), with separate rules for new "
            "listings."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="成交额",
        aliases=("成交金额", "turnover value", "traded value", "trading value"),
        zh=(
            "成交额是一段时间内（通常是一个交易日）成交的总金额，即每笔成交的价格乘以数量之和，用来衡量交易的活跃程度和"
            "资金规模；成交量则是成交的股数或手数。成交额与流通市值之比近似于按金额计算的换手率。"
        ),
        en=(
            "Turnover value is the total amount traded in a period, usually one session: the sum of price times "
            "quantity over all trades. It measures how actively and with how much money a stock is traded; volume is "
            "the number of shares (or lots) traded. Turnover value over the free-float market value is roughly the "
            "turnover rate in money terms."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="估值日",
        aliases=("valuation date",),
        zh=(
            "估值日是计算市盈率、市净率和市值时所用收盘价的交易日期。财务数据来自最近一期报告（报告期），估值则按估值日"
            "的股价计算，两者日期不同；FinSight 在基本面数据旁同时标明报告期和估值日。"
        ),
        en=(
            "The valuation date is the session whose close is used for the P/E, P/B and market value. The financial "
            "figures come from the latest report (the reporting period) while the valuation uses the price on the "
            "valuation date, so the two dates differ; FinSight states both next to the fundamentals."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="均线",
        aliases=("移动平均线", "MA5", "MA10", "MA20", "MA60", "moving average"),
        zh=(
            "均线（移动平均线，MA）是最近若干个交易日收盘价的算术平均，FinSight 展示五日均线和二十日均线，每天随新的收"
            "盘价滚动更新。股价在均线上方常被视为短期偏强，短期均线上穿长期均线常被称为“金叉”；均线是对过去价格的描述，"
            "本身不预测未来走势。"
        ),
        en=(
            "A moving average (MA) is the mean close of the last several sessions; FinSight shows the five-day and the "
            "twenty-day averages, which roll forward with each new close. A price above its average is often read as "
            "short-term strength, and a short average crossing above a long one as a 'golden cross'; an average "
            "describes past prices and does not predict the next move."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="RSI",
        aliases=("RSI(14)", "RSI（14）", "相对强弱指标", "相对强弱指数", "relative strength index"),
        zh=(
            "RSI（相对强弱指标）比较一段时间内上涨幅度与下跌幅度的大小，取值在零到一百之间，FinSight 用最近十四个交易日"
            "计算。数值很高常被视为超买、很低常被视为超卖，但强势行情中 RSI 可能长期偏高；它描述的是过去的涨跌力度，不"
            "是买卖信号。"
        ),
        en=(
            "The relative strength index (RSI) compares the size of up moves with down moves over a period, on a scale "
            "from zero to one hundred; FinSight uses the last fourteen sessions. A very high reading is often called "
            "overbought and a very low one oversold, but in a strong trend the RSI can stay high for long; it "
            "describes past momentum and is not a trading signal."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
    GlossaryEntry(
        term="波动率",
        aliases=("20日波动率", "历史波动率", "volatility", "20-day volatility"),
        zh=(
            "波动率衡量价格变动的剧烈程度。FinSight 的二十日波动率是最近二十个交易日日收益率的标准差，按一年的交易日数"
            "年化，以百分比表示。波动率越高，价格短期大幅上下波动的可能越大；它描述风险的大小，不说明涨跌方向。"
        ),
        en=(
            "Volatility measures how much a price moves. FinSight's twenty-day volatility is the standard deviation of "
            "the last twenty daily returns, annualised over a year of trading days and shown in percent. Higher "
            "volatility means larger swings in either direction; it describes the size of the risk, not the direction."
        ),
        has_data_series=True,
        definition_only=True,
        adjacent_cue=True,
    ),
)

_BY_FORM: dict[str, GlossaryEntry] = {}
for _entry in GLOSSARY:
    for _form in _entry.surface_forms:
        _BY_FORM.setdefault(_form.lower(), _entry)
# Longest first: "融资融券" before "两融", "沪深港通" before "沪港通".
_FORMS = sorted(_BY_FORM, key=len, reverse=True)
_LATIN_FORM = re.compile(r"^[a-z0-9 .-]+$")
# A term with an everyday meaning counts only in a question that asks what it means, or that is about the market.
_CUE = re.compile(
    r"是什么|是啥|什么意思|啥意思|什么叫|指的是|是指|含义|定义|概念|怎么理解|如何理解|怎么看|"
    r"股|基金|etf|指数|资金|市场|行情|大盘|估值|仓位|净流入|净买入|买入|卖出|余额|机构|"
    r"\bwhat (?:is|are|does)\b|\bmean\b|\bdefinition\b|\bstocks?\b|\bshares?\b|\bmarket\b|\bfunds?\b|\bequities\b",
    re.IGNORECASE,
)


def lookup_concept(text: str) -> GlossaryEntry | None:
    """The glossary entry a question is about, or ``None``.

    A CJK form must appear in the question; a Latin form must appear as whole words ("national team" but not
    inside a longer word), so ordinary sentences do not match by accident. Terms that also have an everyday
    meaning (``needs_cue``) additionally require a definition question or market wording, so "国家队队员名单"
    (a sports question) matches nothing while "国家队是指什么" does.
    """
    lowered = (text or "").lower()
    if not lowered:
        return None
    for form in _FORMS:
        entry = _BY_FORM[form]
        if _LATIN_FORM.match(form):
            if not re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", lowered):
                continue
        elif form not in lowered:
            continue
        if entry.needs_cue and not _CUE.search(lowered.replace(form, " ")):
            continue
        if entry.definition_only and not _DEFINITION_CUE.search(lowered):
            continue
        if entry.adjacent_cue and not _cue_next_to(lowered, form):
            continue
        return entry
    return None


def definition_question(text: str) -> GlossaryEntry | None:
    """(round 16, L11) The glossary entry a question asks the definition of ("归母净利润是什么意思", "What does P/E
    mean?"), or ``None``. Inside a session such a question is about the word, not about the previous turn's company:
    the ellipsis carry must not turn it into that company's data."""
    entry = lookup_concept(text)
    return entry if entry is not None and _DEFINITION_CUE.search((text or "").lower()) else None


def lookup_concepts(text: str, limit: int = 3) -> list[GlossaryEntry]:
    """(round 16, L11) Every glossary entry a definition question names, in the order of the question ("市盈率和市净率有
    什么区别": both), at most ``limit``; the first is ``lookup_concept``'s."""
    first = lookup_concept(text)
    if first is None:
        return []
    lowered = (text or "").lower()
    found: dict[str, tuple[int, GlossaryEntry]] = {first.term: (-1, first)}
    taken: list[tuple[int, int]] = []
    for form in _FORMS:
        entry = _BY_FORM[form]
        if _LATIN_FORM.match(form):
            match = re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", lowered)
            start = match.start() if match else -1
        else:
            start = lowered.find(form)
        if start < 0 or any(a <= start < b for a, b in taken):
            continue
        taken.append((start, start + len(form)))
        if entry.term in found or entry.needs_cue or not entry.definition_only:
            continue
        found[entry.term] = (start, entry)
    ordered = sorted(found.values(), key=lambda item: item[0])
    return [entry for _start, entry in ordered][:limit]


# (round 16, L11) the definition cue right before or after the word: "什么是市盈率", "市值是什么意思",
# "PE和PB有什么区别", "What does P/B mean?", "How is net margin calculated?"
_CUE_BEFORE = re.compile(
    r"(?:什么是|什么叫做?|何为|何谓|解释一下|解释|说说|讲讲|介绍一下|科普一下|"
    r"\bwhat(?:'s|\s+is|\s+are|\s+does|\s+do)\s+(?:a\s+|an\s+|the\s+)?|\bdefine\s+(?:the\s+)?|"
    r"\bexplain\s+(?:the\s+)?|\bmeaning of\s+(?:the\s+)?|\b(?:difference|differences)\s+between\s+(?:the\s+|a\s+)?|"
    r"\bcompare\s+(?:the\s+)?|\bhow (?:is|are)\s+(?:the\s+)?|"
    r"\bhow do (?:you|i|we) (?:calculate|read|interpret)\s+(?:the\s+|a\s+)?)\s*$",
    re.IGNORECASE,
)
_CUE_AFTER = re.compile(
    r"^\s*(?:ratio\s*)?(?:到底|究竟|应该|应当|又该|该|要)?\s*(?:是什么|是啥|什么意思|啥意思|指什么|指的是|是指|的含义|的定义|的意思|的概念|含义|"
    r"定义|怎么算|如何计算|怎么计算|的计算公式|计算公式|怎么理解|如何理解|是怎么算的|是如何计算的|"
    r"(?:有|又有)?(?:什么|啥)(?:区别|差别|不同)|"
    r"(?:和|与|跟|vs\.?|对比)\s*.{1,16}?(?:有什么|有啥|的|之间的)?\s*(?:区别|差别|不同|差异|差在哪|有何不同)|"
    r"\bmeans?\b|\bstands? for\b|\b(?:is|are)\s+(?:calculated|defined|computed|measured)\b|"
    r"(?:\s+and\s+|\s*/\s*|\s+vs\.?\s+).{1,30}?\b(?:differ|difference)\b)",
    re.IGNORECASE,
)


def _cue_next_to(lowered: str, form: str) -> bool:
    """A definition cue right before or after an occurrence of ``form``."""
    pattern = rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])" if _LATIN_FORM.match(form) else re.escape(form)
    for match in re.finditer(pattern, lowered):
        if _CUE_BEFORE.search(lowered, 0, match.start()) or _CUE_AFTER.match(lowered[match.end() :]):
            return True
    return False


# (round 15, K12) a question about what a figure's label means, or how two labels differ
_DEFINITION_CUE = re.compile(
    r"是什么|什么是|是啥|什么意思|啥意思|什么叫|何为|何谓|指的是|指什么|是指|含义|定义|概念|区别|差别|差异|不同|差在哪|"
    r"怎么理解|如何理解|怎么算|如何计算|计算公式|口径|"
    r"\bwhat (?:is|are|does)\b|\bmean(?:s|ing)?\b|\bdifference\b|\bdiffer\b|\bdefin(?:e|ed|ition)\b|\bexplain\b|"
    r"\bhow is\b.{0,40}\bcalculated\b",
    re.IGNORECASE,
)


def entry_for_term(term: str) -> GlossaryEntry | None:
    """The entry whose headword or alias is exactly ``term`` (the ``explain_concept`` tool's argument): a term the
    planner or the model already chose needs no definition cue."""
    return _BY_FORM.get((term or "").strip().lower())


def concept_terms() -> tuple[str, ...]:
    return tuple(entry.term for entry in GLOSSARY)
