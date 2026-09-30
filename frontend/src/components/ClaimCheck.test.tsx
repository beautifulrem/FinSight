import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { I18nContext, makeTranslate, type Lang } from "@/lib/i18n";
import type { ClaimReport } from "@/lib/types";

import { ClaimCheckView, ClaimReportCard } from "./ClaimCheck";
import { TooltipProvider } from "./ui/tooltip";

const REPORT: ClaimReport = {
  claim: "茅台市盈率只有15倍，股价昨天跌了5%",
  verdict: "partially_supported",
  checks: [
    {
      target: "贵州茅台",
      metric: "pe_ttm",
      claimed: 15,
      actual: 24.6,
      status: "contradicted",
      evidence_id: "fundamental_600519.SH",
      source: "tushare",
      as_of: "2025-12-31",
      note: "",
    },
    {
      target: "贵州茅台",
      metric: "close",
      claimed: 1409.5,
      actual: 1409.5,
      status: "supported",
      evidence_id: "price_600519.SH",
      source: null,
      as_of: "2026-04-22",
      note: "",
    },
    { target: "贵州茅台", metric: null, claimed: 5, actual: null, status: "unverifiable", note: "metric not recognised" },
  ],
  targets: [{ name: "贵州茅台", symbol: "600519.SH" }],
  evidence_sources: [
    { evidence_id: "fundamental_600519.SH", source_name: "tushare", as_of: "2025-12-31", title: "贵州茅台 fundamentals" },
    { evidence_id: "price_600519.SH", source_name: null, as_of: "2026-04-22", title: "贵州茅台 daily market data" },
  ],
  disclaimer: "核查只比对声明中的数字与所列数据源，不评价观点本身，也不构成投资建议。",
};

function wrap(ui: React.ReactNode, lang: Lang = "zh") {
  return render(
    <I18nContext.Provider value={{ lang, t: makeTranslate(lang) }}>
      <TooltipProvider>{ui}</TooltipProvider>
    </I18nContext.Provider>,
  );
}

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const RAW = ["pe_ttm", "partially_supported", "contradicted", "unverifiable", "metric not recognised"];

describe("ClaimReportCard", () => {
  it("shows the verdict, claimed vs actual values, sources and the disclaimer without raw codes", () => {
    wrap(<ClaimReportCard report={REPORT} />);
    const card = screen.getByRole("article", { name: "核查结果" });
    expect(within(card).getByText("部分相符")).toBeInTheDocument();
    expect(card).toHaveTextContent("共 3 项 · 1 项相符 · 1 项不符 · 1 项无法核实");

    const checks = within(card).getAllByRole("listitem");
    expect(checks).toHaveLength(3);
    const [pe, close, unknown] = checks as [HTMLElement, HTMLElement, HTMLElement];
    expect(within(pe).getByRole("heading", { name: "市盈率(TTM)" })).toBeInTheDocument();
    expect(within(pe).getByText("不符")).toBeInTheDocument();
    expect(pe.querySelector(".claim-claimed")).toHaveTextContent("15 倍");
    expect(pe.querySelector(".claim-actual")).toHaveTextContent("24.6 倍");
    expect(pe).toHaveTextContent("tushare");
    expect(pe.querySelector("time")).toHaveAttribute("dateTime", "2025-12-31");
    expect(pe.querySelector(".claim-source")).toHaveTextContent("可能过时");

    // No source name on the price record: the source type is shown instead.
    expect(close).toHaveTextContent("收盘价");
    expect(close).toHaveTextContent("行情");
    expect(close.querySelector(".claim-actual")).toHaveTextContent("1,409.5 元");

    expect(unknown).toHaveTextContent("未识别的指标");
    expect(unknown).toHaveTextContent("无数据");
    expect(unknown).toHaveTextContent("未能判断这个数字指的是哪个指标");

    expect(card).toHaveTextContent("不构成投资建议");
    for (const code of RAW) expect(card).not.toHaveTextContent(code);
  });

  it("renders English labels and an empty result", () => {
    wrap(<ClaimReportCard report={{ ...REPORT, verdict: "unverifiable", checks: [], targets: [] }} />, "en");
    expect(screen.getByText("Unverifiable")).toBeInTheDocument();
    expect(screen.getByText("No checkable numbers found")).toBeInTheDocument();
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
  });
});

describe("ClaimReportCard parts that were not checked", () => {
  const RELATION_AND_OPINION: ClaimReport = {
    claim: "茅台的市盈率比五粮液高，中国平安市盈率8.7倍，ROE很高",
    verdict: "supported",
    checks: [
      {
        target: "贵州茅台",
        metric: "pe_ttm",
        claimed: null,
        comparator: "gt",
        reference: "五粮液",
        reference_value: 20.9,
        actual: 24.6,
        status: "supported",
      },
      { target: "中国平安", metric: "pe_ttm", claimed: 8.7, actual: 8.7, status: "supported" },
    ],
    unchecked: [{ text: "ROE很高", reason: "no_claim" }],
    targets: [],
    disclaimer: "",
  };

  it("lists every part of the claim: the relation, the number and a not-checked row, and counts all three", () => {
    wrap(<ClaimReportCard report={RELATION_AND_OPINION} />);
    const card = screen.getByRole("article", { name: "核查结果" });
    expect(card).toHaveTextContent("共 3 项 · 2 项相符 · 1 项未核查");
    const rows = within(card).getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    expect(rows[0]!.querySelector(".claim-claimed")).toHaveTextContent("> 五粮液");
    const unchecked = rows[2]!;
    expect(unchecked).toHaveAttribute("data-status", "unchecked");
    expect(unchecked).toHaveTextContent("未核查");
    expect(unchecked).toHaveTextContent("“ROE很高”");
    expect(unchecked).toHaveTextContent("没有可以比对的数字");
  });

  it("shows not-checked rows in English, and them alone when nothing was checked", () => {
    wrap(<ClaimReportCard report={{ ...RELATION_AND_OPINION, verdict: "unverifiable", checks: [] }} />, "en");
    const card = screen.getByRole("article", { name: "Result" });
    expect(card).toHaveTextContent("Parts: 1 · 1 not checked");
    expect(within(card).getByText("Not checked")).toBeInTheDocument();
    expect(within(card).queryByText("No checkable numbers found")).not.toBeInTheDocument();
  });
});

describe("ClaimReportCard comparators", () => {
  it("shows the comparator of each claim and the date basis", () => {
    const report: ClaimReport = {
      ...REPORT,
      claim: "贵州茅台ROE超过30%，市盈率不是15倍",
      verdict: "supported",
      checks: [
        { ...REPORT.checks[0]!, metric: "roe", claimed: 30, claimed_unit: "%", comparator: "gt", actual: 33, status: "supported", as_of_basis: "report_date" },
        { ...REPORT.checks[0]!, claimed: 15, comparator: "ne", negated: true, status: "supported", as_of: "2026-09-24", as_of_basis: "valuation_date" },
      ],
    };
    wrap(<ClaimReportCard report={report} />);
    const [roe, pe] = screen.getAllByRole("listitem") as [HTMLElement, HTMLElement];
    expect(roe.querySelector(".claim-claimed")).toHaveTextContent("> 30%");
    expect(roe.querySelector(".claim-claimed")).toHaveTextContent("高于 30%");
    expect(roe.querySelector(".claim-actual")).toHaveTextContent("33%");
    expect(roe).toHaveTextContent("(报告期)");
    expect(pe.querySelector(".claim-claimed")).toHaveTextContent("≠ 15 倍");
    expect(pe).toHaveTextContent("(估值日)");
  });
});

describe("ClaimCheckView", () => {
  afterEach(() => vi.restoreAllMocks());

  it("checks a typed claim on Enter and announces the verdict", async () => {
    const user = userEvent.setup();
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, REPORT));
    wrap(<ClaimCheckView apiKey="k1" />);
    await user.type(screen.getByLabelText("要核查的说法"), "茅台市盈率只有15倍{Enter}");
    expect(await screen.findByRole("article", { name: "核查结果" })).toBeInTheDocument();
    const [url, init] = spy.mock.calls[0]!;
    expect(url).toBe("/agent/claim-check");
    expect(JSON.parse(String(init?.body))).toEqual({ claim: "茅台市盈率只有15倍", language: "zh" });
    expect((init?.headers as Record<string, string>)["X-API-Key"]).toBe("k1");
    expect(screen.getByRole("status")).toHaveTextContent("核查完成：部分相符");
  });

  it("runs an example chip and blocks too-short claims", async () => {
    const user = userEvent.setup();
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, REPORT));
    wrap(<ClaimCheckView apiKey="" />);
    await user.type(screen.getByLabelText("要核查的说法"), "x");
    await user.click(screen.getByRole("button", { name: "核查" }));
    expect(screen.getByText("请至少输入 2 个字符")).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();

    const chips = screen.getAllByRole("button").filter((button) => button.classList.contains("claim-example"));
    expect(chips).toHaveLength(3);
    await user.click(chips[0]!);
    expect(screen.getByLabelText("要核查的说法")).toHaveValue("茅台市盈率只有15倍，股价昨天跌了5%");
    expect(await screen.findByText("部分相符")).toBeInTheDocument();
    expect(JSON.parse(String(spy.mock.calls[0]![1]?.body)).claim).toBe("茅台市盈率只有15倍，股价昨天跌了5%");
  });

  it("shows loading, then server and network errors with a working retry", async () => {
    const user = userEvent.setup();
    let resolve: (response: Response) => void = () => undefined;
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementationOnce(() => new Promise<Response>((done) => (resolve = done)))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(jsonResponse(200, REPORT));
    wrap(<ClaimCheckView apiKey="" />, "en");
    await user.type(screen.getByLabelText("Claim to check"), "Moutai P/E is 15{Enter}");
    expect(screen.getByRole("button", { name: "Checking" })).toBeDisabled();
    expect(document.querySelector(".claim-loading")).toHaveAttribute("aria-busy", "true");

    resolve(jsonResponse(500, { detail: "boom" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Request failed: boom");

    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Cannot reach the local server"));

    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Partly supported")).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(3);
  });

  it("explains auth and validation errors", async () => {
    const user = userEvent.setup();
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(401, { detail: "missing API key" }))
      .mockResolvedValueOnce(jsonResponse(422, { detail: [{ msg: "String should have at most 2000 characters" }] }));
    wrap(<ClaimCheckView apiKey="" />);
    await user.type(screen.getByLabelText("要核查的说法"), "茅台市盈率15倍{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("服务端要求 API Key");
    await user.click(screen.getByRole("button", { name: "重试" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("说法需要 2 到 2000 个字符"));
  });
});
