import { render, screen } from "@testing-library/react";

import { I18nContext, makeTranslate, type Lang } from "@/lib/i18n";
import type { Kpi, MarketData } from "@/lib/marketData";

import { DataPanel } from "./DataPanel";

const tile = (extra: Partial<Kpi>): Kpi => ({
  key: `fundamental_600036.SH:${extra.label ?? "kpi.pe"}`,
  label: "kpi.pe",
  value: 6.86,
  format: "ratio",
  subject: "招商银行",
  evidenceId: "fundamental_600036.SH",
  ...extra,
});

function renderPanel(data: MarketData, lang: Lang = "zh", props: Partial<Parameters<typeof DataPanel>[0]> = {}) {
  return render(
    <I18nContext.Provider value={{ lang, t: makeTranslate(lang) }}>
      <DataPanel data={data} themeKey="dark" {...props} />
    </I18nContext.Provider>,
  );
}

describe("DataPanel KPI tiles (round 13, I15)", () => {
  it("shows an English label in full: it wraps instead of being cut, and the title repeats it", () => {
    const { container } = renderPanel(
      { series: [], kpis: [tile({ label: "kpi.netProfit", value: 150.18e9, format: "money" })] },
      "en",
      { displayName: (name) => (name === "招商银行" ? "China Merchants Bank" : name) },
    );
    const label = container.querySelector(".kpi-label")!;
    expect(label.textContent).toBe("China Merchants Bank · Net profit");
    expect(label.getAttribute("title")).toBe("China Merchants Bank · Net profit");
    // no single-line ellipsis
    expect(label.className).not.toMatch(/\btruncate\b/);
    expect(label.className).toContain("[overflow-wrap:anywhere]");
  });

  it("names the definition of a figure when the data states it (round 13)", () => {
    const { container } = renderPanel({
      series: [],
      kpis: [
        tile({ label: "kpi.roe", basisLabel: "kpi.roeWeighted", value: 6.89, format: "percentLevel" }),
        tile({ label: "kpi.revenue", basisLabel: "kpi.revenueTotal", value: 458.5e9, format: "money" }),
        tile({ label: "kpi.netProfit", value: 1.5e9, format: "money" }),
      ],
    });
    const labels = [...container.querySelectorAll(".kpi-label")].map((node) => node.textContent);
    expect(labels).toEqual(["招商银行 · ROE（加权）", "招商银行 · 营业总收入", "招商银行 · 净利润"]);
  });

  it("dates P/E and P/B at their valuation date and says so", () => {
    renderPanel({
      series: [],
      kpis: [
        tile({ asOf: "2026-09-30", valuation: true }),
        tile({ label: "kpi.roe", value: 8.68, format: "percentLevel", asOf: "2025-12-31" }),
      ],
    });
    const [pe, roe] = screen.getAllByText(/招商银行 · /).map((node) => node.parentElement!);
    expect(pe!.textContent).toContain("2026/09/30");
    expect(pe!.textContent).toContain("估值日");
    expect(roe!.textContent).toContain("8.68%");
    expect(roe!.textContent).toContain("2025/12/31");
    expect(roe!.textContent).not.toContain("估值日");
  });

  it("marks no tile as the asked metric when nothing was asked (a forecast question)", () => {
    const { container } = renderPanel(
      { series: [], kpis: [tile({ label: "kpi.close", value: 291.11, format: "price", asOf: "2026-09-30" })] },
      "zh",
      { asked: [] },
    );
    expect(container.querySelector(".kpi-featured")).toBeNull();
    expect(screen.queryByText(/所问指标/)).toBeNull();
  });

  it("marks the asked metric when one was asked", () => {
    const { container } = renderPanel(
      { series: [], kpis: [tile({ label: "kpi.close", value: 291.11, format: "price", asOf: "2026-09-30" })] },
      "zh",
      { asked: ["close"] },
    );
    expect(container.querySelector(".kpi-featured")).not.toBeNull();
    expect(screen.getByText(/所问指标/)).toBeInTheDocument();
  });
});
