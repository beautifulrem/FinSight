import { evidenceFreshness, summarizeFreshness } from "./freshness";
import type { EvidenceSource } from "./types";

const NOW = new Date("2026-09-26T12:00:00");

const price = (id: string, mode = "live", asOf = "2026-09-24", extra: Record<string, unknown> = {}): EvidenceSource => ({
  evidence_id: id,
  kind: "structured",
  source_type: "market_api",
  as_of: asOf,
  payload: { provenance: { mode, is_live: mode !== "snapshot", as_of: asOf, freshness: "fresh", ...extra } },
});

const industrySnapshot: EvidenceSource = {
  evidence_id: "industry_白酒",
  kind: "structured",
  source_type: "industry_sql",
  as_of: "2026-04-21",
  payload: {
    provenance: { source: "offline_snapshot", mode: "snapshot", is_live: false, as_of: "2026-04-21", freshness: "stale" },
  },
};

describe("evidenceFreshness", () => {
  it("reads provenance modes and staleness", () => {
    expect(evidenceFreshness(price("p1"), NOW)).toMatchObject({ mode: "live", stale: false, days: 2 });
    expect(evidenceFreshness(price("p2", "live_fallback"), NOW).mode).toBe("fallback");
    expect(evidenceFreshness(price("p3", "last_known_good"), NOW).mode).toBe("cached");
    expect(evidenceFreshness(industrySnapshot, NOW)).toMatchObject({ mode: "snapshot", stale: true });
  });

  it("falls back to the age of documents without provenance", () => {
    const news: EvidenceSource = { evidence_id: "news_1", kind: "document", source_type: "news", as_of: "2026-06-01" };
    const faq: EvidenceSource = { evidence_id: "faq_1", kind: "document", source_type: "faq", as_of: "2020-01-01" };
    expect(evidenceFreshness(news, NOW)).toMatchObject({ mode: "unknown", stale: true });
    expect(evidenceFreshness(faq, NOW).stale).toBe(false);
  });
});

describe("summarizeFreshness", () => {
  it("warns when an April snapshot sits next to September prices", () => {
    const summary = summarizeFreshness([price("p1"), industrySnapshot], new Set(["p1"]), NOW);
    expect(summary).toMatchObject({ level: "warn", mixed: true, from: "2026-04-21", to: "2026-09-24" });
    expect(summary.snapshot).toEqual(["industry_白酒"]);
    expect(summary.spreadDays).toBe(156);
  });

  it("is informational when only a fallback source was used, and silent when all is live", () => {
    expect(summarizeFreshness([price("p1", "live_fallback")], new Set(), NOW).level).toBe("info");
    expect(summarizeFreshness([price("p1"), price("p2")], new Set(), NOW).level).toBe("none");
  });

  it("ignores old documents the answer does not cite", () => {
    const old: EvidenceSource = { evidence_id: "news_old", kind: "document", source_type: "news", as_of: "2025-01-01" };
    expect(summarizeFreshness([price("p1"), old], new Set(["p1"]), NOW).level).toBe("none");
    expect(summarizeFreshness([price("p1"), old], new Set(["news_old"]), NOW).level).toBe("warn");
  });
});

describe("summarizeFreshness: valuation dates (round 15, review-11 K17)", () => {
  it("includes the valuation date of P/E and P/B in the banner's date range", () => {
    // 2025-12-31 statements valued at the 2026-04-22 close, next to a 2026-04-21 industry row: the banner must not
    // end at 04-21 while the P/E tile says 估值日 2026-04-22.
    const fundamentals: EvidenceSource = {
      evidence_id: "fundamental_600519.SH",
      kind: "structured",
      source_type: "fundamental_sql",
      as_of: "2025-12-31",
      payload: {
        report_date: "2025-12-31",
        valuation_date: "2026-04-22",
        provenance: { source: "offline_snapshot", mode: "snapshot", is_live: false, as_of: "2025-12-31", freshness: "stale" },
      },
    };
    const summary = summarizeFreshness([fundamentals, industrySnapshot], new Set(), NOW);
    expect(summary).toMatchObject({ from: "2025-12-31", to: "2026-04-22" });
  });

  it("reads a compact valuation date and ignores a missing one", () => {
    const compact: EvidenceSource = {
      evidence_id: "fundamental_x",
      kind: "structured",
      source_type: "fundamental_sql",
      as_of: "2025-12-31",
      payload: { valuation_date: "20260930" },
    };
    expect(summarizeFreshness([compact], new Set(), NOW)).toMatchObject({ from: "2025-12-31", to: "2026-09-30" });
    const none: EvidenceSource = { ...compact, payload: {} };
    expect(summarizeFreshness([none], new Set(), NOW)).toMatchObject({ from: "2025-12-31", to: "2025-12-31" });
  });
});
