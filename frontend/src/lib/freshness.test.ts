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
