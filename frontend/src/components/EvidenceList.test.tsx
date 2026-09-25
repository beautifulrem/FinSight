import { render, screen } from "@testing-library/react";

import { EvidenceList } from "./EvidenceList";

describe("EvidenceList", () => {
  beforeAll(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("numbers sources, marks cited ones and only links http(s) urls", () => {
    render(
      <EvidenceList
        sources={[
          { evidence_id: "news_1", kind: "document", source_type: "news", title: "年报", source_url: "https://example.com/a" },
          { evidence_id: "bad_1", kind: "document", source_type: "news", title: "脚本", source_url: "javascript:alert(1)" },
          { evidence_id: "price_1", kind: "structured", source_type: "market_api", title: "茅台日线" },
        ]}
        cited={new Set(["news_1"])}
        highlight="price_1"
        highlightNonce={1}
      />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getAllByRole("link")).toHaveLength(1);
    expect(screen.getByRole("link")).toHaveAttribute("href", "https://example.com/a");
    const highlighted = screen.getByText("茅台日线").closest("li");
    expect(highlighted).toHaveAttribute("aria-current", "true");
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });
});
