import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { evidenceIndex } from "@/lib/citations";

import { RichText } from "./RichText";
import { TooltipProvider } from "./ui/tooltip";

describe("RichText", () => {
  it("renders citation chips that report the evidence id when clicked", async () => {
    const onCite = vi.fn();
    render(
      <TooltipProvider>
        <RichText
          text={"**走势**偏弱 [price_600519.SH]。\n- PE 24.6 [fundamental_600519.SH]"}
          cite={{ index: evidenceIndex(["price_600519.SH", "fundamental_600519.SH"]), titles: new Map(), onCite }}
        />
      </TooltipProvider>,
    );
    expect(screen.getByText("走势").tagName).toBe("STRONG");
    expect(screen.getByRole("listitem")).toHaveTextContent("PE 24.6");
    await userEvent.click(screen.getByRole("button", { name: /E2/ }));
    expect(onCite).toHaveBeenCalledWith("fundamental_600519.SH");
  });

  it("never interprets answer text as HTML", () => {
    const { container } = render(<RichText text={'<img src=x onerror="alert(1)"> 收盘价'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container).toHaveTextContent('<img src=x onerror="alert(1)"> 收盘价');
  });
});
