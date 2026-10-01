import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import App from "./App";
import { STORAGE_KEYS } from "./lib/storage";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const TURNS = [
  { query: "五粮液的ROE是多少", answer: "五粮液 ROE 29.4% [fundamental_000858.SZ]。" },
  { query: "茅台的呢", answer: "贵州茅台 ROE 33%。" },
  { query: "两者差几个点", answer: "差 3.6 个百分点。" },
  { query: "那PE呢", answer: "五粮液 20.9 倍，贵州茅台 24.6 倍。" },
  { query: "谁更高", answer: "贵州茅台更高。" },
];

function serve(session: { turns: typeof TURNS } | null) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/health")) return json(200, { status: "ok" });
    if (url.startsWith("/agent/sessions/")) {
      return session ? json(200, { session_id: "s1", turns: session.turns, pending_clarification: null }) : json(404, { detail: "not found" });
    }
    return json(404, { detail: "not found" });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  window.localStorage.clear();
  // jsdom has no layout: the conversation's follow-the-bottom scrolling is a no-op here
  Element.prototype.scrollTo ??= function scrollTo() {};
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("restored session (round 12, H14)", () => {
  it("shows the restored turns under the banner that counts them, and lets the reader fold them away", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "s1");
    serve({ turns: TURNS });
    render(<App />);
    const banner = await screen.findByText("已恢复本会话的 5 轮历史");
    const history = banner.closest("details")!;
    expect(history).toHaveAttribute("open");
    const items = within(history).getAllByRole("listitem");
    expect(items).toHaveLength(5);
    for (const [i, turn] of TURNS.entries()) {
      expect(items[i]).toHaveTextContent(turn.query);
      expect(within(items[i]!).getByText(turn.query)).toBeVisible();
    }
    // citations are not shown in the compact history
    expect(history).not.toHaveTextContent("[fundamental_000858.SZ]");

    await userEvent.click(within(history).getByText("收起"));
    expect(history).not.toHaveAttribute("open");
    expect(within(history).getByText("展开")).toBeInTheDocument();
  });

  it("shows no banner when the server remembers no turns", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "s1");
    const fetchMock = serve(null);
    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/agent/sessions/s1"), expect.anything());
    expect(screen.queryByText(/已恢复/)).not.toBeInTheDocument();
    expect(document.querySelector(".session-history")).toBeNull();
  });
});

describe("system notices follow the language (round 12, H14)", () => {
  it("re-renders the new-session notice after switching zh -> en -> zh", async () => {
    serve(null);
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "新会话" }));
    const zh = "已开始新会话，之前的问题不再作为上下文。";
    const en = "Started a new session; earlier questions are no longer used as context.";
    expect(screen.getByText(zh)).toHaveClass("notice");

    await user.click(document.getElementById("lang-toggle")!);
    expect(screen.getByText(en)).toHaveClass("notice");
    expect(screen.queryByText(zh)).not.toBeInTheDocument();

    await user.click(document.getElementById("lang-toggle")!);
    expect(screen.getByText(zh)).toBeInTheDocument();
    expect(screen.queryByText(en)).not.toBeInTheDocument();
  });
});
