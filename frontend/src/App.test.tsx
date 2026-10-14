import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import App from "./App";
import { STORAGE_KEYS } from "./lib/storage";
import { archiveTurn, loadArchive } from "./lib/turnArchive";
import type { AgentResponse, SessionTurn } from "./lib/types";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const TURNS = [
  { query: "五粮液的ROE是多少", answer: "五粮液 ROE 29.4% [fundamental_000858.SZ]。" },
  { query: "茅台的呢", answer: "贵州茅台 ROE 33%。" },
  { query: "两者差几个点", answer: "差 3.6 个百分点。" },
  { query: "那PE呢", answer: "五粮液 20.9 倍，贵州茅台 24.6 倍。" },
  { query: "谁更高", answer: "贵州茅台更高。" },
];

function serve(session: { turns: SessionTurn[] } | null) {
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

/** An answer the browser received before the reload, with a price payload the KPI tiles are drawn from. */
const TYPED = "長江電力收盤價多少";
const NORMALISED = "长江电力收盘价多少";
const PRICE_ANSWER: AgentResponse = {
  status: "ok",
  session_id: "s1",
  trace_id: "trace-cjdl",
  turn_index: 0,
  query: NORMALISED,
  route: "workflow",
  answer: "长江电力最新收盘价 27.5 元 [price_600900.SH]。",
  evidence_used: ["price_600900.SH"],
  evidence_sources: [
    {
      evidence_id: "price_600900.SH",
      kind: "structured",
      source_type: "market_api",
      as_of: "2026-04-22",
      payload: {
        name: "长江电力",
        close: 27.5,
        pct_change_1d: 1.1,
      },
    },
  ],
};

describe("restored turns are drawn from the answers this browser received (round 18, N15)", () => {
  it("shows the tiles, the inspector and the question as typed, not the server's simplified form", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "s1");
    archiveTurn("s1", { query: TYPED, via: "stream", response: PRICE_ANSWER });
    serve({ turns: [{ query: NORMALISED, answer: PRICE_ANSWER.answer! }] });
    render(<App />);
    const bubble = await screen.findByText(TYPED, { selector: ".user-message" });
    expect(bubble).toBeInTheDocument();
    expect(screen.queryByText(NORMALISED, { selector: ".user-message" })).not.toBeInTheDocument();
    // a full answer card with its KPI tiles, not the compact text history
    expect(document.querySelector(".session-history")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
    expect(document.querySelectorAll(".kpi-tile").length).toBeGreaterThan(0);
    // the inspector describes the restored turn instead of its empty state
    const inspector = document.querySelector(".inspector")!;
    expect(inspector).toHaveTextContent(`对应问题：${TYPED}`);
    expect(inspector).not.toHaveTextContent("提问后，这里会显示证据来源");
    expect(screen.getByText("已恢复本会话的 1 轮历史")).toBeInTheDocument();
  });

  it("falls back to the compact history for a turn whose answer is not stored here, showing the typed text", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "s1");
    // stored for another session: not used
    archiveTurn("other", { query: TYPED, via: "stream", response: { ...PRICE_ANSWER, session_id: "other" } });
    serve({ turns: [{ query: NORMALISED, typed_query: TYPED, answer: PRICE_ANSWER.answer! }] });
    render(<App />);
    const history = (await screen.findByText("已恢复本会话的 1 轮历史")).closest("details")!;
    expect(history).toHaveTextContent(TYPED);
    expect(history).not.toHaveTextContent(NORMALISED);
    expect(document.querySelectorAll(".kpi-tile")).toHaveLength(0);
  });

  it("never puts a stored answer under a different question (answer text must match)", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "s1");
    archiveTurn("s1", { query: TYPED, via: "stream", response: PRICE_ANSWER });
    serve({ turns: [{ query: "茅台的PE", answer: "贵州茅台 PE 24.6 倍。" }] });
    render(<App />);
    const history = (await screen.findByText("已恢复本会话的 1 轮历史")).closest("details")!;
    expect(history).toHaveTextContent("茅台的PE");
    expect(screen.queryByText(TYPED)).not.toBeInTheDocument();
  });
});

describe("session ids are issued by the server (round 18, N9)", () => {
  function streamServer() {
    const bodies: Record<string, unknown>[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/health")) return json(200, { status: "ok" });
      if (url === "/agent/chat/stream") {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
        bodies.push(body);
        const answer = { ...PRICE_ANSWER, session_id: "issued1", turn_index: bodies.length - 1 };
        const events = [
          ["session", { session_id: "issued1" }],
          ["answer", answer],
          ["done", { session_id: "issued1" }],
        ] as const;
        const text = events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("");
        return new Response(text, { status: 200, headers: { "Content-Type": "text/event-stream" } });
      }
      return json(404, { detail: "not found" });
    });
    vi.stubGlobal("fetch", fetchMock);
    return { fetchMock, bodies };
  }

  it("sends no session id on the first turn and the issued one on the next", async () => {
    const { bodies } = streamServer();
    const user = userEvent.setup();
    render(<App />);
    const box = screen.getByRole("textbox");
    await user.type(box, "长江电力收盘价{Enter}");
    await screen.findAllByRole("article");
    await user.type(box, "那PE呢{Enter}");
    await vi.waitFor(() => expect(bodies).toHaveLength(2));
    expect(bodies[0]).not.toHaveProperty("session_id");
    expect(bodies[1]!.session_id).toBe("issued1");
    expect(window.localStorage.getItem(STORAGE_KEYS.session)).toBe("issued1");
  });

  it("forgets a stored session id the server does not know and starts afresh", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "gone1");
    archiveTurn("gone1", { query: TYPED, via: "stream", response: { ...PRICE_ANSWER, session_id: "gone1" } });
    const { fetchMock, bodies } = streamServer();
    const user = userEvent.setup();
    render(<App />);
    await vi.waitFor(() =>
      expect(fetchMock.mock.calls.some(([url]) => String(url).startsWith("/agent/sessions/gone1"))).toBe(true),
    );
    await vi.waitFor(() => expect(window.localStorage.getItem(STORAGE_KEYS.session)).toBeNull());
    expect(loadArchive("gone1")).toEqual([]);
    await user.type(screen.getByRole("textbox"), "长江电力收盘价{Enter}");
    await vi.waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).not.toHaveProperty("session_id");
  });
});

describe("session id persistence (round 12)", () => {
  it("keeps no session id for a page with no turns, so a reload does not ask the server for it (404)", async () => {
    const fetchMock = serve(null);
    const user = userEvent.setup();
    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(window.localStorage.getItem(STORAGE_KEYS.session)).toBeNull();
    await user.click(screen.getByRole("button", { name: "新会话" }));
    expect(window.localStorage.getItem(STORAGE_KEYS.session)).toBeNull();
    expect(fetchMock.mock.calls.some(([url]) => String(url).startsWith("/agent/sessions/"))).toBe(false);
  });

  it("keeps the id of a restored session and forgets it on New session", async () => {
    window.localStorage.setItem(STORAGE_KEYS.session, "s1");
    serve({ turns: TURNS });
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("已恢复本会话的 5 轮历史");
    expect(window.localStorage.getItem(STORAGE_KEYS.session)).toBe("s1");
    await user.click(screen.getByRole("button", { name: "新会话" }));
    expect(window.localStorage.getItem(STORAGE_KEYS.session)).toBeNull();
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
