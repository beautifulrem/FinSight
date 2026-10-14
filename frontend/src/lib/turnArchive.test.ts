import { act, renderHook } from "@testing-library/react";

import { useChat } from "@/hooks/useChat";

import { ARCHIVE_KEY, archiveTurn, loadArchive, matchArchived, MAX_ARCHIVED_TURNS } from "./turnArchive";
import type { AgentResponse } from "./types";

const reply = (index: number, answer: string, session = "s1"): AgentResponse => ({
  status: "ok",
  session_id: session,
  turn_index: index,
  answer,
});

beforeEach(() => window.localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("stored answers of the current session (round 18, N15)", () => {
  it("keeps one session's answers; a new session replaces them", () => {
    archiveTurn("s1", { query: "茅臺市盈率", via: "stream", response: reply(0, "a") });
    archiveTurn("s1", { query: "那ROE呢", via: "stream", response: reply(1, "b") });
    expect(loadArchive("s1").map((entry) => entry.query)).toEqual(["茅臺市盈率", "那ROE呢"]);
    archiveTurn("s2", { query: "比亞迪", via: "stream", response: reply(0, "c", "s2") });
    expect(loadArchive("s1")).toEqual([]);
    expect(loadArchive("s2")).toHaveLength(1);
  });

  it("does not store clarification requests", () => {
    archiveTurn("s1", { query: "它呢", via: "stream", response: { status: "needs_clarification", session_id: "s1" } });
    expect(window.localStorage.getItem(ARCHIVE_KEY)).toBeNull();
  });

  it("keeps only the newest answers, and drops the oldest when storage is full", () => {
    for (let i = 0; i < MAX_ARCHIVED_TURNS + 3; i++) archiveTurn("s1", { query: `q${i}`, via: "stream", response: reply(i, `a${i}`) });
    const kept = loadArchive("s1");
    expect(kept).toHaveLength(MAX_ARCHIVED_TURNS);
    expect(kept[0]!.query).toBe("q3");

    const setItem = Storage.prototype.setItem;
    let calls = 0;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (this: Storage, key: string, value: string) {
      calls += 1;
      if (calls < 3) throw new DOMException("full", "QuotaExceededError");
      setItem.call(this, key, value);
    });
    archiveTurn("s1", { query: "new", via: "stream", response: reply(MAX_ARCHIVED_TURNS + 3, "n") });
    const after = loadArchive("s1");
    expect(after[after.length - 1]!.query).toBe("new");
    expect(after.length).toBe(MAX_ARCHIVED_TURNS - 2);
  });

  it("matches a stored answer to the server turn at its index whose answer it starts with", () => {
    const archive = [
      { query: "一", via: "stream" as const, response: reply(0, "长江电力收盘价 27.5 元。".repeat(60)) },
      { query: "二", via: "stream" as const, response: reply(1, "别的回答") },
    ];
    const server = [
      // the server keeps the first 600 characters
      { query: "一", answer: "长江电力收盘价 27.5 元。".repeat(60).slice(0, 600) },
      { query: "二", answer: "另一个回答" }, // answered elsewhere: not this stored one
      { query: "三", answer: "第三" },
    ];
    const matched = matchArchived(server, archive);
    expect(matched.map((entry) => entry?.query ?? null)).toEqual(["一", null, null]);
  });
});

describe("session ids come from the server (round 18, N9)", () => {
  it("first turn sends no session_id; the next sends the id the server issued", async () => {
    const bodies: Record<string, unknown>[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      const text = [
        ["session", { session_id: "srv1" }],
        ["answer", reply(bodies.length - 1, "ok", "srv1")],
        ["done", {}],
      ]
        .map(([event, data]) => `event: ${String(event)}\ndata: ${JSON.stringify(data)}\n\n`)
        .join("");
      return new Response(text, { status: 200, headers: { "Content-Type": "text/event-stream" } });
    });
    const onSession = vi.fn();
    const { result } = renderHook(() => useChat({ sessionId: null, mode: "auto", apiKey: "", onSession }));
    await act(async () => {
      await result.current.send("茅台的PE");
    });
    await act(async () => {
      await result.current.send("那ROE呢");
    });
    expect(bodies[0]).not.toHaveProperty("session_id");
    expect(bodies[1]!.session_id).toBe("srv1");
    expect(onSession).toHaveBeenCalledWith("srv1");
    // the answers are stored under the issued id, with the question as typed
    expect(loadArchive("srv1").map((entry) => entry.query)).toEqual(["茅台的PE", "那ROE呢"]);

    // a new conversation forgets the issued id
    act(() => result.current.reset());
    await act(async () => {
      await result.current.send("比亚迪");
    });
    expect(bodies[2]).not.toHaveProperty("session_id");
  });
});
