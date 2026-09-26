import { sendFeedback } from "./api";

const body = { trace_id: "t1", session_id: "s1", rating: "up" as const, comment: null };

function mockFetch(status: number, json: unknown) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(json), { status }));
}

describe("sendFeedback", () => {
  afterEach(() => vi.restoreAllMocks());

  it("posts the rating to /agent/feedback", async () => {
    const spy = mockFetch(200, { ok: true });
    await expect(sendFeedback(body, { apiKey: "k" })).resolves.toBe("ok");
    const [url, init] = spy.mock.calls[0]!;
    expect(url).toBe("/agent/feedback");
    expect(JSON.parse(String(init?.body))).toEqual(body);
    expect((init?.headers as Record<string, string>)["X-API-Key"]).toBe("k");
  });

  it("degrades when the endpoint is missing or the trace is unknown", async () => {
    mockFetch(404, { detail: "Not Found" });
    await expect(sendFeedback(body, {})).resolves.toBe("unsupported");
    vi.restoreAllMocks();
    mockFetch(405, { detail: "Method Not Allowed" });
    await expect(sendFeedback(body, {})).resolves.toBe("unsupported");
    vi.restoreAllMocks();
    mockFetch(404, { detail: "unknown trace_id" });
    await expect(sendFeedback(body, {})).resolves.toBe("unknown_trace");
    vi.restoreAllMocks();
    mockFetch(500, { detail: "boom" });
    await expect(sendFeedback(body, {})).rejects.toThrow("boom");
  });
});
