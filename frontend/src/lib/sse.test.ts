import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses events split across chunks and CRLF line endings", () => {
    const parser = new SseParser();
    expect(parser.push('event: step\r\ndata: {"node":"guard_in"')).toEqual([]);
    const events = parser.push('}\r\n\r\nevent: done\ndata: {}\n\n');
    expect(events).toEqual([
      { event: "step", data: '{"node":"guard_in"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("joins multi-line data, skips comments and flushes a trailing event", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\n\ndata: a\ndata: b\n\n")).toEqual([{ event: "message", data: "a\nb" }]);
    parser.push("event: answer\ndata: {}");
    expect(parser.flush()).toEqual([{ event: "answer", data: "{}" }]);
  });
});
