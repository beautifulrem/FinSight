export interface RawSseEvent {
  event: string;
  data: string;
}

/**
 * Incremental Server-Sent Events parser. Feed decoded text chunks; complete events are returned
 * as soon as their terminating blank line arrives. Handles CRLF, multi-line `data:` fields and
 * comment lines (`: keep-alive`).
 */
export class SseParser {
  private buffer = "";

  push(chunk: string): RawSseEvent[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const events: RawSseEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
      boundary = this.buffer.indexOf("\n\n");
    }
    return events;
  }

  /** Flush a trailing event that was not followed by a blank line. */
  flush(): RawSseEvent[] {
    const rest = this.buffer.trim();
    this.buffer = "";
    const parsed = rest ? parseBlock(rest) : null;
    return parsed ? [parsed] : [];
  }
}

function parseBlock(block: string): RawSseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (!data.length) return null;
  return { event, data: data.join("\n") };
}
