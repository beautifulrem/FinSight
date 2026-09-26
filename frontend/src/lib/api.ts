import { SseParser } from "./sse";
import type {
  AgentMode,
  AgentResponse,
  ClassicResponse,
  FeedbackRequest,
  SessionInfo,
  StreamEvent,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface ApiOptions {
  apiKey?: string;
  signal?: AbortSignal;
}

function headers(apiKey?: string): HeadersInit {
  const result: Record<string, string> = { "Content-Type": "application/json", Accept: "application/json" };
  if (apiKey) result["X-API-Key"] = apiKey;
  return result;
}

async function errorFrom(response: Response): Promise<ApiError> {
  let message = `HTTP ${response.status}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail))
      message = body.detail
        .map((item: { msg?: string }) => item?.msg)
        .filter(Boolean)
        .join("; ");
    else if (body.detail) message = JSON.stringify(body.detail);
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(message, response.status);
}

async function postJson<T>(path: string, body: unknown, options: ApiOptions): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: headers(options.apiKey),
    body: JSON.stringify(body),
    signal: options.signal,
  });
  if (!response.ok) throw await errorFrom(response);
  return (await response.json()) as T;
}

/** Original pipeline: `POST /chat` with the default `mode` (workflow = legacy path). */
export function classicChat(query: string, options: ApiOptions): Promise<ClassicResponse> {
  return postJson<ClassicResponse>("/chat", { query }, options);
}

export function resumeClarification(sessionId: string, reply: string, options: ApiOptions): Promise<AgentResponse> {
  return postJson<AgentResponse>("/agent/resume", { session_id: sessionId, reply }, options);
}

export async function fetchSession(sessionId: string, options: ApiOptions): Promise<SessionInfo> {
  const response = await fetch(`/agent/sessions/${encodeURIComponent(sessionId)}`, {
    headers: headers(options.apiKey),
    signal: options.signal,
  });
  if (!response.ok) throw await errorFrom(response);
  return (await response.json()) as SessionInfo;
}

export type FeedbackResult = "ok" | "unsupported" | "unknown_trace";

/**
 * `POST /agent/feedback`. Servers without the endpoint answer 404 "Not Found" (or 405/501): that is
 * reported as `unsupported` so the UI keeps the rating locally instead of showing an error. A 404
 * with any other detail means the server no longer knows the trace (e.g. it restarted).
 */
export async function sendFeedback(body: FeedbackRequest, options: ApiOptions): Promise<FeedbackResult> {
  const response = await fetch("/agent/feedback", {
    method: "POST",
    headers: headers(options.apiKey),
    body: JSON.stringify(body),
    signal: options.signal,
  });
  if (response.ok) return "ok";
  if (response.status === 405 || response.status === 501) return "unsupported";
  const error = await errorFrom(response);
  if (response.status === 404) return error.message === "Not Found" ? "unsupported" : "unknown_trace";
  throw error;
}

export async function fetchHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch("/health", { signal });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * `POST /agent/chat/stream`: yields typed SSE events until the server closes the stream.
 * Events: session → step / tool_call / tool_result … → answer | clarification → done (error on failure).
 */
export async function* streamAgentChat(
  query: string,
  sessionId: string,
  mode: AgentMode,
  options: ApiOptions,
): AsyncGenerator<StreamEvent> {
  const response = await fetch("/agent/chat/stream", {
    method: "POST",
    headers: { ...headers(options.apiKey), Accept: "text/event-stream" },
    body: JSON.stringify({ query, session_id: sessionId, mode }),
    signal: options.signal,
  });
  if (!response.ok || !response.body) throw await errorFrom(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      const raw = done ? parser.flush() : parser.push(value);
      for (const item of raw) {
        try {
          yield { event: item.event, data: JSON.parse(item.data) } as StreamEvent;
        } catch {
          /* ignore malformed event payloads */
        }
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}
