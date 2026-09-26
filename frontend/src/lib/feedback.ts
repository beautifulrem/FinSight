import { readStorage, STORAGE_KEYS, writeStorage } from "./storage";
import type { FeedbackRating } from "./types";

/**
 * Per-trace feedback remembered in this browser, so a rating survives reloads and re-renders.
 * `state` records what the server said: `sent` (stored with the trace), `local` (the server has no
 * feedback endpoint, or no longer knows the trace), or `error` (network/other failure; can retry).
 */
export interface FeedbackRecord {
  rating: FeedbackRating;
  comment?: string;
  state: "sent" | "local" | "error";
  reason?: "unsupported" | "unknown_trace";
  /** Set when stored; the newest entries are kept. */
  at?: number;
}

const MAX_ENTRIES = 300;

function readAll(): Record<string, FeedbackRecord> {
  try {
    const parsed = JSON.parse(readStorage(STORAGE_KEYS.feedback, "{}")) as unknown;
    return parsed && typeof parsed === "object" ? (parsed as Record<string, FeedbackRecord>) : {};
  } catch {
    return {};
  }
}

export function readFeedback(traceId: string): FeedbackRecord | undefined {
  const record = readAll()[traceId];
  return record && (record.rating === "up" || record.rating === "down") ? record : undefined;
}

export function writeFeedback(traceId: string, record: FeedbackRecord): void {
  const all = readAll();
  all[traceId] = { ...record, at: Date.now() };
  // Keep the newest entries only.
  const entries = Object.entries(all).sort(([, a], [, b]) => (b.at ?? 0) - (a.at ?? 0)).slice(0, MAX_ENTRIES);
  writeStorage(STORAGE_KEYS.feedback, JSON.stringify(Object.fromEntries(entries)));
}
