import type { AgentResponse, SessionTurn } from "./types";

/**
 * (round 18, review-14 N15) The answers of this browser's current session, kept so a reload can show the restored turns
 * as they were answered: KPI tiles, chart, citations and the inspector, and the question as the user typed it.
 *
 * The server's session record (`GET /agent/sessions/{id}`) keeps a compact memory per turn: the question in the
 * normalised (simplified-character) form the pipeline read, and the first 600 characters of the answer, without the
 * evidence payloads the tiles and the inspector are drawn from. The full responses therefore stay in the browser that
 * received them, one session at a time; a turn without a stored response (another browser, storage full or
 * disabled) falls back to the compact text view.
 */

export const ARCHIVE_KEY = "finsight.turns";
/** At most this many answers are kept (the newest); older restored turns fall back to the compact view. */
export const MAX_ARCHIVED_TURNS = 20;

export interface ArchivedTurn {
  /** The question as the user typed it (traditional characters, spacing and all). */
  query: string;
  via: "stream" | "resume";
  response: AgentResponse;
}

interface Archive {
  session_id: string;
  turns: ArchivedTurn[];
}

function read(): Archive | null {
  try {
    const raw = window.localStorage.getItem(ARCHIVE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Archive;
    return parsed && typeof parsed.session_id === "string" && Array.isArray(parsed.turns) ? parsed : null;
  } catch {
    return null;
  }
}

/** The stored answers of `sessionId` (none when the archive belongs to another session). */
export function loadArchive(sessionId: string): ArchivedTurn[] {
  const archive = read();
  return archive && archive.session_id === sessionId ? archive.turns : [];
}

/** Remember a finished answer. Another session's archive is replaced; when storage is full the oldest answers go. */
export function archiveTurn(sessionId: string, turn: ArchivedTurn): void {
  if (!sessionId || turn.response.status !== "ok") return;
  let turns = [...loadArchive(sessionId), turn].slice(-MAX_ARCHIVED_TURNS);
  while (turns.length) {
    try {
      window.localStorage.setItem(ARCHIVE_KEY, JSON.stringify({ session_id: sessionId, turns } satisfies Archive));
      return;
    } catch {
      turns = turns.slice(1); // quota: drop the oldest and try again
    }
  }
  clearArchive();
}

export function clearArchive(): void {
  try {
    window.localStorage.removeItem(ARCHIVE_KEY);
  } catch {
    /* storage disabled */
  }
}

/**
 * For each server turn, the stored answer it was (or `null`). A stored answer matches the turn at its `turn_index`
 * whose remembered answer text starts it (the server keeps the first 600 characters), so an archive that is out of
 * step with the server (a turn answered in another tab, a server that lost the session) never puts an answer under
 * the wrong question.
 */
export function matchArchived(turns: SessionTurn[], archive: ArchivedTurn[]): (ArchivedTurn | null)[] {
  const used = new Set<number>();
  return turns.map((turn, index) => {
    const head = turn.answer ?? "";
    const at = archive.findIndex((entry, position) => {
      if (used.has(position)) return false;
      const answer = entry.response.answer ?? "";
      if (entry.response.turn_index != null && entry.response.turn_index !== index) return false;
      if (entry.response.turn_index == null && !head) return false;
      return head ? answer.startsWith(head) : !answer;
    });
    if (at < 0) return null;
    used.add(at);
    return archive[at]!;
  });
}
