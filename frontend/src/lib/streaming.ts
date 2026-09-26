import { stripCitations } from "./citations";

/** Text the reader sees while the answer streams: citation markers (complete or half-received) are hidden. */
export function streamingText(draft: string): string {
  return stripCitations(draft).replace(/\[[^\]\n]*$/, "");
}

const comparable = (text: string) => stripCitations(text).replace(/\s+/g, "");

/** True when verification (or compliance) changed the streamed draft before it became final. */
export function answerEdited(draft: string | undefined, final: string): boolean {
  return Boolean(draft?.trim()) && comparable(draft ?? "") !== comparable(final);
}
