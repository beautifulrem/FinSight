// Like query_intelligence/agent/verifier.py `_CITATION`, but also accepts "[a, b]" lists.
const CITATION = /\[([^[\]\n]{2,320})\]/g;
const SEPARATORS = /[,，;；、]/;

export type Segment =
  | { type: "text"; text: string }
  | { type: "cite"; id: string; index: number }
  | { type: "invalid"; id: string };

/** Map evidence ids to 1-based display numbers (E1, E2, …) in evidence-panel order. */
export function evidenceIndex(ids: readonly string[]): Map<string, number> {
  const map = new Map<string, number>();
  ids.forEach((id) => {
    if (!map.has(id)) map.set(id, map.size + 1);
  });
  return map;
}

function looksLikeEvidenceId(value: string): boolean {
  return /[_.:]/.test(value) || /^E\d+$/i.test(value);
}

/**
 * Split answer text into plain text and citation segments. Brackets whose content is a known
 * evidence id become citation chips; id-like brackets that are not in the run are flagged as
 * invalid (the verifier reports them too); anything else stays as text.
 */
export function splitCitations(text: string, index: Map<string, number>): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  const pushText = (value: string) => {
    if (!value) return;
    const previous = segments[segments.length - 1];
    if (previous?.type === "text") previous.text += value;
    else segments.push({ type: "text", text: value });
  };
  // Evidence ids are case-sensitive, but a model (or an old compliance pass) may change their case
  // ("fundamental_600519.sh"); such a citation still points at exactly one item, so resolve it to that item.
  const folded = new Map<string, string>();
  for (const id of index.keys()) folded.set(id.toLowerCase(), folded.has(id.toLowerCase()) ? "" : id);
  const resolve = (id: string) => (index.has(id) ? id : folded.get(id.toLowerCase()) || id);
  for (const match of text.matchAll(CITATION)) {
    const start = match.index ?? 0;
    const inner = match[1] ?? "";
    const ids = inner.split(SEPARATORS).map((part) => resolve(part.trim())).filter(Boolean);
    // Every part must be an evidence id; otherwise this is ordinary bracketed prose.
    if (!ids.length || ids.some((id) => /\s/.test(id) || (!index.has(id) && !looksLikeEvidenceId(id)))) continue;
    pushText(text.slice(last, start).replace(/\s+$/, (space) => (space.includes("\n") ? space : "")));
    for (const id of ids) {
      const number = index.get(id);
      if (number !== undefined) segments.push({ type: "cite", id, index: number });
      else if (looksLikeEvidenceId(id)) segments.push({ type: "invalid", id });
    }
    last = start + match[0].length;
  }
  pushText(text.slice(last));
  return segments;
}

/** Remove citation markers, e.g. for copying plain text. */
export function stripCitations(text: string): string {
  const isCitation = (inner: string) =>
    inner.split(SEPARATORS).every((part) => part.trim() && !/\s/.test(part.trim()) && looksLikeEvidenceId(part.trim()));
  return text.replace(CITATION, (match, inner: string) => (isCitation(inner) ? "" : match)).replace(/[ \t]+([。．.，,；;])/g, "$1");
}
