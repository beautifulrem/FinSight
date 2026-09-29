// localStorage can throw (private mode, disabled storage); the UI keeps working without it.
// The API key is not kept here: see apiKey.ts (sessionStorage unless the user opts in to remembering it).
export const STORAGE_KEYS = {
  mode: "finsight.mode",
  session: "finsight.session",
  theme: "finsight.theme",
  lang: "finsight.lang",
  feedback: "finsight.feedback",
  view: "finsight.view",
} as const;

export function readStorage(key: string, fallback = ""): string {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeStorage(key: string, value: string): void {
  try {
    if (value) window.localStorage.setItem(key, value);
    else window.localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}

export function newSessionId(): string {
  const random = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}${Math.random().toString(16).slice(2)}`;
  return random.replaceAll("-", "");
}
