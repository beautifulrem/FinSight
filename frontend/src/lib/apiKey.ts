// Where the UI keeps the API key (C19, round-3 review).
//
// Default: sessionStorage. The key lives only as long as the tab and is not shared with other tabs, so closing
// the tab forgets it. If sessionStorage is unavailable the key stays in memory for this page load.
//
// Opt-in "remember on this device": the key is also written to localStorage and survives restarts.
//
// Trade-off: both storages are readable by any script running on the page's origin, so neither protects the
// key from an XSS bug; sessionStorage only shortens how long a key sits on disk and in which tabs it is
// visible. An HttpOnly cookie would hide it from scripts, but would need a login endpoint and CSRF protection;
// see SECURITY.md.
//
// Keys saved by older builds (always localStorage, without the remember flag) are moved to sessionStorage and
// deleted from localStorage on first load.

export const API_KEY_STORAGE = "finsight.apiKey";
export const API_KEY_REMEMBER = "finsight.apiKey.remember";

let memoryKey = "";

function session(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function local(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function get(storage: Storage | null, key: string): string {
  try {
    return storage?.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function set(storage: Storage | null, key: string, value: string): boolean {
  if (!storage) return false;
  try {
    if (value) storage.setItem(key, value);
    else storage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

export interface StoredApiKey {
  key: string;
  remember: boolean;
}

export function loadApiKey(): StoredApiKey {
  const persistent = local();
  const remember = get(persistent, API_KEY_REMEMBER) === "1";
  const remembered = get(persistent, API_KEY_STORAGE);
  if (remembered && !remember) {
    // legacy: an older build stored every key in localStorage; keep it for this tab only
    set(persistent, API_KEY_STORAGE, "");
    if (!set(session(), API_KEY_STORAGE, remembered)) memoryKey = remembered;
    return { key: remembered, remember: false };
  }
  const key = get(session(), API_KEY_STORAGE) || (remember ? remembered : "") || memoryKey;
  return { key, remember };
}

export function saveApiKey(key: string, remember: boolean): void {
  const value = key.trim();
  memoryKey = "";
  if (!set(session(), API_KEY_STORAGE, value)) memoryKey = value;
  const persistent = local();
  if (remember && value) {
    set(persistent, API_KEY_STORAGE, value);
    set(persistent, API_KEY_REMEMBER, "1");
  } else {
    set(persistent, API_KEY_STORAGE, "");
    set(persistent, API_KEY_REMEMBER, "");
  }
}
