import { API_KEY_REMEMBER, API_KEY_STORAGE, loadApiKey, saveApiKey } from "./apiKey";

describe("API key storage (C19)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });
  afterEach(() => vi.restoreAllMocks());

  it("keeps the key in sessionStorage only by default", () => {
    saveApiKey(" k1 ", false);

    expect(window.sessionStorage.getItem(API_KEY_STORAGE)).toBe("k1");
    expect(window.localStorage.getItem(API_KEY_STORAGE)).toBeNull();
    expect(window.localStorage.getItem(API_KEY_REMEMBER)).toBeNull();
    expect(loadApiKey()).toEqual({ key: "k1", remember: false });
  });

  it("forgets the key when the tab's session storage is gone", () => {
    saveApiKey("k1", false);
    window.sessionStorage.clear(); // a new tab or a restarted browser

    expect(loadApiKey()).toEqual({ key: "", remember: false });
  });

  it("persists to localStorage only when the user opts in, and unticking removes it", () => {
    saveApiKey("k2", true);
    window.sessionStorage.clear();

    expect(window.localStorage.getItem(API_KEY_STORAGE)).toBe("k2");
    expect(loadApiKey()).toEqual({ key: "k2", remember: true });

    saveApiKey("k2", false);
    expect(window.localStorage.getItem(API_KEY_STORAGE)).toBeNull();
    expect(window.localStorage.getItem(API_KEY_REMEMBER)).toBeNull();
    expect(loadApiKey()).toEqual({ key: "k2", remember: false }); // still usable in this tab
  });

  it("clearing the key clears it everywhere", () => {
    saveApiKey("k3", true);
    saveApiKey("", true);

    expect(window.localStorage.getItem(API_KEY_STORAGE)).toBeNull();
    expect(window.sessionStorage.getItem(API_KEY_STORAGE)).toBeNull();
    expect(loadApiKey()).toEqual({ key: "", remember: false });
  });

  it("moves a key saved by an older build out of localStorage", () => {
    window.localStorage.setItem(API_KEY_STORAGE, "legacy-key"); // no remember flag: the old default

    expect(loadApiKey()).toEqual({ key: "legacy-key", remember: false });
    expect(window.localStorage.getItem(API_KEY_STORAGE)).toBeNull();
    expect(window.sessionStorage.getItem(API_KEY_STORAGE)).toBe("legacy-key");
  });

  it("falls back to memory when sessionStorage is unavailable", () => {
    vi.spyOn(window, "sessionStorage", "get").mockImplementation(() => {
      throw new Error("SecurityError");
    });

    saveApiKey("k4", false);

    expect(window.localStorage.getItem(API_KEY_STORAGE)).toBeNull();
    expect(loadApiKey()).toEqual({ key: "k4", remember: false });
  });
});
