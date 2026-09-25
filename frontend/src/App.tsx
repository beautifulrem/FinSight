import { Dialog } from "radix-ui";
import { domAnimation, LazyMotion, MotionConfig } from "motion/react";
import { X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Composer, type ComposerHandle } from "@/components/Composer";
import { EmptyState } from "@/components/EmptyState";
import { Header, type AppStatus } from "@/components/Header";
import { HistoryView } from "@/components/HistoryView";
import { Inspector, type InspectorTab } from "@/components/Inspector";
import { SettingsDialog, type ThemePref } from "@/components/SettingsDialog";
import { TurnView } from "@/components/TurnView";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useChat, type Turn } from "@/hooks/useChat";
import { fetchHealth, fetchSession } from "@/lib/api";
import { I18nContext, makeTranslate, type Lang } from "@/lib/i18n";
import { newSessionId, readStorage, STORAGE_KEYS, writeStorage } from "@/lib/storage";
import type { UiMode } from "@/lib/types";
import { answerView, type AnswerView } from "@/lib/view";

const MODES: UiMode[] = ["auto", "agent", "workflow", "classic"];

function meta(name: string): string {
  const value = document.querySelector<HTMLMetaElement>(`meta[name="finsight:${name}"]`)?.content ?? "";
  return value.includes("{{") ? "" : value;
}

function systemDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

function isNarrow(): boolean {
  return window.matchMedia?.("(max-width: 1023px)").matches ?? false;
}

interface InspectState {
  turnId: string | null;
  tab: InspectorTab;
  highlight: string | null;
  nonce: number;
}

export default function App() {
  const [lang, setLang] = useState<Lang>(() => (readStorage(STORAGE_KEYS.lang) === "en" ? "en" : "zh"));
  const [themePref, setThemePref] = useState<ThemePref>(() => {
    const stored = readStorage(STORAGE_KEYS.theme);
    return stored === "light" || stored === "dark" ? stored : "system";
  });
  const [prefersDark, setPrefersDark] = useState(systemDark);
  const [mode, setMode] = useState<UiMode>(() => {
    const stored = readStorage(STORAGE_KEYS.mode) as UiMode;
    return MODES.includes(stored) ? stored : "auto";
  });
  const [apiKey, setApiKey] = useState(() => readStorage(STORAGE_KEYS.apiKey));
  const [sessionId, setSessionId] = useState(() => readStorage(STORAGE_KEYS.session) || newSessionId());
  const [online, setOnline] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [inspect, setInspect] = useState<InspectState>({ turnId: null, tab: "evidence", highlight: null, nonce: 0 });
  const composer = useRef<ComposerHandle>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const i18n = useMemo(() => ({ lang, t: makeTranslate(lang) }), [lang]);
  const { t } = i18n;
  const dark = themePref === "dark" || (themePref === "system" && prefersDark);
  const config = useMemo(() => ({ placeholder: meta("placeholder"), submitText: meta("submit-text") }), []);

  const chat = useChat({ sessionId, mode, apiKey });
  const { items, pending, busy, send, stop, reset, setPending, restore } = chat;

  // Preferences ------------------------------------------------------------------------------
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);
  useEffect(() => {
    const query = window.matchMedia?.("(prefers-color-scheme: dark)");
    const listener = (event: MediaQueryListEvent) => setPrefersDark(event.matches);
    query?.addEventListener("change", listener);
    return () => query?.removeEventListener("change", listener);
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    writeStorage(STORAGE_KEYS.lang, lang);
  }, [lang]);
  useEffect(() => writeStorage(STORAGE_KEYS.theme, themePref === "system" ? "" : themePref), [themePref]);
  useEffect(() => writeStorage(STORAGE_KEYS.mode, mode), [mode]);
  useEffect(() => writeStorage(STORAGE_KEYS.session, sessionId), [sessionId]);

  // Server health and session restore ---------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await fetchHealth();
      if (!cancelled) setOnline(ok);
    };
    void check();
    const timer = window.setInterval(check, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const restored = useRef(false);
  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    const controller = new AbortController();
    fetchSession(sessionId, { apiKey, signal: controller.signal })
      .then((info) => restore(info.turns ?? [], info.pending_clarification ?? null))
      .catch(() => undefined);
    return () => controller.abort();
  }, [sessionId, apiKey, restore]);

  // Scrolling: follow new content unless the reader scrolled up --------------------------------
  const onScroll = () => {
    const element = scroller.current;
    if (!element) return;
    stick.current = element.scrollHeight - element.scrollTop - element.clientHeight < 120;
  };
  useEffect(() => {
    const element = scroller.current;
    if (element && stick.current) element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [items]);

  // Views --------------------------------------------------------------------------------------
  const turns = items.filter((item): item is Turn => item.kind === "turn");
  const views = useMemo(() => {
    const map = new Map<string, AnswerView | null>();
    for (const turn of turns) map.set(turn.id, answerView(turn));
    return map;
  }, [turns]);
  const latestAnswered = [...turns].reverse().find((turn) => views.get(turn.id));
  const inspected = (inspect.turnId && turns.find((turn) => turn.id === inspect.turnId)) || latestAnswered;
  const inspectedView = inspected ? views.get(inspected.id) : null;
  const lastTurn = turns[turns.length - 1];

  const status: AppStatus = busy
    ? "running"
    : !online
      ? "offline"
      : pending && mode !== "classic"
        ? "waiting"
        : lastTurn?.status === "error"
          ? "error"
          : "ready";

  // Actions ------------------------------------------------------------------------------------
  const ask = useCallback(
    (query: string) => {
      stick.current = true;
      setInspect((state) => ({ ...state, turnId: null, highlight: null }));
      void send(query);
    },
    [send],
  );

  const onCite = useCallback((turnId: string, evidenceId: string) => {
    setInspect((state) => ({ turnId, tab: "evidence", highlight: evidenceId, nonce: state.nonce + 1 }));
    if (isNarrow()) setSheetOpen(true);
  }, []);

  const onInspect = useCallback((turnId: string, tab: InspectorTab) => {
    setInspect((state) => ({ ...state, turnId, tab, highlight: null }));
    if (isNarrow()) setSheetOpen(true);
  }, []);

  const onRetry = useCallback((turn: Turn) => ask(turn.query), [ask]);

  const newSession = () => {
    const id = newSessionId();
    setSessionId(id);
    reset(t("answer.newSession"));
    setInspect({ turnId: null, tab: "evidence", highlight: null, nonce: 0 });
    composer.current?.focus();
  };

  const onEvidence = (id: string) => inspected && onCite(inspected.id, id);

  const inspector = (
    <Inspector
      turn={inspected}
      view={inspectedView}
      tab={inspect.tab}
      onTab={(tab) => setInspect((state) => ({ ...state, tab }))}
      highlight={inspected?.id === inspect.turnId ? inspect.highlight : null}
      highlightNonce={inspect.nonce}
      sessionId={sessionId}
      onEvidence={onEvidence}
      className="h-full"
    />
  );

  const empty = items.length === 0;

  return (
    <I18nContext.Provider value={i18n}>
      <LazyMotion features={domAnimation} strict>
      <MotionConfig reducedMotion="user">
        <TooltipProvider delayDuration={300}>
          <div className="flex h-dvh flex-col">
            <a
              href="#query-input"
              className="sr-only z-50 rounded-md bg-cobalt px-3 py-2 text-cobalt-ink focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
            >
              {t("a11y.skip")}
            </a>
            <Header
              status={status}
              dark={dark}
              onToggleTheme={() => setThemePref(dark ? "light" : "dark")}
              onToggleLang={() => setLang(lang === "zh" ? "en" : "zh")}
              onNewSession={newSession}
              onSettings={() => setSettingsOpen(true)}
              onInspector={() => setSheetOpen(true)}
              canInspect={Boolean(inspectedView)}
            />
            <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_minmax(340px,420px)]">
              <main className="flex min-h-0 flex-col">
                <div ref={scroller} onScroll={onScroll} className="scrollbar-thin min-h-0 flex-1 overflow-y-auto">
                  <div id="chat-messages" className="mx-auto w-full max-w-3xl space-y-6 px-3 py-5 sm:px-5 sm:py-8" aria-live="polite">
                    {empty ? (
                      <EmptyState onAsk={ask} />
                    ) : (
                      items.map((item) =>
                        item.kind === "notice" ? (
                          <p key={item.id} className="notice text-center text-[12.5px] text-faint">
                            {item.text}
                          </p>
                        ) : item.kind === "history" ? (
                          <HistoryView key={item.id} turns={item.turns} />
                        ) : (
                          <TurnView
                            key={item.id}
                            turn={item}
                            view={views.get(item.id) ?? null}
                            isLast={item.id === lastTurn?.id}
                            activeEvidence={inspect.turnId === item.id ? inspect.highlight : null}
                            themeKey={dark ? "dark" : "light"}
                            onCite={onCite}
                            onInspect={onInspect}
                            onAsk={ask}
                            onRetry={onRetry}
                          />
                        ),
                      )
                    )}
                  </div>
                </div>
                <div className="mx-auto w-full max-w-3xl shrink-0 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-5">
                  <Composer
                    ref={composer}
                    busy={busy}
                    mode={mode}
                    onMode={setMode}
                    pending={pending}
                    onCancelPending={() => setPending(null)}
                    onSend={ask}
                    onStop={stop}
                    placeholder={config.placeholder}
                    submitText={config.submitText}
                  />
                  <p className="risk-footer mt-1.5 text-center text-[11.5px] text-faint">{t("disclaimer.footer")}</p>
                </div>
              </main>
              <aside
                className="hidden min-h-0 border-l border-line bg-surface/60 p-4 lg:block"
                aria-label={t("inspector.title")}
              >
                {inspector}
              </aside>
            </div>
          </div>

          <Dialog.Root open={sheetOpen} onOpenChange={setSheetOpen}>
            <Dialog.Portal>
              <Dialog.Overlay className="fixed inset-0 z-40 bg-ink/30 backdrop-blur-[2px] lg:hidden" />
              <Dialog.Content
                aria-describedby={undefined}
                className="inspector-sheet fixed inset-x-0 bottom-0 z-50 flex h-[82dvh] flex-col rounded-t-2xl border-t border-line bg-surface p-4 pb-[max(1rem,env(safe-area-inset-bottom))] shadow-2xl outline-none lg:hidden"
              >
                <div className="mb-3 flex items-center justify-between">
                  <Dialog.Title className="text-[15px] font-semibold">{t("inspector.title")}</Dialog.Title>
                  <Dialog.Close asChild>
                    <Button size="icon-sm" aria-label={t("inspector.close")}>
                      <X />
                    </Button>
                  </Dialog.Close>
                </div>
                <div className="min-h-0 flex-1">{inspector}</div>
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>

          <SettingsDialog
            open={settingsOpen}
            onOpenChange={setSettingsOpen}
            apiKey={apiKey}
            onApiKey={(key) => {
              setApiKey(key);
              writeStorage(STORAGE_KEYS.apiKey, key);
            }}
            sessionId={sessionId}
            lang={lang}
            onLang={setLang}
            theme={themePref}
            onTheme={setThemePref}
          />
        </TooltipProvider>
      </MotionConfig>
      </LazyMotion>
    </I18nContext.Provider>
  );
}
